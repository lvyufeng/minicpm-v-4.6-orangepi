"""Reusable MiniCPM-V 4.6 session: the C++ engine binary
`minicpmv_hybrid_decode` runs in `--server` mode and handles every NPU op
(embedding lookup, vision tower, prefill, decode). Python only does
CPU-side prompt tokenization + image preprocessing via HF AutoProcessor
— no torch_npu dependency, no HF model on NPU. This avoids the 30-50s
torch_npu per-shape JIT compile that used to dominate first-call latency
in the chat UX.

Image inputs route the HF processor's `pixel_values` (naflex patch-strip
layout) + `target_sizes` straight through to the engine; the engine runs
the full SigLIP vision tower + merger and scatters the resulting image
features into the prompt at the `image_token_id` positions before
prefill.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

from transformers import AutoProcessor

# Repo root is .../src/python/minicpmv/__init__.py up 3 levels.
_THIS_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS_FILE))))

DEFAULT_MODEL_PATH = os.path.join(REPO_ROOT, "MiniCPM-V-4.6")
DEFAULT_ENGINE_BIN = os.path.join(REPO_ROOT, "build", "minicpmv_hybrid_decode")
DEFAULT_CUSTOM_OPP = os.path.join(REPO_ROOT, "custom_opp_install", "vendors", "customize")

# MiniCPM-V 4.6 LM has 24 layers arranged 3 linear : 1 full, repeating 6x.
LAYER_TYPES = [
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
] * 6
EOS_TOKEN_IDS = [248044, 248046]
VOCAB_SIZE = 248094
HIDDEN_SIZE = 1024
IMAGE_TOKEN_ID = 248056


def _dtype_fp16():
    # imported lazily so the module's only hard dep is `transformers`
    import torch
    return torch.float16


class MinicpmvSession:
    """Holds the HF processor (CPU-only) + a persistent engine subprocess.

    Construction loads the tokenizer + image processor via AutoProcessor (CPU
    only, no model weights, no NPU). The engine binary in --server mode owns
    the model on the NPU; per request the Python side only tokenizes the
    chat-template'd prompt (and CPU-preprocesses any images) and ships the
    integer token ids + raw pixel bytes to the engine.

    Pass with_vision=True to spawn the engine with vision weights loaded so
    image messages work. Text-only sessions can keep with_vision=False to
    save the ~1.5 GB NPU vision-weight load.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        engine_bin: Optional[str] = None,
        custom_opp: Optional[str] = None,
        with_vision: bool = True,
        max_slice_nums: int = 9,
    ) -> None:
        self.model_path = model_path or os.environ.get("MINICPMV_MODEL_PATH", DEFAULT_MODEL_PATH)
        self.engine_bin = engine_bin or os.environ.get("MINICPMV_ENGINE_BIN", DEFAULT_ENGINE_BIN)
        self.custom_opp = custom_opp or os.environ.get("MINICPMV_CUSTOM_OPP", DEFAULT_CUSTOM_OPP)
        self.with_vision = with_vision
        self.max_slice_nums = max_slice_nums

        # CPU-only processor: tokenizer + chat template + image processor.
        # Holds no model weights, never touches the NPU.
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        # Persistent engine subprocess in --server mode.
        self._engine_proc: Optional[subprocess.Popen] = None

    def close(self) -> None:
        proc = self._engine_proc
        self._engine_proc = None
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                try:
                    proc.stdin.write("quit\n")
                    proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                proc.stdin.close()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ----- Public API -------------------------------------------------------

    def generate(
        self,
        messages: List[Dict],
        max_new_tokens: int = 128,
        conversation_id: Optional[str] = None,
    ) -> Generator[Tuple[int, str, Dict], None, None]:
        """Stream the assistant turn for `messages`.

        Args:
            messages: HF chat-template format, e.g.
                [{"role": "user", "content": [{"type": "text", "text": "hi"}]},
                 {"role": "user", "content": [{"type": "image", "url": "/path/to.png"},
                                              {"type": "text", "text": "describe"}]},
                 ...]
            max_new_tokens: hard cap on generated tokens (engine will also stop on EOS).

        Yields:
            (token_id, decoded_text_chunk, stats) per token. Stats is a dict
            with token_count, elapsed_s, decode_s, tps for the current stream.
        """
        prepared = self._prepare_inputs(messages)
        bundle_json, side_paths = self._write_bundle(prepared, max_new_tokens, conversation_id)
        try:
            yield from self._run_engine_streaming(bundle_json)
        finally:
            for p in [bundle_json, *side_paths]:
                try:
                    os.remove(p)
                except OSError:
                    pass

    # ----- Internals --------------------------------------------------------

    def _prepare_inputs(self, messages: List[Dict]) -> Dict[str, Any]:
        """Run the HF processor (CPU) and return a normalised dict of bytes
        ready for the bundle:
            {
              'input_ids': List[int],
              'pixel_values': Optional[bytes],     # fp16 little-endian, naflex
              'target_h': Optional[int],
              'target_w': Optional[int],
            }
        Only the first image in the message stream is currently handled.
        """
        # Reject unsupported part types (audio/video) early.
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                for p in content:
                    if isinstance(p, dict):
                        kind = p.get("type")
                        if kind not in (None, "text", "image"):
                            raise NotImplementedError(
                                f"content part type {kind!r} not supported")

        has_image = any(
            isinstance(p, dict) and p.get("type") == "image"
            for m in messages
            if isinstance(m.get("content"), list)
            for p in m["content"]
        )
        if has_image and not self.with_vision:
            raise RuntimeError("session was constructed with with_vision=False but "
                               "messages contain an image; recreate the session "
                               "with with_vision=True")

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            # max_slice_nums controls MiniCPM-V's high-res slicing. The HF
            # processor splits one image into 1 overview + up to N-1
            # sub-images (each ~448×448, aspect-ratio-respecting); the engine
            # loops the vision tower per slice and stitches features.
            max_slice_nums=self.max_slice_nums,
        )
        input_ids = inputs["input_ids"]
        if hasattr(input_ids, "tolist"):
            ids_list = input_ids[0].tolist() if hasattr(input_ids[0], "tolist") else list(input_ids[0])
        else:
            ids_list = list(input_ids[0]) if isinstance(input_ids[0], (list, tuple)) else list(input_ids)
        ids_list = [int(x) for x in ids_list]

        result: Dict[str, Any] = {"input_ids": ids_list}
        pixel_values = inputs.get("pixel_values")
        target_sizes = inputs.get("target_sizes")
        if has_image and pixel_values is not None and target_sizes is not None:
            import numpy as np
            pv = pixel_values
            if hasattr(pv, "to") and hasattr(pv, "numpy"):
                pv_np = pv[:1].to("cpu").contiguous().to(_dtype_fp16()).numpy()
            else:
                pv_np = np.ascontiguousarray(pv).astype(np.float16)
            # target_sizes is [N_slices, 2] — list out each [h, w] pair.
            ts = target_sizes
            if hasattr(ts, "tolist"):
                ts_list = ts.tolist()
            else:
                ts_list = [list(r) for r in ts]
            slice_sizes = [[int(r[0]), int(r[1])] for r in ts_list]
            result["pixel_values"] = pv_np.tobytes()
            result["slice_sizes"] = slice_sizes
        return result

    def _write_bundle(
        self,
        prepared: Dict[str, Any],
        max_new_tokens: int,
        conversation_id: Optional[str],
    ) -> Tuple[str, List[str]]:
        input_ids = prepared["input_ids"]
        seq_len = len(input_ids)
        prefix = os.path.join(
            tempfile.gettempdir(),
            f"minicpmv_{os.getpid()}_{int(time.time() * 1000)}",
        )
        json_path = prefix + ".json"
        side_paths: List[str] = []

        bundle = {
            "version": 1,
            "conversation_id": conversation_id or "",
            "seq_len": seq_len,
            "hidden_size": HIDDEN_SIZE,
            # Reserve a large fixed budget per conversation so the engine
            # doesn't drop and rebuild the prefix cache every time the
            # request grows. We still bound this to the engine's pre-built
            # rope-table size (8192).
            "max_seq_len": min(8192, max(seq_len + max_new_tokens + 4, 2048)),
            "max_new_tokens": max_new_tokens,
            "vocab_size": VOCAB_SIZE,
            "eos_token_ids": EOS_TOKEN_IDS,
            "layer_types": LAYER_TYPES,
            "embeddings_dtype": "float16",
            "embeddings_endianness": "little",
            "embeddings_path": "",  # engine does embedding lookup from input_ids
            "weights_path": os.path.join(self.model_path, "model.safetensors"),
            "input_ids": input_ids,
        }
        if "pixel_values" in prepared:
            pixels_path = prefix + ".pixels.f16"
            with open(pixels_path, "wb") as f:
                f.write(prepared["pixel_values"])
            side_paths.append(pixels_path)
            bundle["image_pixels_path"] = pixels_path
            bundle["image_slice_sizes"] = prepared["slice_sizes"]
            bundle["image_token_id"] = IMAGE_TOKEN_ID

        with open(json_path, "w") as f:
            json.dump(bundle, f)
        return json_path, side_paths

    # ----- Internals --------------------------------------------------------

    def _prepare_inputs_embeds(self, *args, **kwargs):
        raise NotImplementedError(
            "_prepare_inputs_embeds was removed when the torch_npu dependency "
            "was dropped. Use _prepare_input_ids."
        )

    def _write_bundle_legacy(self, *args, **kwargs):
        raise NotImplementedError("legacy embeddings-file bundle writer removed")

    def _run_engine_streaming(
        self,
        bundle_json: str,
    ) -> Generator[Tuple[int, str, Dict], None, None]:
        proc = self._ensure_engine_server()
        assert proc.stdin is not None and proc.stdout is not None
        try:
            proc.stdin.write(bundle_json + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            self._engine_proc = None
            raise RuntimeError(f"engine subprocess pipe broken: {e}")

        start_time = time.time()
        first_token_time: Optional[float] = None
        count = 0
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("# done"):
                # Per-bundle terminator from the engine; this request is over.
                break
            if line.startswith("# error"):
                raise RuntimeError(f"engine: {line[len('# error'):].strip()}")
            if not line:
                continue
            try:
                tok = int(line)
            except ValueError:
                # Unknown line — log to stderr and continue.
                print(f"[engine] {line}", file=sys.stderr)
                continue
            if first_token_time is None:
                first_token_time = time.time()
            count += 1
            text = self.processor.tokenizer.decode(
                [tok], skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )
            elapsed = time.time() - start_time
            decode_elapsed = (time.time() - first_token_time) if first_token_time else 0.0
            tps = (count - 1) / decode_elapsed if decode_elapsed > 0 else 0.0
            yield tok, text, {
                "token_count": count,
                "elapsed_s": elapsed,
                "decode_s": decode_elapsed,
                "tps": tps,
            }

        # Sanity: if the engine died, raise so the caller sees it.
        if proc.poll() is not None:
            rc = proc.returncode
            err = ""
            if proc.stderr is not None:
                try:
                    err = proc.stderr.read()
                except Exception:
                    pass
            self._engine_proc = None
            raise RuntimeError(f"engine subprocess exited rc={rc}, stderr={err!r}")

    def _ensure_engine_server(self) -> subprocess.Popen:
        """Spawn the engine subprocess in --server mode if not already alive."""
        if self._engine_proc is not None and self._engine_proc.poll() is None:
            return self._engine_proc

        env = os.environ.copy()
        env["ASCEND_CUSTOM_OPP_PATH"] = self.custom_opp
        op_lib = self.custom_opp + "/op_api/lib"
        env["LD_LIBRARY_PATH"] = op_lib + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )

        weights_path = os.path.join(self.model_path, "model.safetensors")
        cmd = [self.engine_bin, "--server", "--weights", weights_path]
        if self.with_vision:
            cmd.append("--with-vision")
        load_hint = "~60s" if self.with_vision else "~30s"
        print(f"[minicpmv_session] spawning engine server (weights load {load_hint})…",
              file=sys.stderr, flush=True)
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,
        )

        # Block until the engine signals it is ready. Errors during weight
        # load surface as the process exiting before "# server_ready".
        assert proc.stderr is not None
        startup_log: list[str] = []
        while True:
            line = proc.stderr.readline()
            if not line:
                rc = proc.wait()
                joined = "".join(startup_log)
                raise RuntimeError(
                    f"engine server exited rc={rc} before ready. log:\n{joined}"
                )
            startup_log.append(line)
            if "# server_ready" in line:
                break

        # Forward post-startup engine stderr to ours so prefix-cache diagnostics
        # and any engine warnings are visible to the caller.
        def _drain_stderr() -> None:
            try:
                for line in iter(proc.stderr.readline, ""):
                    sys.stderr.write("[engine] " + line)
                    sys.stderr.flush()
            except Exception:
                pass

        threading.Thread(target=_drain_stderr, daemon=True).start()

        self._engine_proc = proc
        print("[minicpmv_session] engine server ready", file=sys.stderr, flush=True)
        return proc
