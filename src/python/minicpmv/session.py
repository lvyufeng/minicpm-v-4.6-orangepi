"""Reusable MiniCPM-V 4.6 session: the C++ engine binary
`minicpmv_hybrid_decode` runs in `--server` mode and handles every NPU op
(embedding lookup, prefill, decode). Python only does CPU-side prompt
tokenization via HF AutoProcessor — no torch_npu dependency, no HF model on
NPU. This avoids the 30-50s torch_npu per-shape JIT compile that used to
dominate first-call latency in the chat UX.

For multimodal (image) inputs the Python side currently raises — see TODO at
the top of `_prepare_inputs_embeds` for how to add CPU- or engine-side
vision support later.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from typing import Dict, Generator, List, Optional, Tuple

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


class MinicpmvSession:
    """Holds the HF processor (CPU-only) + a persistent engine subprocess.

    Construction loads the tokenizer + chat template via AutoProcessor (CPU
    only, no model weights, no NPU). The engine binary in --server mode owns
    the model on the NPU; per request the Python side only tokenizes the
    chat-template'd prompt and ships the integer token ids to the engine.

    Image inputs currently raise NotImplementedError — see the module
    docstring for how to add vision support later without re-introducing the
    torch_npu dependency.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        engine_bin: Optional[str] = None,
        custom_opp: Optional[str] = None,
    ) -> None:
        self.model_path = model_path or os.environ.get("MINICPMV_MODEL_PATH", DEFAULT_MODEL_PATH)
        self.engine_bin = engine_bin or os.environ.get("MINICPMV_ENGINE_BIN", DEFAULT_ENGINE_BIN)
        self.custom_opp = custom_opp or os.environ.get("MINICPMV_CUSTOM_OPP", DEFAULT_CUSTOM_OPP)

        # CPU-only tokenizer + chat template. AutoProcessor downloads the
        # image processor config too, but we don't touch any NPU op here.
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        # Persistent engine subprocess in --server mode. Lazily spawned on
        # first generate() call so __init__ doesn't trip on a missing binary
        # during e.g. dataset prep.
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
    ) -> Generator[Tuple[int, str, Dict], None, None]:
        """Stream the assistant turn for `messages`.

        Args:
            messages: HF chat-template format, e.g.
                [{"role": "user", "content": [{"type": "text", "text": "hi"}]},
                 {"role": "assistant", "content": [{"type": "text", "text": "..."}]},
                 {"role": "user", "content": [{"type": "text", "text": "..."}]}]
                Image parts are currently rejected.
            max_new_tokens: hard cap on generated tokens (engine will also stop on EOS).

        Yields:
            (token_id, decoded_text_chunk, stats) per token. Stats is a dict
            with token_count, elapsed_s, decode_s, tps for the current stream.
        """
        input_ids = self._prepare_input_ids(messages)
        bundle_json = self._write_bundle(input_ids, max_new_tokens)
        try:
            yield from self._run_engine_streaming(bundle_json)
        finally:
            try:
                os.remove(bundle_json)
            except OSError:
                pass

    # ----- Internals --------------------------------------------------------

    def _prepare_input_ids(self, messages: List[Dict]) -> List[int]:
        # TODO: image support without torch_npu. Options:
        #   (1) run the HF SiglipVisionModel on CPU (slow but no NPU dep) and
        #       insert resulting features into the bundle as a separate
        #       embeddings-by-position override.
        #   (2) port the vision encoder + image-to-token-features projection
        #       into the C++ engine.
        # For now any image part raises so the UX surface stays honest.
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                for p in content:
                    if isinstance(p, dict) and p.get("type") not in (None, "text"):
                        raise NotImplementedError(
                            f"non-text content part {p.get('type')!r} not supported in "
                            "torch_npu-free mode; text only for now"
                        )

        # apply_chat_template is CPU-only (tokenizer ops). No tensors moved
        # to NPU here; we read input_ids straight off the CPU list.
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        input_ids = inputs["input_ids"]
        if hasattr(input_ids, "tolist"):
            ids_list = input_ids[0].tolist() if hasattr(input_ids[0], "tolist") else list(input_ids[0])
        else:
            ids_list = list(input_ids[0]) if isinstance(input_ids[0], (list, tuple)) else list(input_ids)
        return [int(x) for x in ids_list]

    def _write_bundle(self, input_ids: List[int], max_new_tokens: int) -> str:
        seq_len = len(input_ids)
        prefix = os.path.join(
            tempfile.gettempdir(),
            f"minicpmv_{os.getpid()}_{int(time.time() * 1000)}",
        )
        json_path = prefix + ".json"
        bundle = {
            "version": 1,
            "seq_len": seq_len,
            "hidden_size": HIDDEN_SIZE,
            "max_seq_len": seq_len + max_new_tokens + 4,
            "max_new_tokens": max_new_tokens,
            "vocab_size": VOCAB_SIZE,
            "eos_token_ids": EOS_TOKEN_IDS,
            "layer_types": LAYER_TYPES,
            # No embeddings file — engine does embedding lookup from input_ids.
            "embeddings_dtype": "float16",
            "embeddings_endianness": "little",
            "embeddings_path": "",
            "weights_path": os.path.join(self.model_path, "model.safetensors"),
            "input_ids": input_ids,
        }
        with open(json_path, "w") as f:
            json.dump(bundle, f)
        return json_path

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
        print(f"[minicpmv_session] spawning engine server (weights load ~30s)…",
              file=sys.stderr, flush=True)
        proc = subprocess.Popen(
            [self.engine_bin, "--server", "--weights", weights_path],
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

        # Drain any remaining stderr in a background thread so it doesn't
        # backpressure the engine.
        def _drain_stderr() -> None:
            try:
                for _ in iter(proc.stderr.readline, ""):
                    pass
            except Exception:
                pass

        threading.Thread(target=_drain_stderr, daemon=True).start()

        self._engine_proc = proc
        print("[minicpmv_session] engine server ready", file=sys.stderr, flush=True)
        return proc
