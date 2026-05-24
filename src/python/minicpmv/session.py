"""Reusable MiniCPM-V 4.6 session: Python prepares prompt embeddings and
(optionally) image features, the C++ engine binary `minicpmv_hybrid_decode`
runs the language-model decode loop.

The HF processor + model are loaded once at session construction; each
`generate(messages, ...)` call rebuilds prompt embeddings, dumps an
embeddings+config bundle to a tmp file, and spawns the engine subprocess to
stream tokens back.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from typing import Dict, Generator, List, Optional, Tuple

import torch
import torch_npu  # noqa: F401  (registers NPU backend)
from transformers import AutoModelForImageTextToText, AutoProcessor

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


class MinicpmvSession:
    """Holds the HF processor + model + paths for one decode session.

    Construction loads the model on the NPU once (the slow step). Each
    `generate(...)` call rebuilds prompt embeddings and spawns the engine
    subprocess to stream tokens back as a generator.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        engine_bin: Optional[str] = None,
        custom_opp: Optional[str] = None,
        device: str = "npu:0",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.model_path = model_path or os.environ.get("MINICPMV_MODEL_PATH", DEFAULT_MODEL_PATH)
        self.engine_bin = engine_bin or os.environ.get("MINICPMV_ENGINE_BIN", DEFAULT_ENGINE_BIN)
        self.custom_opp = custom_opp or os.environ.get("MINICPMV_CUSTOM_OPP", DEFAULT_CUSTOM_OPP)
        self.device = device
        self.dtype = dtype

        torch.set_grad_enabled(False)
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                dtype=self.dtype,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
            .to(self.device)
            .eval()
        )

    # ----- Public API -------------------------------------------------------

    def generate(
        self,
        messages: List[Dict],
        max_new_tokens: int = 128,
        downsample_mode: str = "16x",
    ) -> Generator[Tuple[int, str, Dict], None, None]:
        """Stream the assistant turn for `messages`.

        Args:
            messages: HF chat-template format, e.g.
                [{"role": "user", "content": [{"type": "text", "text": "hi"},
                                              {"type": "image", "url": "/path/to.png"}]},
                 {"role": "assistant", "content": [{"type": "text", "text": "..."}]},
                 {"role": "user", "content": [{"type": "text", "text": "..."}]}]
            max_new_tokens: hard cap on generated tokens (engine will also stop on EOS).
            downsample_mode: HF processor image downsample mode.

        Yields:
            (token_id, decoded_text_chunk, stats) per token. Stats is a dict
            with token_count, elapsed_s, tps for the current stream.
        """
        input_ids, _, inputs_embeds = self._prepare_inputs_embeds(messages, downsample_mode)
        bundle_json, embeddings_path = self._write_bundle(input_ids, inputs_embeds, max_new_tokens)
        try:
            yield from self._run_engine_streaming(bundle_json)
        finally:
            for p in (bundle_json, embeddings_path):
                try:
                    os.remove(p)
                except OSError:
                    pass

    # ----- Internals --------------------------------------------------------

    def _prepare_inputs_embeds(
        self,
        messages: List[Dict],
        downsample_mode: str,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            downsample_mode=downsample_mode,
            max_slice_nums=9,
        ).to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        pixel_values = inputs.get("pixel_values")
        target_sizes = inputs.get("target_sizes")

        inputs_embeds = self.model.model.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            vision_output = self.model.model.get_image_features(
                pixel_values[:1], target_sizes, downsample_mode=downsample_mode,
            )
            image_features = torch.cat(vision_output.pooler_output, dim=0).to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype,
            )
            mask = self.model.model.get_placeholder_mask(
                input_ids, inputs_embeds, image_features, self.model.config.image_token_id,
            )
            inputs_embeds = inputs_embeds.masked_scatter(mask, image_features)
        return input_ids, attention_mask, inputs_embeds

    def _write_bundle(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        max_new_tokens: int,
    ) -> Tuple[str, str]:
        seq_len = int(inputs_embeds.shape[1])
        hidden_size = int(inputs_embeds.shape[2])
        embeds_cpu = inputs_embeds[0].detach().to(torch.float16).contiguous().cpu()
        prefix = os.path.join(
            tempfile.gettempdir(),
            f"minicpmv_{os.getpid()}_{int(time.time() * 1000)}",
        )
        embeddings_path = prefix + ".embeds.f16"
        json_path = prefix + ".json"
        with open(embeddings_path, "wb") as f:
            f.write(embeds_cpu.numpy().tobytes())
        bundle = {
            "version": 1,
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "max_seq_len": seq_len + max_new_tokens + 4,
            "max_new_tokens": max_new_tokens,
            "vocab_size": VOCAB_SIZE,
            "eos_token_ids": EOS_TOKEN_IDS,
            "layer_types": LAYER_TYPES,
            "embeddings_dtype": "float16",
            "embeddings_endianness": "little",
            "embeddings_layout": "contiguous_row_major_BTH",
            "embeddings_path": embeddings_path,
            "weights_path": os.path.join(self.model_path, "model.safetensors"),
            "input_ids": [int(x) for x in input_ids[0].cpu().tolist()],
        }
        with open(json_path, "w") as f:
            json.dump(bundle, f)
        return json_path, embeddings_path

    def _run_engine_streaming(
        self,
        bundle_json: str,
    ) -> Generator[Tuple[int, str, Dict], None, None]:
        env = os.environ.copy()
        env["ASCEND_CUSTOM_OPP_PATH"] = self.custom_opp
        op_lib = self.custom_opp + "/op_api/lib"
        env["LD_LIBRARY_PATH"] = op_lib + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )
        proc = subprocess.Popen(
            [self.engine_bin, "--bundle", bundle_json],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,
        )
        assert proc.stdout is not None
        start_time = time.time()
        first_token_time: Optional[float] = None
        count = 0
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                if line.startswith("# done"):
                    continue
                if not line:
                    continue
                try:
                    tok = int(line)
                except ValueError:
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
        finally:
            err = proc.stderr.read() if proc.stderr is not None else ""
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"engine failed rc={rc}, stderr={err!r}")
