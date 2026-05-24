#!/usr/bin/env python3
"""Hybrid MiniCPM-V-4.6 runner: Python prepares prompt embeddings, C++ engine
runs the language model decode loop. Text-only first; image support is added on
top by passing --image."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Optional

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.environ.get("MINICPMV_MODEL_PATH", os.path.join(REPO_ROOT, "MiniCPM-V-4.6"))
DEVICE = "npu:0"
DTYPE = torch.float16
ENGINE_BIN = os.environ.get(
    "MINICPMV_ENGINE_BIN",
    os.path.join(REPO_ROOT, "build", "minicpmv_hybrid_decode"),
)
CUSTOM_OPP = os.environ.get(
    "MINICPMV_CUSTOM_OPP",
    os.path.join(REPO_ROOT, "custom_opp_install", "vendors", "customize"),
)
LAYER_TYPES = [
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
]
EOS_TOKEN_IDS = [248044, 248046]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--image", default=None)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--downsample-mode", default="16x")
    p.add_argument("--engine-bin", default=ENGINE_BIN)
    p.add_argument("--bundle-prefix", default=None)
    p.add_argument("--keep-bundle", action="store_true")
    p.add_argument("--compare-hf", action="store_true",
                   help="Also run HF greedy and report token-level match")
    return p.parse_args()


def prepare_inputs_embeds(model, processor, prompt: str, image_path: Optional[str], downsample_mode: str):
    content = []
    if image_path:
        content.append({"type": "image", "url": image_path})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        downsample_mode=downsample_mode,
        max_slice_nums=9,
    ).to(DEVICE)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs.get("pixel_values")
    target_sizes = inputs.get("target_sizes")
    inputs_embeds = model.model.get_input_embeddings()(input_ids)
    if pixel_values is not None:
        vision_output = model.model.get_image_features(
            pixel_values[:1], target_sizes, downsample_mode=downsample_mode,
        )
        image_features = torch.cat(vision_output.pooler_output, dim=0).to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype,
        )
        mask = model.model.get_placeholder_mask(
            input_ids, inputs_embeds, image_features, model.config.image_token_id,
        )
        inputs_embeds = inputs_embeds.masked_scatter(mask, image_features)
    return input_ids, attention_mask, inputs_embeds


def write_bundle(prefix: str, input_ids: torch.Tensor, inputs_embeds: torch.Tensor,
                 max_new_tokens: int) -> tuple[str, str]:
    seq_len = int(inputs_embeds.shape[1])
    hidden_size = int(inputs_embeds.shape[2])
    embeds_cpu = inputs_embeds[0].detach().to(torch.float16).contiguous().cpu()
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
        "vocab_size": 248094,
        "eos_token_ids": EOS_TOKEN_IDS,
        "layer_types": LAYER_TYPES,
        "embeddings_dtype": "float16",
        "embeddings_endianness": "little",
        "embeddings_layout": "contiguous_row_major_BTH",
        "embeddings_path": embeddings_path,
        "weights_path": os.path.join(MODEL_PATH, "model.safetensors"),
        "input_ids": [int(x) for x in input_ids[0].cpu().tolist()],
    }
    with open(json_path, "w") as f:
        json.dump(bundle, f)
    return json_path, embeddings_path


def run_engine_streaming(engine_bin: str, bundle_json: str, on_token):
    env = os.environ.copy()
    env["ASCEND_CUSTOM_OPP_PATH"] = CUSTOM_OPP
    lib = CUSTOM_OPP + "/op_api/lib"
    env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    proc = subprocess.Popen(
        [engine_bin, "--bundle", bundle_json],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    )
    assert proc.stdout is not None
    emitted = []
    done_line = None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line.startswith("# done"):
            done_line = line
            continue
        if not line:
            continue
        try:
            tok = int(line)
        except ValueError:
            continue
        emitted.append(tok)
        on_token(tok)
    err = proc.stderr.read() if proc.stderr is not None else ""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"engine failed rc={rc}, stderr={err!r}")
    return emitted, done_line


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        dtype=DTYPE,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(DEVICE).eval()

    input_ids, attention_mask, inputs_embeds = prepare_inputs_embeds(
        model, processor, args.prompt, args.image, args.downsample_mode,
    )

    prefix = args.bundle_prefix or os.path.join(
        tempfile.gettempdir(), f"minicpmv_run_{os.getpid()}_{int(time.time())}",
    )
    bundle_json, embeddings_path = write_bundle(prefix, input_ids, inputs_embeds, args.max_new_tokens)

    pieces = []

    def on_token(tok):
        text = processor.tokenizer.decode([tok], skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)
        pieces.append(text)
        sys.stdout.write(text)
        sys.stdout.flush()

    tokens, done_line = run_engine_streaming(args.engine_bin, bundle_json, on_token)
    sys.stdout.write("\n")
    sys.stdout.flush()

    if args.compare_hf:
        with torch.inference_mode():
            ref = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )
        ref_new = ref[0, input_ids.shape[1]:].cpu().tolist()
        print(f"hf_tokens={ref_new}")
        print(f"engine_tokens={tokens}")
        common = min(len(ref_new), len(tokens))
        match_prefix = 0
        for i in range(common):
            if int(ref_new[i]) != int(tokens[i]):
                break
            match_prefix += 1
        print(f"match_prefix={match_prefix}/{common}")

    if done_line:
        print(done_line)

    if not args.keep_bundle:
        try:
            os.remove(bundle_json)
            os.remove(embeddings_path)
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
