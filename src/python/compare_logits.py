#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForImageTextToText

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.environ.get("MINICPMV_MODEL_PATH", os.path.join(REPO_ROOT, "MiniCPM-V-4.6"))
DEVICE = "npu:0"
DTYPE = torch.float16
ENGINE_BIN = os.environ.get(
    "MINICPMV_LOGITS_DUMP_BIN",
    os.path.join(REPO_ROOT, "build", "test_language_logits_dump"),
)
CUSTOM_OPP = os.environ.get(
    "MINICPMV_CUSTOM_OPP",
    os.path.join(REPO_ROOT, "custom_opp_install", "vendors", "customize"),
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", nargs="+", type=int, default=[1, 2, 10, 100])
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--engine-bin", default=ENGINE_BIN)
    p.add_argument("--engine-mode", choices=["prefill", "step"], default="prefill")
    p.add_argument("--reference-mode", choices=["prefill", "step"], default=None)
    return p.parse_args()


def run_engine(tokens, engine_bin, engine_mode):
    env = os.environ.copy()
    env["ASCEND_CUSTOM_OPP_PATH"] = CUSTOM_OPP
    lib = f"{CUSTOM_OPP}/op_api/lib"
    env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    cmd = [engine_bin, "--mode", engine_mode] + [str(t) for t in tokens]
    proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"engine output did not contain JSON; stdout={proc.stdout!r} stderr={proc.stderr!r}")


def run_reference(tokens, topk, reference_mode):
    torch.set_grad_enabled(False)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        dtype=DTYPE,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(DEVICE).eval()

    input_ids = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        inputs_embeds = model.model.get_input_embeddings()(input_ids)
        if reference_mode == "prefill":
            out = model.model.language_model(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                use_cache=False,
            )
            hidden = out.last_hidden_state[:, -1, :]
        else:
            past = None
            hidden = None
            for i in range(len(tokens)):
                step_embeds = inputs_embeds[:, i : i + 1, :]
                step_mask = attention_mask[:, : i + 1]
                out = model.model.language_model(
                    attention_mask=step_mask,
                    inputs_embeds=step_embeds,
                    past_key_values=past,
                    use_cache=True,
                )
                past = out.past_key_values
                hidden = out.last_hidden_state[:, -1, :]
            assert hidden is not None
        logits = model.lm_head(hidden).float()[0]
        values, indices = torch.topk(logits, k=topk)
    return {
        "next_token": int(indices[0].item()),
        "topk_ids": [int(x) for x in indices.detach().cpu().tolist()],
        "topk_logits": [float(x) for x in values.detach().cpu().tolist()],
        "logits": logits.detach().cpu(),
    }


def main():
    args = parse_args()
    reference_mode = args.reference_mode or args.engine_mode
    engine = run_engine(args.tokens, args.engine_bin, args.engine_mode)
    ref = run_reference(args.tokens, args.topk, reference_mode)

    engine_ids = engine["topk_ids"][: args.topk]
    ref_ids = ref["topk_ids"][: args.topk]
    overlap = len(set(engine_ids) & set(ref_ids))
    union_ids = sorted(set(engine_ids) | set(ref_ids))

    engine_logits_by_id = {int(i): float(v) for i, v in zip(engine["topk_ids"], engine["topk_logits"])}
    diffs = []
    for i in union_ids:
        if i in engine_logits_by_id:
            diffs.append(abs(float(ref["logits"][i].item()) - engine_logits_by_id[i]))
    max_abs = max(diffs) if diffs else float("nan")
    mean_abs = sum(diffs) / len(diffs) if diffs else float("nan")

    print(f"tokens={args.tokens} engine_mode={args.engine_mode} reference_mode={reference_mode}")
    print(f"ref_next={ref['next_token']} engine_next={engine['next_token']} match={ref['next_token'] == engine['next_token']}")
    print(f"ref_top{args.topk}={list(zip(ref_ids, ref['topk_logits'][:args.topk]))}")
    print(f"engine_top{args.topk}={list(zip(engine_ids, engine['topk_logits'][:args.topk]))}")
    print(f"top{args.topk}_overlap={overlap}/{args.topk} union_logit_diff_max={max_abs:.6g} mean={mean_abs:.6g}")

    return 0 if ref["next_token"] == int(engine["next_token"]) else 1


if __name__ == "__main__":
    sys.exit(main())
