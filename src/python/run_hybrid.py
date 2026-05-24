#!/usr/bin/env python3
"""CLI runner for MiniCPM-V 4.6 + Ascend engine.

Streams generated tokens to stdout. Wraps the reusable
`minicpmv.session.MinicpmvSession`; see that module for the Python API."""

from __future__ import annotations

import argparse
import os
import sys

# Add ../ to path so the `minicpmv` package is importable when this file is
# run directly via `python3 src/python/run_hybrid.py`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from minicpmv.session import MinicpmvSession  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", required=True, help="User text prompt")
    p.add_argument("--image", default=None, help="Optional image file path")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--downsample-mode", default="16x")
    p.add_argument("--engine-bin", default=None,
                   help="Override engine binary path (default: $REPO/build/minicpmv_hybrid_decode)")
    p.add_argument("--compare-hf", action="store_true",
                   help="Also run HF greedy and report token-level match")
    return p.parse_args()


def build_user_message(prompt: str, image_path: str | None) -> list[dict]:
    content: list[dict] = []
    if image_path:
        content.append({"type": "image", "url": image_path})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def main() -> int:
    args = parse_args()
    session = MinicpmvSession(engine_bin=args.engine_bin)

    messages = build_user_message(args.prompt, args.image)
    tokens: list[int] = []
    last_stats: dict = {}
    for tok, text, stats in session.generate(
        messages,
        max_new_tokens=args.max_new_tokens,
        downsample_mode=args.downsample_mode,
    ):
        tokens.append(tok)
        last_stats = stats
        sys.stdout.write(text)
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    if last_stats:
        print(
            f"# done generated={last_stats['token_count']} "
            f"elapsed={last_stats['elapsed_s']:.2f}s "
            f"decode_tps={last_stats['tps']:.2f}",
            file=sys.stderr,
        )

    if args.compare_hf:
        # Re-run with HF greedy for a token-level comparison.
        input_ids, attention_mask, inputs_embeds = session._prepare_inputs_embeds(  # noqa: SLF001
            messages, args.downsample_mode,
        )
        with torch.inference_mode():
            ref = session.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )
        ref_new = ref[0, input_ids.shape[1]:].cpu().tolist()
        print(f"hf_tokens={ref_new}", file=sys.stderr)
        print(f"engine_tokens={tokens}", file=sys.stderr)
        common = min(len(ref_new), len(tokens))
        match_prefix = 0
        for i in range(common):
            if int(ref_new[i]) != int(tokens[i]):
                break
            match_prefix += 1
        print(f"match_prefix={match_prefix}/{common}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
