#!/usr/bin/env python3
"""CLI runner for MiniCPM-V 4.6 + Ascend engine.

Streams generated tokens to stdout. Wraps the reusable
`minicpmv.session.MinicpmvSession`; see that module for the Python API.
Text-only — no torch_npu dependency. Image support TBD."""

from __future__ import annotations

import argparse
import os
import sys

# Add ../ to path so the `minicpmv` package is importable when this file is
# run directly via `python3 src/python/run_hybrid.py`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minicpmv.session import MinicpmvSession  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", required=True, help="User text prompt")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--engine-bin", default=None,
                   help="Override engine binary path (default: $REPO/build/minicpmv_hybrid_decode)")
    return p.parse_args()


def build_user_message(prompt: str) -> list[dict]:
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]


def main() -> int:
    args = parse_args()
    session = MinicpmvSession(engine_bin=args.engine_bin)

    messages = build_user_message(args.prompt)
    last_stats: dict = {}
    for _tok, text, stats in session.generate(messages, max_new_tokens=args.max_new_tokens):
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
