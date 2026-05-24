#!/usr/bin/env python3
"""Gradio web UI for MiniCPM-V 4.6 on Ascend 310B.

Streaming chat over the C++ AscendC engine. Text-only — no torch_npu
dependency. Image input is shown as TBD until a CPU/engine-side vision
encoder lands. Run from the repo root:

    pip install gradio
    source scripts/set_env.sh
    python3 src/python/gradio_app.py

Then open http://localhost:7860 in a browser. Pass --share to expose via
gradio's tunnel.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Make sibling `minicpmv` package importable when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr  # noqa: E402

from minicpmv.session import MinicpmvSession  # noqa: E402


def _history_to_messages(history: list[dict]) -> list[dict]:
    """Translate gradio's `messages`-format history into HF chat format.
    Text-only: image entries from prior turns are ignored with a warning."""
    out: list[dict] = []
    for entry in history:
        role = entry["role"]
        raw = entry["content"]
        if isinstance(raw, str):
            out.append({"role": role, "content": [{"type": "text", "text": raw}]})
        else:
            text = str(raw)
            out.append({"role": role, "content": [{"type": "text", "text": text}]})
    return out


def build_app(session: MinicpmvSession, max_new_tokens: int = 256) -> gr.Blocks:
    def chat_fn(message: str, history: list[dict]):
        messages = _history_to_messages(history)
        messages.append({"role": "user", "content": [{"type": "text", "text": message}]})
        t0 = time.time()
        response = ""
        stats: dict = {}
        try:
            for _tok, text_chunk, stats in session.generate(
                messages, max_new_tokens=max_new_tokens,
            ):
                response += text_chunk
                yield response
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield response + f"\n\n⚠ engine error: {e}"
            return
        if stats:
            ttft = stats["elapsed_s"] - stats["decode_s"]
            footer = (
                f"\n\n<sub>{stats['token_count']} tok · "
                f"ttft={ttft:.2f}s · decode={stats['tps']:.2f} tps</sub>"
            )
            yield response + footer
        print(f"[chat_fn] {stats.get('token_count', 0)} tok in {time.time()-t0:.1f}s", flush=True)

    desc = (
        "MiniCPM-V 4.6 streaming chat via the custom AscendC engine "
        "(text-only — image support coming after the vision encoder is "
        "ported off torch_npu)."
    )
    return gr.ChatInterface(
        fn=chat_fn,
        title="MiniCPM-V 4.6 · Orange Pi AIPro 20T (Ascend 310B)",
        description=desc,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="0.0.0.0", help="Bind address")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Expose via gradio tunnel")
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print("[gradio_app] loading tokenizer + spawning engine…", flush=True)
    session = MinicpmvSession()
    # Force the engine subprocess to spawn now (rather than on first request)
    # so model weights are uploaded to the NPU before we accept any user load.
    session._ensure_engine_server()  # noqa: SLF001
    # One warmup call to absorb the engine's first-time aclnn JIT (~9s).
    # After this, any user prompt shape is fast (the engine has no per-shape
    # JIT cost; only the very first prefill ever pays this).
    print("[gradio_app] warming engine (first-time aclnn JIT)…", flush=True)
    t0 = time.time()
    msg = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    n = 0
    for _ in session.generate(msg, max_new_tokens=4):
        n += 1
    print(f"[gradio_app] warmup done: {n} tok in {time.time()-t0:.1f}s", flush=True)
    print("[gradio_app] ready", flush=True)
    app = build_app(session, max_new_tokens=args.max_new_tokens)
    app.queue().launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    sys.exit(main())
