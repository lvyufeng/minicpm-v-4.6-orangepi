#!/usr/bin/env python3
"""Gradio web UI for MiniCPM-V 4.6 on Ascend 310B.

Launches a multimodal chat (text + image) that streams tokens from the C++
engine. Run from the repo root:

    conda activate minicpm46  # or your env with torch_npu + transformers
    pip install gradio pillow  # one-time
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


def _convert_history(history: list[dict]) -> list[dict]:
    """Translate gradio's `messages`-format history into HF chat format.

    Each gradio history entry is `{"role": "user"|"assistant", "content": ...}`
    where `content` is either a str or a tuple `(filepath,)` for media. HF
    chat template wants `content` to be a list of typed parts."""
    out: list[dict] = []
    for entry in history:
        role = entry["role"]
        raw = entry["content"]
        if isinstance(raw, str):
            out.append({"role": role, "content": [{"type": "text", "text": raw}]})
            continue
        # Gradio attaches uploaded files as tuples or dicts depending on
        # version; normalize to a single content list.
        parts: list[dict] = []
        if isinstance(raw, (list, tuple)):
            for item in raw:
                if isinstance(item, str) and os.path.isfile(item):
                    parts.append({"type": "image", "url": item})
                elif isinstance(item, dict) and "path" in item:
                    parts.append({"type": "image", "url": item["path"]})
                elif isinstance(item, str):
                    parts.append({"type": "text", "text": item})
        elif isinstance(raw, dict):
            text = raw.get("text")
            if text:
                parts.append({"type": "text", "text": text})
            for f in raw.get("files", []) or []:
                if isinstance(f, dict):
                    parts.append({"type": "image", "url": f.get("path") or f.get("url")})
                else:
                    parts.append({"type": "image", "url": f})
        if not parts:
            parts.append({"type": "text", "text": str(raw)})
        out.append({"role": role, "content": parts})
    return out


def _build_user_turn(message: dict | str) -> dict:
    """Convert the latest gradio message (text + uploaded files) into a chat
    template user turn. `message` is either a str (text-only) or a dict with
    `text` + `files`."""
    if isinstance(message, str):
        return {"role": "user", "content": [{"type": "text", "text": message}]}
    text = message.get("text", "") or ""
    files = message.get("files", []) or []
    parts: list[dict] = []
    for f in files:
        path = f["path"] if isinstance(f, dict) else f
        parts.append({"type": "image", "url": path})
    parts.append({"type": "text", "text": text})
    return {"role": "user", "content": parts}


def build_app(session: MinicpmvSession, max_new_tokens: int = 256) -> gr.Blocks:
    def chat_fn(message, history):
        messages = _convert_history(history)
        messages.append(_build_user_turn(message))
        t0 = time.time()
        response = ""
        stats: dict = {}
        for _tok, text_chunk, stats in session.generate(
            messages, max_new_tokens=max_new_tokens,
        ):
            response += text_chunk
            yield response
        # Final yield includes a stats footer so users can see decode tps.
        if stats:
            ttft = stats["elapsed_s"] - stats["decode_s"]
            footer = (
                f"\n\n<sub>{stats['token_count']} tok · "
                f"ttft={ttft:.2f}s · decode={stats['tps']:.2f} tps</sub>"
            )
            yield response + footer

    desc = (
        "Streaming MiniCPM-V 4.6 inference via the custom AscendC engine. "
        "Upload an image with your message to test vision."
    )
    return gr.ChatInterface(
        fn=chat_fn,
        multimodal=True,
        title="MiniCPM-V 4.6 · Orange Pi AIPro 20T (Ascend 310B)",
        description=desc,
        textbox=gr.MultimodalTextbox(
            file_types=["image"],
            placeholder="Type a message, optionally drop in an image…",
            file_count="multiple",
        ),
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
    print("[gradio_app] loading model + processor (this takes a moment)…", flush=True)
    session = MinicpmvSession()
    print("[gradio_app] ready", flush=True)
    app = build_app(session, max_new_tokens=args.max_new_tokens)
    app.queue().launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    sys.exit(main())
