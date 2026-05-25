#!/usr/bin/env python3
"""Gradio web UI for MiniCPM-V 4.6 on Ascend 310B.

Streaming multimodal chat over the C++ AscendC engine. Vision encoder
runs on the NPU via the engine (no torch_npu). Run from the repo root:

    pip install gradio pillow
    source scripts/set_env.sh
    python3 src/python/gradio_app.py

Then open http://localhost:7860 in a browser. Pass --share to expose via
Gradio's tunnel.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid

# Make sibling `minicpmv` package importable when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr  # noqa: E402

from minicpmv.session import MinicpmvSession  # noqa: E402


def _entry_to_message(entry: dict) -> dict:
    """Translate one Gradio chat entry into an HF chat-template `messages` dict.

    History entries can contain plain text or image/file payloads. We preserve
    the ordering so image placeholder tokens line up with the same positions in
    the engine-side prefix cache.
    """
    role = entry["role"]
    raw = entry["content"]
    parts: list[dict] = []
    if isinstance(raw, str):
        parts.append({"type": "text", "text": raw})
    elif isinstance(raw, dict):
        text = raw.get("text") or ""
        for f in raw.get("files", []) or []:
            p = f["path"] if isinstance(f, dict) else f
            parts.append({"type": "image", "url": p})
        if text:
            parts.append({"type": "text", "text": text})
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, dict) and "path" in item:
                parts.append({"type": "image", "url": item["path"]})
            elif isinstance(item, str) and os.path.isfile(item):
                parts.append({"type": "image", "url": item})
            elif isinstance(item, str):
                parts.append({"type": "text", "text": item})
    if not parts:
        parts.append({"type": "text", "text": str(raw)})
    return {"role": role, "content": parts}


def build_app(session: MinicpmvSession, max_new_tokens: int = 256) -> gr.Blocks:
    desc = (
        "MiniCPM-V 4.6 streaming multimodal chat via the custom AscendC engine. "
        "Vision encoder runs on the NPU (no torch_npu). Drop an image into "
        "the message box to ask vision questions."
    )

    def new_conversation_id() -> str:
        return str(uuid.uuid4())

    def chat_fn(message, history: list[dict], conversation_id: str):
        messages = [_entry_to_message(e) for e in history]
        messages.append(_entry_to_message({"role": "user", "content": message}))
        t0 = time.time()
        response = ""
        stats: dict = {}
        try:
            for _tok, text_chunk, stats in session.generate(
                messages,
                max_new_tokens=max_new_tokens,
                conversation_id=conversation_id,
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
        print(
            f"[chat_fn] conv={conversation_id[:8]} tokens={stats.get('token_count', 0)} "
            f"elapsed={time.time()-t0:.1f}s",
            flush=True,
        )

    def clear_chat() -> tuple[list, str]:
        # New conversation_id forces engine-side prefix cache reset.
        return [], new_conversation_id()

    with gr.Blocks(title="MiniCPM-V 4.6 · Orange Pi AIPro 20T (Ascend 310B)") as demo:
        gr.Markdown("# MiniCPM-V 4.6 · Orange Pi AIPro 20T (Ascend 310B)")
        gr.Markdown(desc)

        conv_id = gr.State(new_conversation_id())
        chatbot = gr.Chatbot(type="messages", height=600)
        msg = gr.MultimodalTextbox(
            file_types=["image"],
            placeholder="Type a message, optionally drop in an image…",
            file_count="multiple",
            show_label=False,
        )
        clear_btn = gr.Button("Clear history")

        msg.submit(chat_fn, [msg, chatbot, conv_id], [chatbot])
        clear_btn.click(clear_chat, outputs=[chatbot, conv_id])

    return demo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="0.0.0.0", help="Bind address")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Expose via Gradio tunnel")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--text-only",
        action="store_true",
        help="Skip loading the vision tower (saves ~1.5 GB NPU + ~30s startup)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with_vision = not args.text_only
    print(f"[gradio_app] loading tokenizer + spawning engine (vision={with_vision})…", flush=True)
    session = MinicpmvSession(with_vision=with_vision)
    session._ensure_engine_server()  # noqa: SLF001
    # One warmup call to absorb the engine's first-time aclnn JIT.
    print("[gradio_app] warming engine (first-time aclnn JIT)…", flush=True)
    t0 = time.time()
    warm_conv = str(uuid.uuid4())
    msg = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    n = 0
    for _ in session.generate(msg, max_new_tokens=4, conversation_id=warm_conv):
        n += 1
    print(f"[gradio_app] warmup done: {n} tok in {time.time()-t0:.1f}s", flush=True)
    print("[gradio_app] ready", flush=True)
    app = build_app(session, max_new_tokens=args.max_new_tokens)
    app.queue().launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    sys.exit(main())
