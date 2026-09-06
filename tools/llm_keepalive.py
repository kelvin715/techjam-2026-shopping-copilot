"""Keep the optional language-layer endpoint warm before a live demo.

Shared model servers put a model to sleep after a few idle minutes and can
take minutes to reload it. This sends a one-token request every ``--interval``
seconds so the first judge question is answered in milliseconds, not minutes.
Standard library only; stop with Ctrl-C.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm import ChatClient, LLMSettings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("ARC_LLM_BASE_URL", ""))
    parser.add_argument("--model", default=os.environ.get("ARC_LLM_MODEL", "gemma4"))
    parser.add_argument("--api-key", default=os.environ.get("ARC_LLM_API_KEY", "unused"))
    parser.add_argument("--interval", type=float, default=240.0)
    parser.add_argument("--once", action="store_true", help="send one request and report")
    args = parser.parse_args()
    if not args.base_url:
        parser.error("--base-url (or ARC_LLM_BASE_URL) is required")
    client = ChatClient(LLMSettings(
        mode="ground", base_url=args.base_url.rstrip("/"), model=args.model,
        api_key=args.api_key, timeout=300.0,
    ))
    while True:
        # Streaming survives a cold start: the proxy drops a silent
        # non-streaming request long before a multi-minute weight load ends.
        reply = client.chat(
            [{"role": "user", "content": "Reply with OK."}], max_tokens=2, stream=True
        )
        stamp = time.strftime("%H:%M:%S")
        if reply.ok:
            print(f"{stamp} warm · {reply.latency_ms:.0f} ms · {args.model}", flush=True)
        else:
            print(f"{stamp} FAILED · {reply.error}", flush=True)
        if args.once:
            sys.exit(0 if reply.ok else 1)
        time.sleep(max(10.0, args.interval))


if __name__ == "__main__":
    main()
