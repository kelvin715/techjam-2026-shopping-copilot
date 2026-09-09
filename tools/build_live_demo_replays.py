"""Record real stage events for the final demo's offline fallback.

Run after agent changes, before rebuilding the main demo bundle. Uses the
catalog reader without model calls; targets are joined only after respond.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import Agent
from demo.server import DemoRuntime
from src.llm import LLMSettings
from tools.build_demo_bundle import sha256_file


def build(catalog_path, sample_ids):
    runtime = DemoRuntime(catalog_path=catalog_path, live_requested=True)
    runtime._agent = Agent(catalog_path, llm_settings=LLMSettings(mode="lexical"))
    sessions = []
    for sample_id in sample_ids:
        session = runtime.lab.reset({"sample_id": sample_id})
        message = session["suggestion"]
        turns = []
        for turn in range(1, 11):
            result = runtime.lab.turn({"session_id": session["session_id"], "turn": turn,
                                       "message": message}, lambda *_: None)
            turns.append(result)
            if result["done"]:
                break
            message = result["suggestion"]
        sessions.append({"sample_id": sample_id, "session": session, "turns": turns})
        print(f"{sample_id}: {len(turns)} turns, hit={turns[-1]['overlay']['hit']}", flush=True)
    sources = [ROOT / "agent.py", *sorted((ROOT / "src").glob("*.py")),
               ROOT / "demo/live.py", ROOT / "demo/server.py", Path(__file__)]
    return {"format": "arc-live-lab-v1", "recorded_at": datetime.now(timezone.utc).isoformat(),
            "source_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in sources},
            "catalog_sha256": sha256_file(catalog_path), "sessions": sessions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.jsonl")
    parser.add_argument("--sessions", default="public_0099,public_0144,public_0187,public_0029,public_0035,public_0004")
    parser.add_argument("--output", type=Path, default=ROOT / "demo/static/lab-replays.json")
    args = parser.parse_args()
    result = build(args.catalog, args.sessions.split(","))
    args.output.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
