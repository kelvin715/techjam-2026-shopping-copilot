"""Verify the strict-replay attribution grid against the first-pass runs.

For every (shopper, level, arm) present in ``results/attribution/replay``:
  * the replay was strict on both caches and reports zero grounding misses,
  * per-session outcomes and trajectories are present,
  * the aggregate metrics equal the first-pass file (``results/attribution``
    or, for the model-as-agent arm, ``results/attribution/fresh_llm_agent``)
    to six decimals.

Exit status 1 on any discrepancy, so the check can gate the paper build.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ATTR = ROOT / "results" / "attribution"
REPLAY = ATTR / "replay"
KEYS = ("hit_rate_at_10", "mrr", "mttc", "technical_score")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", default=str(REPLAY.relative_to(ROOT)),
                        help="directory of the strict replay (default: results/attribution/replay)")
    parser.add_argument("--first", default=str(ATTR.relative_to(ROOT)),
                        help="directory of the first-pass (cache-filling) runs")
    args = parser.parse_args()
    replay_dir = ROOT / args.replay
    first_root = ROOT / args.first
    problems = []
    rows = []
    for path in sorted(replay_dir.glob("*_*_*.json")):
        if path.name.startswith(("grounding_", "summary", "combined", "run_")) or path.name.endswith(".traces.json"):
            continue
        shopper, level, arm = path.stem.split("_", 2)
        data = json.loads(path.read_text(encoding="utf-8"))
        exp = list(data["experiments"].values())[0]
        first_dir = first_root / "fresh_llm_agent" if (arm == "llm_agent" and (first_root / "fresh_llm_agent").is_dir()) else first_root
        first_path = first_dir / path.name
        first = list(json.loads(first_path.read_text(encoding="utf-8"))["experiments"].values())[0] if first_path.is_file() else None
        traces = path.with_name(path.stem + ".traces.json")
        status = []
        if not data["caches"].get("strict_replay"):
            status.append("rewrites not strict")
        if exp.get("grounding_cache_misses"):
            status.append(f"{exp['grounding_cache_misses']} grounding misses")
        if exp.get("customer_rewrite_failures"):
            status.append(f"{exp['customer_rewrite_failures']} rewrite fallbacks")
        if len(exp.get("sessions", [])) != data["sample_count"]:
            status.append("sessions missing")
        if not traces.is_file():
            status.append("traces missing")
        if first is None:
            # Cells that make no model calls need no cache and no first pass:
            # the replay is the run.
            if exp["grounding"]["model_calls"]:
                status.append("no first-pass file")
        else:
            for key in KEYS:
                if abs(first[key] - exp[key]) > 5e-7:
                    status.append(f"{key} {first[key]:.6f} -> {exp[key]:.6f}")
            if first["grounding"]["model_calls"] != exp["grounding"]["model_calls"]:
                status.append(f"calls {first['grounding']['model_calls']} -> {exp['grounding']['model_calls']}")
        rows.append((shopper, level, arm, exp["technical_score"], exp["grounding"]["model_calls"], status))
        if status:
            problems.append((path.name, status))
    for shopper, level, arm, score, calls, status in rows:
        print(f"{shopper:6s} {level:10s} {arm:22s} score={score:.6f} calls={calls:5d} {'OK' if not status else '; '.join(status)}")
    print(f"{len(rows)} replayed runs, {len(problems)} with discrepancies")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
