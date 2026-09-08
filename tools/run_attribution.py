"""Run the attribution grid: shopper family x wording level x agent arm.

Every job is one ``tools/human_language_benchmark.py`` process with strict
rewrite replay from an immutable cache, its own log, and its own output file
under ``results/attribution/``. Jobs run concurrently up to ``--jobs``. When
all jobs finish, ``summary.json`` and ``summary.md`` are written.

    python3 tools/run_attribution.py --base-url http://host:port/v1 --jobs 12

Caches
  Rewrites come from one immutable cache per shopper; grounding replies from
  one cache per arm, shared by every (shopper, level) cell of that arm. Cells
  of one arm that run concurrently can request the same prompt (a retention
  rewrite can equal the source message) and receive different replies from a
  non-deterministic server; the last flush wins, and the other cell's
  trajectory then no longer replays. Run ``--strict-grounding`` afterwards:
  a cell that misses must be refilled (non-strict) and replayed again, and
  ``tools/check_attribution_replay.py`` confirms fill == replay for all cells.

Arms
  deterministic      frozen controller, no free-text reader
  lexical            exact catalog-string matching, no model
  dense              embedding similarity added to the evidence score
  cascade            catalog reads first, model consulted on demand (the shipped reader)
  hybrid             model proposes on every off-protocol message, catalog verifies
  hybrid_unverified  model proposes, everything admitted
  hybrid_nohints     model proposes without catalog vocabulary in the prompt
  llm_agent          the model picks the question and ranks a shortlist
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "attribution"

SHOPPERS = {
    "gemma": {"customer_model": "gemma4", "cache": "results/rewrites_gemma.json"},
    "qwen": {"customer_model": "qwen2.5-7b-instruct", "cache": "results/human_language_cache_qwen_customer.json"},
}
ARMS = ("deterministic", "lexical", "dense", "cascade", "hybrid", "hybrid_unverified", "hybrid_nohints", "hybrid_flatweights", "hybrid_verbatim_only", "llm_agent")
LEVELS = ("natural", "paraphrase")


def job_command(args, shopper: str, level: str, arm: str) -> tuple[list[str], Path, Path]:
    spec = SHOPPERS[shopper]
    out_dir = OUT / args.out_dir if args.out_dir else OUT
    output = out_dir / f"{shopper}_{level}_{arm}.json"
    log = out_dir / "logs" / f"{shopper}_{level}_{arm}.log"
    cmd = [
        sys.executable, "-u", str(ROOT / "tools" / "human_language_benchmark.py"),
        "--count", str(args.count), "--levels", level, "--arms", arm,
        "--customer-model", spec["customer_model"], "--cache", spec["cache"], "--strict-replay",
        "--grounding-cache", str(OUT / f"grounding_{arm}.json"),
        "--base-url", args.base_url, "--model", args.model, "--timeout", str(args.timeout),
        "--output", str(output),
    ]
    if args.strict_grounding:
        # Replay pass: every grounding / agent call must come from the cache
        # written by the original run, so the job needs no endpoint at all.
        cmd.append("--strict-grounding")
    if args.traces:
        cmd.append("--traces")
    return cmd, output, log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.environ.get("ARC_LLM_BASE_URL", ""))
    parser.add_argument("--model", default="gemma4")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--shoppers", default="gemma,qwen")
    parser.add_argument("--levels", default="canonical,natural,paraphrase")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--strict-grounding", action="store_true",
                        help="replay grounding calls from the per-arm cache; abort on any miss")
    parser.add_argument("--traces", action="store_true", help="keep every session's turn-level trajectory")
    parser.add_argument("--out-dir", default="",
                        help="subdirectory of results/attribution/ for outputs and logs (e.g. replay)")
    args = parser.parse_args()
    if not args.base_url and not args.summary_only and not args.strict_grounding:
        parser.error("--base-url is required unless --strict-grounding replays from the caches")
    out_dir = OUT / args.out_dir if args.out_dir else OUT
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    if not args.summary_only:
        jobs = []
        for level in args.levels.split(","):
            shoppers = ["gemma"] if level == "canonical" else args.shoppers.split(",")
            for shopper in shoppers:
                for arm in args.arms.split(","):
                    cmd, output, log = job_command(args, shopper, level, arm)
                    if args.skip_existing and output.is_file():
                        continue
                    jobs.append((shopper, level, arm, cmd, output, log))
        print(f"{len(jobs)} jobs, {args.jobs} at a time")
        running: list[tuple[subprocess.Popen, tuple]] = []
        started = time.perf_counter()
        pending = list(jobs)
        failures = []
        while pending or running:
            while pending and len(running) < args.jobs:
                shopper, level, arm, cmd, output, log = pending.pop(0)
                handle = open(log, "w")
                proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=str(ROOT))
                running.append((proc, (shopper, level, arm, output, log, handle)))
                print(f"start {shopper}/{level}/{arm}", flush=True)
            time.sleep(5)
            still = []
            for proc, meta in running:
                if proc.poll() is None:
                    still.append((proc, meta))
                    continue
                shopper, level, arm, output, log, handle = meta
                handle.close()
                status = "ok" if proc.returncode == 0 and output.is_file() else f"FAILED rc={proc.returncode}"
                if status != "ok":
                    failures.append((shopper, level, arm, str(log)))
                print(f"done  {shopper}/{level}/{arm}: {status} ({time.perf_counter() - started:.0f}s)", flush=True)
            running = still
        if failures:
            print("failures:")
            for item in failures:
                print("  ", item)

    # ---- summary
    rows = []
    for path in sorted(out_dir.glob("*_*_*.json")):
        if path.name.startswith(("grounding_", "summary", "combined")) or path.name.endswith(".traces.json"):
            continue
        shopper, level, arm = path.stem.split("_", 2)
        data = json.loads(path.read_text(encoding="utf-8"))
        for label, exp in data["experiments"].items():
            n = data["sample_count"]
            rows.append({
                "shopper": shopper, "level": exp["level"], "arm": exp["arm"],
                "hit_rate_at_10": exp["hit_rate_at_10"], "mrr": exp["mrr"], "mttc": exp["mttc"],
                "technical_score": exp["technical_score"],
                "calls_per_session": exp["grounding"]["model_calls"] / n,
                "tokens_per_session": exp["reported_token_usage"]["total_tokens"] / n,
                "mean_respond_latency_ms": exp["mean_respond_latency_ms"],
                "llm_agent_fallbacks": exp.get("llm_agent_fallbacks"),
                "config_overrides": exp.get("config_overrides", {}),
                "grounding_cache_misses": exp.get("grounding_cache_misses"),
                "customer_rewrite_failures": exp.get("customer_rewrite_failures"),
                "sessions_recorded": len(exp.get("sessions", [])),
                "file": str(path.relative_to(ROOT)),
            })
    # Combined per-shopper files in the benchmark's own output format, so the
    # paper's table scripts can read the replay grid like a single run.
    combined: dict[str, dict] = {}
    for path in sorted(out_dir.glob("*_*_*.json")):
        if path.name.startswith(("grounding_", "summary", "combined")) or path.name.endswith(".traces.json"):
            continue
        shopper = path.stem.split("_", 1)[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        target = combined.setdefault(shopper, {
            "status": data["status"], "source": "strict replay of the attribution grid; one benchmark process per (level, arm)",
            "sample_count": data["sample_count"], "levels": [], "arms": [],
            "grounding_model": data["grounding_model"], "customer_model": data["customer_model"],
            "caches": {"rewrites": data["caches"]["rewrites"], "strict_replay": data["caches"]["strict_replay"]},
            "experiments": {}, "files": [],
        })
        for level, examples in data.get("sample_rewrites", {}).items():
            if examples:
                target.setdefault("sample_rewrites", {}).setdefault(level, examples)
        for label, exp in data["experiments"].items():
            target["experiments"][label] = exp
            if exp["level"] not in target["levels"]:
                target["levels"].append(exp["level"])
            if exp["arm"] not in target["arms"]:
                target["arms"].append(exp["arm"])
        target["files"].append(str(path.relative_to(ROOT)))
    for shopper, data in combined.items():
        data["levels"].sort(key=lambda level: {"canonical": 0, "natural": 1, "paraphrase": 2}[level])
        (out_dir / f"combined_{shopper}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    order = {arm: index for index, arm in enumerate(ARMS)}
    rows.sort(key=lambda r: ({"canonical": 0, "natural": 1, "paraphrase": 2}[r["level"]], r["shopper"], order.get(r["arm"], 99)))
    (out_dir / "summary.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    lines = ["| level | shopper | arm | Hit@10 | MRR | MTTC | Score | calls/sess | tokens/sess | ms/turn |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['level']} | {r['shopper']} | {r['arm']} | {r['hit_rate_at_10']:.3f} | {r['mrr']:.3f} | {r['mttc']:.2f} | {r['technical_score']:.4f} | {r['calls_per_session']:.1f} | {r['tokens_per_session']:,.0f} | {r['mean_respond_latency_ms']:.0f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
