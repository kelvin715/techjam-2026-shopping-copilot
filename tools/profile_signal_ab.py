"""Test whether aggregate profile tags earn a safe ranking contribution.

Profile evidence is promoted only if it improves public, popularity-matched,
and uniform-target diagnostics. The default remains zero until that gate is
met; retaining unused profile state is preferable to unvalidated personalization.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from src import config
from starter.agent import Agent
from tools.research_common import proxy_samples


def compact(outcome: dict) -> dict:
    return {
        "hit_rate_at_10": outcome["hit_rate_at_10"],
        "mrr": outcome["mrr"],
        "mttc": outcome["mttc"],
        "technical_score": outcome["recommended_technical_score"],
        "scenario_metrics": outcome["scenario_metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument(
        "--proxy", choices=("public", "matched", "uniform"), default="public"
    )
    parser.add_argument("--count", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--weights", default="0,0.02,0.05,0.10,0.20,0.30")
    parser.add_argument("--output", default="results/profile_signal_ab.json")
    args = parser.parse_args()

    if args.proxy == "public":
        samples = load_jsonl(args.dataset)
        ids, categories, products = catalog_index(args.catalog)
    else:
        samples, ids, categories, products = proxy_samples(
            args.catalog, args.dataset, args.proxy, args.count, args.seed
        )
    weights = tuple(float(value) for value in args.weights.split(","))
    original = config.PROFILE_BONUS
    results: dict[str, dict] = {}
    try:
        for weight in weights:
            config.PROFILE_BONUS = weight
            outcome = evaluate(
                Agent(args.catalog), samples, ids, categories, products
            )
            label = f"profile_bonus_{weight:g}"
            results[label] = compact(outcome)
            print(
                f"{label:<22} HR={outcome['hit_rate_at_10']:.6f} "
                f"MRR={outcome['mrr']:.6f} MTTC={outcome['mttc']:.3f} "
                f"score={outcome['recommended_technical_score']:.6f}"
            )
    finally:
        config.PROFILE_BONUS = original

    baseline = results[f"profile_bonus_{weights[0]:g}"]["technical_score"]
    result = {
        "status": "diagnostic_only_profile_not_promoted_without_three_distribution_gain",
        "dataset": args.proxy,
        "sample_count": len(samples),
        "results": results,
        "deltas_vs_first": {
            label: round(row["technical_score"] - baseline, 6)
            for label, row in results.items()
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
