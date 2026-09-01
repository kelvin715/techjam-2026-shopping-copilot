"""A/B late evidence-tie exploration on declared evaluation distributions.

The policy is promoted only when it improves long-tail coverage without a
material public or popularity-matched regression. Labels are consumed solely
by the unchanged offline evaluator; they are never passed to ``Agent``.
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
        "--proxy", choices=("public", "matched", "uniform"), default="uniform"
    )
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--turns", default="off,5",
        help="Comma-separated activation turns; use 'off' for the control.",
    )
    parser.add_argument("--output", default="results/tail_exploration_ab.json")
    args = parser.parse_args()

    if args.proxy == "public":
        samples = load_jsonl(args.dataset)
        ids, categories, products = catalog_index(args.catalog)
    else:
        samples, ids, categories, products = proxy_samples(
            args.catalog, args.dataset, args.proxy, args.count, args.seed
        )

    settings = [value.strip().lower() for value in args.turns.split(",")]
    original = (
        config.TAIL_EXPLORATION_ENABLED,
        config.TAIL_EXPLORATION_TURN,
    )
    results: dict[str, dict] = {}
    try:
        for setting in settings:
            enabled = setting != "off"
            config.TAIL_EXPLORATION_ENABLED = enabled
            if enabled:
                config.TAIL_EXPLORATION_TURN = int(setting)
            label = "off" if not enabled else f"turn_{setting}"
            outcome = evaluate(
                Agent(args.catalog), samples, ids, categories, products
            )
            results[label] = compact(outcome)
            print(
                f"{label:<10} HR={outcome['hit_rate_at_10']:.6f} "
                f"MRR={outcome['mrr']:.6f} MTTC={outcome['mttc']:.3f} "
                f"score={outcome['recommended_technical_score']:.6f}"
            )
    finally:
        (
            config.TAIL_EXPLORATION_ENABLED,
            config.TAIL_EXPLORATION_TURN,
        ) = original

    control = results.get("off")
    result = {
        "status": "diagnostic_only_labels_never_passed_to_agent",
        "dataset": args.proxy,
        "sample_count": len(samples),
        "results": results,
        "deltas_vs_off": {
            label: (
                round(row["technical_score"] - control["technical_score"], 6)
                if control is not None
                else None
            )
            for label, row in results.items()
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
