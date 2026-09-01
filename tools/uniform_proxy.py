"""Deterministic uniform-target long-tail stress diagnostic.

This diagnostic samples targets uniformly from non-public catalog products. It
is deliberately unlike the organizer's popularity-skewed public distribution,
so it exposes whether the bounded quality/popularity prior masks long-tail
retrieval failures.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent
from tools.matched_proxy import percentile, rating_number, scenario_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output", default="results/uniform_proxy_full.json")
    args = parser.parse_args()

    public = load_jsonl(args.dataset)
    public_targets = {str(row["ground_truth"]["parent_asin"]) for row in public}
    profiles = [row["user_profile"] for row in public]
    ids, categories, products = catalog_index(args.catalog)
    pool = sorted(
        asin
        for asin, product in products.items()
        if asin not in public_targets and product.get("features") and product.get("details")
    )
    rng = random.Random(args.seed)
    targets = rng.sample(pool, min(args.count, len(pool)))
    scenarios = scenario_sequence(len(targets), rng)
    samples = [
        {
            "sample_id": f"proxy_uniform_{position:05d}",
            "scenario_type": scenario,
            "category_bucket": "clothing",
            "difficulty_bucket": difficulty,
            "user_profile": rng.choice(profiles),
            "ground_truth": {"parent_asin": asin},
        }
        for position, (asin, (scenario, difficulty)) in enumerate(
            zip(targets, scenarios), start=1
        )
    ]
    outcome = evaluate(Agent(args.catalog), samples, ids, categories, products)
    counts = [rating_number(products[asin]) for asin in targets]
    summary = {
        "status": "diagnostic_only_not_an_official_private_score",
        "sample_count": outcome["sample_count"],
        "seed": args.seed,
        "sampling": "uniform_without_replacement_from_eligible_non_public_targets",
        "target_popularity": {
            "median_rating_number": percentile(counts, 0.50),
            "p25_rating_number": percentile(counts, 0.25),
            "p75_rating_number": percentile(counts, 0.75),
        },
        "hit_rate_at_10": outcome["hit_rate_at_10"],
        "mrr": outcome["mrr"],
        "mttc": outcome["mttc"],
        "technical_score": outcome["recommended_technical_score"],
        "scenario_metrics": outcome["scenario_metrics"],
    }
    full = {**summary, "sessions": outcome["sessions"]}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(full, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
