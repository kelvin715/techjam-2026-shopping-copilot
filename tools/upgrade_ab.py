from __future__ import annotations

import json
import math
import random
import sys

sys.path.insert(0, ".")

import evaluator.local_evaluator as evaluator
from src import config
from starter.agent import Agent


def contribution(row: dict) -> float:
    turn = row["first_hit_turn"] if row["first_hit_turn"] is not None else 11
    efficiency = max(0.0, min(1.0, 1.0 - (turn - 1) / 10.0))
    return (
        0.5 * float(row["hit"])
        + 0.3 * float(row["reciprocal_rank"])
        + 0.2 * efficiency
    )


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[round(p * (len(ordered) - 1))]


def main() -> None:
    catalog = "data/catalog.jsonl"
    samples = evaluator.load_jsonl("data/public_set.jsonl")
    ids, categories, products = evaluator.catalog_index(catalog)
    agent = Agent(catalog)
    max_log = agent.catalog._log_max_ratings

    original_config = (
        config.SIGNATURE_WEIGHT,
        config.POPULARITY_WEIGHT,
        config.PROVEN_MISS_EXCLUSION,
        config.QUESTION_MODE,
    )
    config.SIGNATURE_WEIGHT = 0.0
    config.POPULARITY_WEIGHT = 0.35
    config.PROVEN_MISS_EXCLUSION = False
    config.QUESTION_MODE = "fixed"

    def old_popularity(asin: str) -> float:
        count = math.log1p(agent.catalog.rating_count.get(asin, 0)) / max_log
        quality = max(0.0, min(1.0, agent.catalog.rating.get(asin, 0.0) / 5.0))
        return 0.65 * count + 0.35 * quality

    agent.catalog.popularity = old_popularity
    try:
        old = evaluator.evaluate(agent, samples, ids, categories, products)
    finally:
        (
            config.SIGNATURE_WEIGHT,
            config.POPULARITY_WEIGHT,
            config.PROVEN_MISS_EXCLUSION,
            config.QUESTION_MODE,
        ) = original_config
    new = json.load(open("results/public_full.json", encoding="utf-8"))
    old_rows = {row["sample_id"]: row for row in old["sessions"]}
    new_rows = {row["sample_id"]: row for row in new["sessions"]}
    keys = sorted(old_rows)
    deltas = [
        contribution(new_rows[key]) - contribution(old_rows[key])
        for key in keys
    ]
    rng = random.Random(20260830)
    bootstrap = [
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(20000)
    ]
    print(json.dumps({
        "old_score": old["recommended_technical_score"],
        "new_score": new["recommended_technical_score"],
        "paired_delta": sum(deltas) / len(deltas),
        "bootstrap_seed": 20260830,
        "bootstrap_resamples": len(bootstrap),
        "paired_delta_95_ci": [
            percentile(bootstrap, 0.025),
            percentile(bootstrap, 0.975),
        ],
        "sessions_improved": sum(delta > 0 for delta in deltas),
        "sessions_unchanged": sum(delta == 0 for delta in deltas),
        "sessions_worsened": sum(delta < 0 for delta in deltas),
    }, indent=2))


if __name__ == "__main__":
    main()
