"""Shared deterministic sample construction for research-only diagnostics."""
from __future__ import annotations

import random

from evaluator.local_evaluator import catalog_index, load_jsonl
from tools.matched_proxy import matched_targets, rating_number, scenario_sequence


def proxy_samples(
    catalog_path: str,
    public_path: str,
    kind: str,
    count: int,
    seed: int,
) -> tuple[list[dict], set[str], dict[str, list[str]], dict[str, dict]]:
    """Build a frozen, target-disjoint matched or uniform proxy sample."""
    public = load_jsonl(public_path)
    public_targets = {str(row["ground_truth"]["parent_asin"]) for row in public}
    profiles = [row["user_profile"] for row in public]
    ids, categories, products = catalog_index(catalog_path)
    pool = sorted(
        asin
        for asin, product in products.items()
        if asin not in public_targets
        and product.get("features")
        and product.get("details")
    )
    rng = random.Random(seed)
    if kind == "matched":
        reference = sorted(
            rating_number(products[asin])
            for asin in sorted(public_targets)
            if asin in products
        )
        targets = matched_targets(products, pool, reference, rng, count)
    elif kind == "uniform":
        targets = rng.sample(pool, min(count, len(pool)))
    else:
        raise ValueError(f"unsupported proxy kind: {kind}")
    scenarios = scenario_sequence(len(targets), rng)
    samples = [
        {
            # Preserve the standalone proxy tools' identifiers because the
            # evaluator deterministically seeds hidden override timing from
            # sample_id.
            "sample_id": f"proxy_{kind}_{position:05d}",
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
    return samples, ids, categories, products
