"""Pre-generate human-shopper rewrites with a *different* model family.

The shared model server keeps one model awake at a time, so a benchmark that
alternates customer-model and grounding-model calls would thrash. Instead,
enumerate every customer message the organizer's simulator can produce for a
session (opener, every ``customer_reply`` for every attribute and every
disclosed subset, boundary replies, the no-information reply, and the override
message), rewrite them once with the customer model, and store them in the
same cache ``tools/human_language_benchmark.py`` reads. The benchmark can then
run with ``--customer-model <that model>`` without a single live customer
call, while the grounding model stays awake.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (
    ALLOWED_ATTRIBUTES,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from src.llm import ChatClient, LLMSettings
from tools.human_language_benchmark import CUSTOMER_PROMPTS


def messages_for(sample: dict, categories: dict, products: dict) -> set[str]:
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}
    target = str(sample["ground_truth"]["parent_asin"])
    out: set[str] = set()
    out.add(initial_message(effective, coarse_category(categories.get(target, [])), set()))
    constraints = [*map(str, card.get("hard_constraints", [])), *map(str, card.get("soft_preferences", []))]
    unique = list(dict.fromkeys(constraints))
    attributes = sorted(ALLOWED_ATTRIBUTES)
    for size in range(len(unique) + 1):
        for subset in itertools.combinations(unique, size):
            for attribute in attributes:
                disclosed = set(subset)
                text, _ = customer_reply(effective, attribute, disclosed, True)
                out.add(text)
    for attribute in attributes:
        text, _ = customer_reply(effective, attribute, set(), False)
        out.add(text)
        if sample["scenario_type"] == "boundary":
            out.add(f"I don't have a preference for {attribute}; please use your judgment.")
    out.add("Those options are not quite right yet. Ask me about one specific attribute.")
    override = (behavior or {}).get("override") or {}
    if override.get("message"):
        out.add(str(override["message"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--levels", default="natural,paraphrase")
    parser.add_argument("--base-url", default=os.environ.get("ARC_CUSTOMER_BASE_URL") or os.environ.get("ARC_LLM_BASE_URL", ""))
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="unused")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if not args.base_url:
        parser.error("--base-url is required")

    samples = load_jsonl(args.dataset)[: args.count]
    _, categories, products = catalog_index(args.catalog)
    needed: set[str] = set()
    for sample in samples:
        needed |= messages_for(sample, categories, products)
    print(f"{len(needed)} distinct simulator messages across {len(samples)} sessions")

    client = ChatClient(LLMSettings(
        mode="ground", base_url=args.base_url.rstrip("/"), model=args.model,
        api_key=args.api_key, timeout=args.timeout, cache_path=args.cache,
    ))
    levels = [item.strip() for item in args.levels.split(",") if item.strip()]
    jobs = [(level, message) for level in levels for message in sorted(needed)]

    def work(job: tuple[str, str]) -> bool:
        level, message = job
        reply = client.chat(
            [
                {"role": "system", "content": CUSTOMER_PROMPTS[level]},
                {"role": "user", "content": message},
            ],
            max_tokens=120,
            temperature=0.0,
        )
        return reply.ok

    started = time.perf_counter()
    ok = 0
    # Warm the model with one sequential call so the parallel batch does not
    # pile up on a cold start.
    work(jobs[0])
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, success in enumerate(pool.map(work, jobs), start=1):
            ok += int(success)
            if index % 500 == 0:
                client.flush()
                print(f"  {index}/{len(jobs)} rewrites, {ok} ok, {time.perf_counter() - started:.0f}s")
    client.flush()
    print(f"done: {ok}/{len(jobs)} rewrites cached in {args.cache} ({time.perf_counter() - started:.0f}s)")


if __name__ == "__main__":
    main()
