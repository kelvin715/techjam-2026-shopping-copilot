"""Run the organizer's unmodified evaluator with INPUT_MODE="agentic".

Sets config before constructing Agent instead of patching
evaluator/local_evaluator.py, so that file stays byte-identical to the
organizer's. Makes real OpenAI calls -- point --dataset at a small set,
not the full public set, unless you mean to spend that.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import Agent
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from src import config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results_agentic.json")
    args = parser.parse_args()

    config.INPUT_MODE = "agentic"
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    result = evaluate(Agent(args.catalog), samples, catalog_ids, categories, products)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
