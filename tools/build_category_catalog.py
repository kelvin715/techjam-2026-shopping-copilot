"""Build a competition-schema catalog for another Amazon Reviews 2023 category.

The organizer's 50,000-product catalog is a sample of the
``Clothing_Shoes_and_Jewelry`` metadata of Amazon Reviews 2023 (McAuley Lab,
UCSD). The other categories of that dataset share the same fields, so the
same simulator, the same evaluator and the same Agent can be run on a
catalog that the design never saw. This tool streams one category's
``meta_<Category>.jsonl`` (a local file or the Hugging Face URL), keeps the
ten fields the competition ships, and reservoir-samples a fixed number of
eligible products with a fixed seed so the catalog is reproducible.

    python3 tools/build_category_catalog.py --category Electronics \\
        --output data/catalog_electronics.jsonl

Eligibility mirrors what the proxy diagnostics require of a target: a
``parent_asin``, a title, a category path, and non-empty ``features`` and
``details``. The output is written outside version control (``data/`` is
ignored) and stays under the source dataset's terms; only the summary that
``tools/cross_category.py`` writes is committed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import urllib.request
from pathlib import Path

FIELDS = (
    "parent_asin", "title", "features", "description", "price", "categories",
    "details", "average_rating", "rating_number", "store",
)
HF_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
    "raw/meta_categories/meta_{category}.jsonl"
)


def _price(value: object) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(str(value).replace(",", "").lstrip("$"))
    except ValueError:
        return None


def normalise(row: dict) -> dict | None:
    """Project one raw metadata row onto the competition schema, or None."""
    asin = row.get("parent_asin")
    title = row.get("title")
    features = row.get("features") or []
    details = row.get("details") or {}
    categories = row.get("categories") or []
    if not asin or not title or not features or not details or not categories:
        return None
    if not isinstance(details, dict) or not isinstance(features, list):
        return None
    try:
        rating = float(row.get("average_rating") or 0.0)
        count = int(row.get("rating_number") or 0)
    except (TypeError, ValueError):
        rating, count = 0.0, 0
    return {
        "parent_asin": str(asin),
        "title": str(title),
        "features": [str(item) for item in features if item],
        "description": [str(item) for item in (row.get("description") or []) if item],
        "price": _price(row.get("price")),
        "categories": [str(item) for item in categories if item],
        "details": {str(key): str(value) for key, value in details.items() if value not in (None, "")},
        "average_rating": rating,
        "rating_number": count,
        "store": str(row.get("store") or ""),
    }


def open_source(source: str):
    if source.startswith("http://") or source.startswith("https://"):
        request = urllib.request.Request(source, headers={"User-Agent": "arc-cross-category/1.0"})
        return io.TextIOWrapper(urllib.request.urlopen(request, timeout=120), encoding="utf-8")
    return Path(source).open(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", required=True, help="Amazon Reviews 2023 category name, e.g. Electronics")
    parser.add_argument("--source", default=None, help="local meta_<Category>.jsonl; default: stream from Hugging Face")
    parser.add_argument("--output", default=None, help="default data/catalog_<category>.jsonl")
    parser.add_argument("--count", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()

    source = args.source or HF_URL.format(category=args.category)
    output = Path(args.output or f"data/catalog_{args.category.lower()}.jsonl")
    rng = random.Random(args.seed)
    reservoir: list[dict] = []
    seen_raw = 0
    eligible = 0
    seen_asins: set[str] = set()
    with open_source(source) as handle:
        for line in handle:
            seen_raw += 1
            if seen_raw % 200000 == 0:
                print(f"  {seen_raw:,} rows read, {eligible:,} eligible", file=sys.stderr, flush=True)
            try:
                row = normalise(json.loads(line))
            except json.JSONDecodeError:
                continue
            if row is None or row["parent_asin"] in seen_asins:
                continue
            seen_asins.add(row["parent_asin"])
            eligible += 1
            if len(reservoir) < args.count:
                reservoir.append(row)
            else:
                slot = rng.randrange(eligible)
                if slot < args.count:
                    reservoir[slot] = row
    reservoir.sort(key=lambda item: item["parent_asin"])
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with output.open("w", encoding="utf-8") as handle:
        for row in reservoir:
            text = json.dumps(row, ensure_ascii=False) + "\n"
            handle.write(text)
            digest.update(text.encode("utf-8"))
    summary = {
        "category": args.category,
        "source": source,
        "rows_read": seen_raw,
        "eligible": eligible,
        "sampled": len(reservoir),
        "seed": args.seed,
        "output": str(output),
        "sha256": digest.hexdigest(),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
