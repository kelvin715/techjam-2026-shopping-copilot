"""Assemble a multi-department catalog from per-category catalogs.

Takes several competition-schema catalogs (the organizer's clothing catalog
and the ones ``tools/build_category_catalog.py`` builds for other Amazon
Reviews 2023 categories), draws an equal quota of eligible products from each
with a fixed seed, and writes one mixed catalog plus a sidecar that records
which department every product came from, so a diagnostic can stratify
targets and report per-department results.

    python3 tools/build_mixed_catalog.py --quota 5000 \\
        --output data/catalog_mixed_50k.jsonl \\
        clothing=data/catalog.jsonl electronics=data/catalog_electronics.jsonl ...

Quotas are drawn as a prefix of one seeded shuffle per department, so the
5,000-per-department catalog is a subset of the 25,000-per-department one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def eligible(product: dict) -> bool:
    return bool(product.get("parent_asin") and product.get("title") and product.get("categories")
                and product.get("features") and product.get("details"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", help="NAME=PATH per department")
    parser.add_argument("--quota", type=int, default=5000, help="products per department")
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    departments: dict[str, str] = {}
    digest = hashlib.sha256()
    written = 0
    per_department: dict[str, int] = {}
    seen: set[str] = set()
    with output.open("w", encoding="utf-8") as handle:
        for spec in args.sources:
            name, _, path = spec.partition("=")
            if not name or not path:
                parser.error(f"expected NAME=PATH, got {spec!r}")
            rows = []
            with Path(path).open(encoding="utf-8") as source:
                for line in source:
                    product = json.loads(line)
                    if eligible(product) and product["parent_asin"] not in seen:
                        rows.append(product)
            rows.sort(key=lambda item: item["parent_asin"])
            random.Random(f"{args.seed}:{name}").shuffle(rows)
            chosen = rows[: args.quota]
            for product in chosen:
                seen.add(product["parent_asin"])
                departments[product["parent_asin"]] = name
                text = json.dumps(product, ensure_ascii=False) + "\n"
                handle.write(text)
                digest.update(text.encode("utf-8"))
            per_department[name] = len(chosen)
            written += len(chosen)
            print(f"{name}: {len(rows):,} eligible, {len(chosen):,} taken", flush=True)
    sidecar = output.with_suffix(".departments.json")
    sidecar.write_text(json.dumps({"seed": args.seed, "quota": args.quota, "per_department": per_department,
                                   "department_of": departments}, indent=0) + "\n", encoding="utf-8")
    summary = {"output": str(output), "rows": written, "per_department": per_department,
               "seed": args.seed, "sha256": digest.hexdigest(), "departments_file": str(sidecar)}
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
