"""Build the auditable data bundle consumed by the demo frontend.

The frontend never reads the full catalog and never derives evaluation metrics.
This tool copies measured results, compact public-session evaluator replays,
and the products used by the verified counterfactual story. A source manifest
lets the demo server refuse a green "verified" badge when its replay no longer
matches the working tree.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent

RESULT_SOURCES = (
    "results/demo_walkthrough.json",
    "results/decision_trace.json",
    "results/explore_decision_trace.json",
    "results/public_metrics.json",
    "results/matched_proxy_metrics.json",
    "results/uniform_proxy_metrics.json",
    "results/runtime_profile.json",
    "results/robustness_policy_benchmark.json",
    "results/public_session_replays.json",
    "docs/baseline_results.json",
)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_paths(root: Path) -> list[Path]:
    paths = [root / name for name in RESULT_SOURCES]
    paths.extend((root / "src").glob("*.py"))
    paths.extend((root / "demo" / "static").glob("*"))
    paths.extend((root / "tests").glob("test_*.py"))
    paths.extend(
        root / name
        for name in (
            "agent.py",
            "demo/server.py",
            "tools/build_demo_bundle.py",
            "tools/build_session_replays.py",
        )
    )
    return sorted(
        (path for path in paths if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def source_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in _source_paths(root)
    }


def source_fingerprint(hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(hashes.items()):
        digest.update(f"{name}\0{value}\n".encode())
    return digest.hexdigest()


def _source_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "uncommitted-worktree"
    return result.stdout.strip() or "uncommitted-worktree"


def _test_method_count(root: Path) -> int:
    count = 0
    for path in sorted((root / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        count += sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
            for node in ast.walk(tree)
        )
    return count


def _coarse_category(values: Iterable[object]) -> str:
    excluded = {
        "clothing",
        "clothing shoes & jewelry",
        "clothing, shoes & jewelry",
    }
    cleaned: list[str] = []
    for value in values or []:
        for part in str(value).split(","):
            item = part.strip()
            if item and item.lower() not in excluded:
                cleaned.append(item)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def _short_title(title: str, store: str = "", limit: int = 62) -> str:
    """Shorten a keyword-stuffed catalog title for display next to its store.

    The card already prints the store above the title, so a leading brand token
    is dropped instead of eating the character budget twice.
    """
    title = " ".join(str(title).split())
    store = " ".join(str(store).split())
    if store and title.lower().startswith(f"{store.lower()} "):
        title = title[len(store) + 1:].lstrip()
    if len(title) <= limit:
        return title
    prefix = title[: limit - 1].rsplit(" ", 1)[0]
    return f"{prefix}…"


def _normalise_product(product: dict, matched_evidence: list[str]) -> dict:
    title = str(product.get("title") or product.get("parent_asin") or "Product")
    raw_price = product.get("price")
    try:
        price = float(raw_price) if raw_price not in (None, "") else None
    except (TypeError, ValueError):
        price = None
    try:
        rating = float(product.get("average_rating") or 0.0)
    except (TypeError, ValueError):
        rating = 0.0
    try:
        rating_count = max(0, int(product.get("rating_number") or 0))
    except (TypeError, ValueError):
        rating_count = 0
    store = str(product.get("store") or "Independent seller")
    return {
        "parent_asin": str(product.get("parent_asin") or ""),
        "title": title,
        "display_title": _short_title(title, store),
        "store": store,
        "price": price,
        "average_rating": rating,
        "rating_number": rating_count,
        "categories": [str(item) for item in product.get("categories") or []],
        "matched_evidence": list(matched_evidence),
    }


def catalog_snapshot(
    catalog_path: Path | None,
    wanted_asins: set[str],
    target_asin: str | None = None,
) -> tuple[dict[str, object], dict[str, dict]]:
    if catalog_path is None or not catalog_path.is_file():
        return (
            {
                "available_at_build": False,
                "row_count": 50000,
                "sha256": None,
                "shelf_name": "Tees & Blouses Tunics",
                "shelf_count": None,
            },
            {},
        )

    digest = hashlib.sha256()
    products: dict[str, dict] = {}
    shelf_counts: dict[str, int] = {}
    row_count = 0
    with catalog_path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            row_count += 1
            product = json.loads(raw_line)
            asin = str(product.get("parent_asin") or "")
            shelf = _coarse_category(product.get("categories") or [])
            shelf_counts[shelf] = shelf_counts.get(shelf, 0) + 1
            if asin in wanted_asins:
                products[asin] = product

    missing = sorted(wanted_asins - products.keys())
    if missing:
        raise ValueError(f"catalog is missing story products: {missing}")
    target = target_asin or next(iter(sorted(wanted_asins)))
    target_shelf = _coarse_category(products[target].get("categories") or [])
    return (
        {
            "available_at_build": True,
            "row_count": row_count,
            "sha256": digest.hexdigest(),
            "shelf_name": target_shelf,
            "shelf_count": shelf_counts.get(target_shelf),
        },
        products,
    )


def _fallback_products(story: dict, counterfactual: dict) -> dict[str, dict]:
    target = str(story["target_parent_asin"])
    replacement = str(counterfactual.get("counterfactual_top") or "")
    values = {
        target: {
            "parent_asin": target,
            "title": story.get("target_title") or target,
        }
    }
    if replacement:
        values[replacement] = {
            "parent_asin": replacement,
            "title": replacement,
        }
    return values


def _variant_matrix(robustness: dict) -> list[dict]:
    """Per-variant scores for both readers, as one row per way of speaking."""
    experiments = robustness["experiments"]
    rows = []
    for variant in ("canonical", "surface_noise", "natural_paraphrase", "hidden_one"):
        canonical = experiments[f"canonical_parser__{variant}"]
        robust = experiments[f"robust_parser__{variant}"]
        rows.append({
            "variant": variant,
            "canonical_reader": {
                "hit_rate_at_10": canonical["hit_rate_at_10"],
                "technical_score": canonical["technical_score"],
            },
            "both_readers": {
                "hit_rate_at_10": robust["hit_rate_at_10"],
                "technical_score": robust["technical_score"],
            },
        })
    return rows


def build_bundle(
    root: Path = ROOT,
    catalog_path: Path | None = None,
    *,
    generated_at: str | None = None,
    tests_verified: bool = False,
) -> dict:
    demo = _read_json(root / "results/demo_walkthrough.json")
    public = _read_json(root / "results/public_metrics.json")
    matched = _read_json(root / "results/matched_proxy_metrics.json")
    uniform = _read_json(root / "results/uniform_proxy_metrics.json")
    runtime = _read_json(root / "results/runtime_profile.json")
    robustness = _read_json(root / "results/robustness_policy_benchmark.json")
    trace = _read_json(root / "results/decision_trace.json")
    explore_trace = _read_json(root / "results/explore_decision_trace.json")
    baseline = _read_json(root / "docs/baseline_results.json")
    session_replays = _read_json(root / "results/public_session_replays.json")

    story = demo["story"]
    certificate = story["submitted_certificate"]
    counterfactual = certificate["minimal_counterfactual_explanation"]
    target = str(story["target_parent_asin"])
    replacement = str(counterfactual["counterfactual_top"])
    wanted = {target, replacement}
    catalog, raw_products = catalog_snapshot(catalog_path, wanted, target)
    if not raw_products:
        raw_products = _fallback_products(story, counterfactual)

    constraints = [str(item) for item in certificate.get("constraints") or []]
    retained = [
        str(item) for item in counterfactual.get("retained_constraints") or []
    ]
    products = {
        target: _normalise_product(raw_products[target], constraints),
        replacement: _normalise_product(raw_products[replacement], retained),
    }

    old_paraphrase = robustness["experiments"][
        "canonical_parser__natural_paraphrase"
    ]
    new_paraphrase = robustness["experiments"][
        "robust_parser__natural_paraphrase"
    ]
    hashes = source_hashes(root)
    test_count = _test_method_count(root)
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()

    return {
        "meta": {
            "schema_version": 2,
            "generated_at": generated_at,
            "source_commit": _source_commit(root),
            "source_fingerprint": source_fingerprint(hashes),
            "source_hashes": hashes,
            "mode": "verified_replay",
            "truth_label": "public_labelled_demo_not_private_evaluation",
        },
        "catalog": catalog,
        "story": {
            "sample_id": story["sample_id"],
            "scenario_type": story["scenario_type"],
            "target_parent_asin": target,
            "target_visibility": "offline_evaluator_only_never_sent_to_agent",
            "comparison": demo["comparison"],
            "old": {
                "label": demo["old"]["label"],
                "outcome": story["old_outcome"],
                "trace": story["old_trace"],
            },
            "submitted": {
                "label": demo["submitted"]["label"],
                "outcome": story["submitted_outcome"],
                "trace": story["submitted_trace"],
            },
            "certificate": certificate,
            "products": products,
        },
        "trace": trace,
        "explore_trace": explore_trace,
        "session_replays": session_replays,
        "proof": {
            "baseline": baseline,
            "public": public,
            "matched": matched,
            "uniform": uniform,
            "paraphrase": {
                "sample_count": int(old_paraphrase["policy_audit"]["session_count"]),
                "old": {
                    key: old_paraphrase[key]
                    for key in (
                        "hit_rate_at_10",
                        "mrr",
                        "mttc",
                        "technical_score",
                    )
                },
                "submitted": {
                    key: new_paraphrase[key]
                    for key in (
                        "hit_rate_at_10",
                        "mrr",
                        "mttc",
                        "technical_score",
                    )
                },
                "impact": demo["simulator_equivalent_impact"],
            },
            "robustness": {
                "status": robustness["status"],
                "catalog_mutation": robustness["catalog_mutation"],
                "aggregate": robustness["aggregate_robustness"],
                "matrix": _variant_matrix(robustness),
                "explanation_audit": new_paraphrase["explanation_audit"],
                "policy_audit": new_paraphrase["policy_audit"],
            },
            "runtime": runtime,
            "tests": {
                "count": test_count,
                "verified_pass": bool(tests_verified),
                "command": "python3 -m unittest discover -v",
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "demo/data/demo_bundle.json")
    parser.add_argument(
        "--tests-verified",
        action="store_true",
        help="record that the complete unittest suite passed immediately before build",
    )
    args = parser.parse_args()
    bundle = build_bundle(
        ROOT,
        args.catalog,
        tests_verified=args.tests_verified,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    meta = bundle["meta"]
    print(
        f"wrote {args.output} | fingerprint={meta['source_fingerprint'][:12]} "
        f"| catalog_rows={bundle['catalog']['row_count']}"
    )


if __name__ == "__main__":
    main()
