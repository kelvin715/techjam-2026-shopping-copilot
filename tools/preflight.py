"""Fast, catalog-free submission contract check."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import Agent


ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other", None,
}
REQUIRED_FILES = ("agent.py", "requirements.txt", "README.md", "src")


def main() -> None:
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).exists()]
    if missing:
        raise SystemExit(f"missing required submission files: {missing}")

    products = [
        {
            "parent_asin": "A",
            "title": "Cotton blue scarf",
            "features": ["soft cotton", "color: blue"],
            "details": {"Department": "Women"},
            "description": [],
            "categories": ["Clothing", "Scarves"],
            "store": "Example",
            "average_rating": 4.8,
            "rating_number": 100,
        },
        {
            "parent_asin": "B",
            "title": "Wool red scarf",
            "features": ["warm wool", "color: red"],
            "details": {"Department": "Women"},
            "description": [],
            "categories": ["Clothing", "Scarves"],
            "store": "Example",
            "average_rating": 4.0,
            "rating_number": 20,
        },
    ]
    with tempfile.TemporaryDirectory() as directory:
        catalog = Path(directory) / "catalog.jsonl"
        catalog.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        agent = Agent(catalog)
        agent.reset("preflight", {"preference_tags": []})
        response = agent.respond(
            "preflight",
            "I'm looking for Scarves. A key requirement is: cotton.",
            1,
            10,
        )

    assert isinstance(response, dict)
    assert isinstance(response.get("message"), str)
    assert response.get("ask_attribute") in ALLOWED_ATTRIBUTES
    recommendations = response.get("recommendations")
    assert isinstance(recommendations, list) and len(recommendations) <= 10
    asins = [item["parent_asin"] for item in recommendations]
    assert all(isinstance(asin, str) and asin for asin in asins)
    assert len(asins) == len(set(asins))
    usage = response.get("usage")
    assert isinstance(usage, dict)
    assert usage.get("prompt_tokens") == 0 and usage.get("completion_tokens") == 0
    print("PASS: required files, Agent import, response schema, ASIN uniqueness, and usage")


if __name__ == "__main__":
    main()
