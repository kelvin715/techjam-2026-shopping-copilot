from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src import config
from src.rank import score_candidates
from src.shelf import Catalog, intent_signature, searchable_text


class CanonicalSignatureTest(unittest.TestCase):
    def _catalog(self, directory: str, products: list[dict]) -> Catalog:
        path = Path(directory) / "catalog.jsonl"
        path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        return Catalog(path)

    def test_signature_collapses_case_only_duplicates_seen_by_memory(self) -> None:
        product = {
            "parent_asin": "A",
            "title": "Cotton cap",
            "features": ["Cotton"],
            "details": {"Origin": "Imported"},
            "description": [],
            "categories": ["Clothing", "Hats"],
            "store": "Example",
        }
        self.assertEqual(
            intent_signature(product, searchable_text(product)),
            ("cotton", "origin: imported"),
        )

    def test_ordered_signature_breaks_a_full_text_tie(self) -> None:
        products = [
            {
                "parent_asin": "B",
                "title": "Shared item",
                "features": ["beta clue", "alpha clue"],
                "details": {},
                "description": [],
                "categories": ["Clothing", "Hats"],
                "store": "Example",
            },
            {
                "parent_asin": "A",
                "title": "Shared item",
                "features": ["alpha clue", "beta clue"],
                "details": {},
                "description": [],
                "categories": ["Clothing", "Hats"],
                "store": "Example",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._catalog(directory, products)
            original = (
                config.TYPED_WEIGHT,
                config.SIGNATURE_WEIGHT,
                config.POPULARITY_WEIGHT,
            )
            try:
                config.TYPED_WEIGHT = 0.0
                config.SIGNATURE_WEIGHT = 1.0
                config.POPULARITY_WEIGHT = 0.0
                ranked = score_candidates(
                    catalog, ["B", "A"], ["alpha clue"], scenario="buying"
                )
            finally:
                (
                    config.TYPED_WEIGHT,
                    config.SIGNATURE_WEIGHT,
                    config.POPULARITY_WEIGHT,
                ) = original
        self.assertEqual(ranked[0][0], "A")

    def test_override_uses_membership_without_stale_positions(self) -> None:
        products = [
            {
                "parent_asin": asin,
                "title": "Shared item",
                "features": features,
                "details": {},
                "description": [],
                "categories": ["Clothing", "Hats"],
                "store": "Example",
            }
            for asin, features in (
                ("B", ["beta clue", "alpha clue"]),
                ("A", ["alpha clue", "beta clue"]),
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._catalog(directory, products)
            original = (
                config.TYPED_WEIGHT,
                config.SIGNATURE_WEIGHT,
                config.POPULARITY_WEIGHT,
            )
            try:
                config.TYPED_WEIGHT = 0.0
                config.SIGNATURE_WEIGHT = 1.0
                config.POPULARITY_WEIGHT = 0.0
                ranked = score_candidates(
                    catalog,
                    ["B", "A"],
                    ["alpha clue"],
                    scenario="intent_override",
                )
            finally:
                (
                    config.TYPED_WEIGHT,
                    config.SIGNATURE_WEIGHT,
                    config.POPULARITY_WEIGHT,
                ) = original
        self.assertEqual([asin for asin, _ in ranked], ["B", "A"])


if __name__ == "__main__":
    unittest.main()
