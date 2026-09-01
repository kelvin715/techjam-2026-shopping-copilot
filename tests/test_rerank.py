from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src import config
from src.dialog import SessionState, parse
from src.rank import diversify_evidence_ties, score_candidates
from src.shelf import Catalog


class _NoShelfCatalog:
    @staticmethod
    def match_shelf(message: str) -> None:
        return None


class StructuredRerankTest(unittest.TestCase):
    def test_tail_exploration_never_promotes_weaker_evidence(self) -> None:
        final = [("POPULAR", 2.0), ("TAIL_A", 1.6), ("TAIL_B", 1.5), ("WEAK", 1.4)]
        evidence = [("POPULAR", 1.0), ("TAIL_A", 1.0), ("TAIL_B", 1.0), ("WEAK", 0.9)]
        diversified = diversify_evidence_ties(final, evidence, 1e-9)
        self.assertEqual({asin for asin, _ in diversified[:3]}, {
            "POPULAR", "TAIL_A", "TAIL_B"
        })
        self.assertEqual(diversified[-1][0], "WEAK")
        self.assertEqual(diversified, diversify_evidence_ties(final, evidence, 1e-9))

    def _catalog(self, directory: str, products: list[dict]) -> Catalog:
        path = Path(directory) / "catalog.jsonl"
        path.write_text("".join(json.dumps(product) + "\n" for product in products), encoding="utf-8")
        return Catalog(path)

    def test_typed_material_prefers_the_first_catalog_material(self) -> None:
        products = [
            {
                "parent_asin": "B", "title": "Wool item with cotton trim",
                "features": [], "details": {}, "description": [],
                "categories": ["Clothing", "Scarves"], "store": "Example",
            },
            {
                "parent_asin": "A", "title": "Cotton item with wool trim",
                "features": [], "details": {}, "description": [],
                "categories": ["Clothing", "Scarves"], "store": "Example",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._catalog(directory, products)
            original = config.TYPED_WEIGHT
            try:
                config.TYPED_WEIGHT = 1.0
                ranked = score_candidates(catalog, ["B", "A"], ["cotton"])
            finally:
                config.TYPED_WEIGHT = original
        self.assertEqual(ranked[0][0], "A")

    def test_synthetic_color_does_not_require_a_literal_color_field(self) -> None:
        products = [
            {
                "parent_asin": "TARGET", "title": "Grey cotton shirt",
                "features": ["100% Cotton"], "details": {}, "description": [],
                "categories": ["Clothing", "Shirts"], "store": "Example",
            },
            {
                "parent_asin": "LABELED", "title": "Cotton shirt",
                "features": ["100% Cotton"], "details": {"Color": "Grey"},
                "description": [], "categories": ["Clothing", "Shirts"],
                "store": "Example",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._catalog(directory, products)
            ranked = score_candidates(
                catalog,
                ["TARGET", "LABELED"],
                ["color: grey"],
                popularity_weight=0.0,
            )
        self.assertAlmostEqual(dict(ranked)["TARGET"], dict(ranked)["LABELED"])

    def test_popularity_breaks_an_evidence_tie(self) -> None:
        products = [
            {
                "parent_asin": "B", "title": "Shared clue", "features": [],
                "details": {}, "description": [], "categories": ["Clothing", "Scarves"],
                "store": "Example", "average_rating": 3.0, "rating_number": 1,
            },
            {
                "parent_asin": "A", "title": "Shared clue", "features": [],
                "details": {}, "description": [], "categories": ["Clothing", "Scarves"],
                "store": "Example", "average_rating": 4.9, "rating_number": 1000,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._catalog(directory, products)
            original = (config.POPULARITY_WEIGHT, config.POPULARITY_WINDOW)
            try:
                config.POPULARITY_WEIGHT, config.POPULARITY_WINDOW = 0.5, 0.15
                ranked = score_candidates(
                    catalog, ["B", "A"], ["shared clue"], scenario="buying"
                )
            finally:
                config.POPULARITY_WEIGHT, config.POPULARITY_WINDOW = original
        self.assertEqual(ranked[0][0], "A")

    def test_override_decays_only_the_provisional_constraint(self) -> None:
        state = SessionState({})
        state.add("old feature", provisional=True)
        parse(
            "Actually, ignore my earlier preference. What I need is: cotton.",
            state,
            _NoShelfCatalog(),
        )
        self.assertEqual(state.constraints, ["old feature", "cotton"])
        self.assertEqual(state.constraint_weights(), [config.OVERRIDE_DECAY, 1.0])

    def test_attribute_answer_marks_signature_positions_unreliable(self) -> None:
        state = SessionState({})
        state.asked.append("feature")
        parse(
            "For that, what matters is: Pull On closure; Machine Wash.",
            state,
            _NoShelfCatalog(),
        )
        self.assertFalse(state.signature_positions_reliable)


if __name__ == "__main__":
    unittest.main()
