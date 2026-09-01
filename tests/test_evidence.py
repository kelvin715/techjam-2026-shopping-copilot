from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src import config
from src.dialog import SessionState, parse
from src.evidence import estimate_question_values, minimal_counterfactual_explanation
from src.policy import choose
from src.shelf import Catalog
from starter.agent import Agent


def _catalog(products: list[dict], directory: str) -> Path:
    path = Path(directory) / "catalog.jsonl"
    path.write_text(
        "".join(json.dumps(product) + "\n" for product in products),
        encoding="utf-8",
    )
    return path


class EvidencePlanningTest(unittest.TestCase):
    def test_counterfactual_other_minimises_expected_survivors(self) -> None:
        products = [
            {
                "parent_asin": asin,
                "title": f"{clue} shirt",
                "features": ["cotton", clue],
                "details": {},
                "description": [],
                "categories": ["Clothing", "Shirts"],
                "store": "Example",
            }
            for asin, clue in (("A", "alpha"), ("B", "beta"), ("C", "gamma"))
        ]
        with tempfile.TemporaryDirectory() as directory:
            catalog = Catalog(_catalog(products, directory))
            values = estimate_question_values(
                catalog,
                [("A", 1.0), ("B", 1.0), ("C", 1.0)],
                [],
                ["other", "material"],
            )

        by_name = {value.attribute: value for value in values}
        self.assertEqual(by_name["other"].expected_remaining, 1.0)
        self.assertEqual(by_name["material"].expected_remaining, 3.0)
        self.assertEqual(by_name["other"].answerability, 1.0)
        self.assertEqual(by_name["material"].answerability, 1.0)
        state = SessionState({})
        original = config.QUESTION_MODE
        try:
            config.QUESTION_MODE = "counterfactual"
            self.assertEqual(choose(state, values), "other")
        finally:
            config.QUESTION_MODE = original

    def test_answerability_aware_mvoi_discounts_unusable_none_branch(self) -> None:
        products = [
            {
                "parent_asin": asin,
                "title": f"{clue} shirt",
                "features": features,
                "details": {},
                "description": [],
                "categories": ["Clothing", "Shirts"],
                "store": "Example",
            }
            for asin, clue, features in (
                ("A", "alpha", ["cotton", "alpha"]),
                ("B", "beta", ["cotton", "beta"]),
                ("C", "gamma", ["wool", "gamma"]),
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            catalog = Catalog(_catalog(products, directory))
            values = estimate_question_values(
                catalog,
                [("A", 1.0), ("B", 1.0), ("C", 1.0)],
                ["cotton"],
                ["other", "material"],
            )
        by_name = {value.attribute: value for value in values}
        self.assertGreater(
            by_name["other"].answerable_metric_voi,
            by_name["material"].answerable_metric_voi,
        )
        state = SessionState({})
        original = config.QUESTION_MODE
        try:
            config.QUESTION_MODE = "answerable_metric_voi"
            self.assertEqual(choose(state, values), "other")
        finally:
            config.QUESTION_MODE = original

    def test_minimal_counterfactual_is_verified_by_reranking(self) -> None:
        products = [
            {
                "parent_asin": "A",
                "title": "red cotton shirt",
                "features": ["red", "cotton"],
                "details": {},
                "description": [],
                "categories": ["Clothing", "Shirts"],
                "store": "Example",
            },
            {
                "parent_asin": "B",
                "title": "blue wool shirt",
                "features": ["blue", "wool"],
                "details": {},
                "description": [],
                "categories": ["Clothing", "Shirts"],
                "store": "Example",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            catalog = Catalog(_catalog(products, directory))
            explanation = minimal_counterfactual_explanation(
                catalog,
                ["A", "B"],
                ["blue"],
                [1.0],
                [],
                "buying",
                "B",
            )
        self.assertTrue(explanation["faithful"])
        self.assertEqual(explanation["minimal_removed_count"], 1)
        self.assertEqual(explanation["counterfactual_top"], "A")

    def test_continuation_excludes_previous_slate_and_certifies_it(self) -> None:
        products = [
            {
                "parent_asin": asin,
                "title": title,
                "features": ["shared"],
                "details": {},
                "description": [],
                "categories": ["Clothing", "Necklaces"],
                "store": "Example",
            }
            for asin, title in (("B", "First"), ("A", "Second"))
        ]
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(_catalog(products, directory))
            agent.reset("session", {})
            first = agent.respond(
                "session",
                "I'm looking for Necklaces, but I'm still exploring.",
                1,
                10,
            )
            second = agent.respond(
                "session",
                "Those options are not quite right yet.",
                2,
                10,
            )
            certificate = agent.explain_last_decision("session")

        self.assertEqual(first["recommendations"], [{"parent_asin": "B"}])
        self.assertEqual(second["recommendations"], [{"parent_asin": "A"}])
        self.assertNotIn("evidence", second)  # strict official contract
        self.assertEqual(certificate["proven_miss_count"], 1)
        self.assertEqual(certificate["excluded_parent_asins"], ["B"])
        self.assertIn("previously_shown_products_excluded", certificate["reason_codes"])
        self.assertEqual(certificate["candidate_evidence"][0]["parent_asin"], "A")

    def test_intent_override_clears_pre_override_misses(self) -> None:
        state = SessionState({})
        state.remember_recommendations(["A", "B"])
        state.confirm_previous_misses()
        self.assertEqual(state.proven_misses, {"A", "B"})

        class _Catalog:
            @staticmethod
            def match_shelf(message: str):
                return None

        parse(
            "Actually, ignore my earlier preference. What I need is: cotton.",
            state,
            _Catalog(),
        )
        self.assertEqual(state.proven_misses, set())

    def test_boundary_deflection_falls_back_to_open_question(self) -> None:
        state = SessionState({})
        state.boundary_signal = True
        values = [
            type("Value", (), {"attribute": "other", "expected_remaining": 9.0})(),
            type("Value", (), {"attribute": "feature", "expected_remaining": 1.0})(),
        ]
        original = config.QUESTION_MODE
        try:
            config.QUESTION_MODE = "counterfactual"
            self.assertEqual(choose(state, values), "other")
        finally:
            config.QUESTION_MODE = original


if __name__ == "__main__":
    unittest.main()
