from __future__ import annotations

import collections
import random
import unittest

from tools.build_category_catalog import FIELDS, normalise
from tools.matched_proxy import percentile, scenario_sequence


class SubmissionToolTest(unittest.TestCase):
    def test_even_sample_median_uses_the_documented_lower_middle_value(self) -> None:
        self.assertEqual(percentile([1, 2, 3, 4], 0.5), 2)

    def test_proxy_scenario_mix_is_exact(self) -> None:
        scenarios = scenario_sequence(800, random.Random(7))
        counts = collections.Counter(name for name, _ in scenarios)
        self.assertEqual(
            counts,
            {"buying": 320, "browsing": 320, "intent_override": 120, "boundary": 40},
        )

    def test_category_catalog_rows_match_the_competition_schema(self) -> None:
        raw = {
            "main_category": "Electronics", "parent_asin": "B000TEST", "title": "USB-C hub",
            "features": ["4 ports", ""], "description": ["Small hub"], "price": "$19.99",
            "categories": ["Electronics", "Computers & Accessories", "USB Hubs"],
            "details": {"Brand": "Acme", "Color": "", "Weight": "2 oz"},
            "average_rating": "4.3", "rating_number": "120", "store": "Acme",
            "images": [{"large": "x"}], "videos": [], "bought_together": None,
        }
        row = normalise(raw)
        self.assertEqual(tuple(row), FIELDS)
        self.assertEqual(row["price"], 19.99)
        self.assertEqual(row["features"], ["4 ports"])
        self.assertEqual(row["details"], {"Brand": "Acme", "Weight": "2 oz"})
        self.assertEqual((row["average_rating"], row["rating_number"]), (4.3, 120))
        # A product without a category path, features or details is not a
        # possible simulator target and is left out.
        self.assertIsNone(normalise({**raw, "categories": []}))
        self.assertIsNone(normalise({**raw, "features": []}))
        self.assertIsNone(normalise({**raw, "details": {}}))
        self.assertIsNone(normalise({**raw, "price": "None", "title": ""}))
        self.assertIsNone(normalise({**raw, "price": "abc"})["price"])


if __name__ == "__main__":
    unittest.main()
