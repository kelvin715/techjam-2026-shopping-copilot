from __future__ import annotations

import collections
import random
import unittest

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


if __name__ == "__main__":
    unittest.main()
