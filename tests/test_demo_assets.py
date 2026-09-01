from __future__ import annotations

import unittest

from tools.demo_walkthrough import simulator_equivalent_impact, trace_for_story


class DemoAssetTest(unittest.TestCase):
    def test_simulator_translation_is_explicit_and_arithmetic_is_reproducible(self) -> None:
        old = {"sample_count": 100, "hit_rate_at_10": 0.88, "mrr": 0.50,
               "technical_score": 0.72}
        new = {"sample_count": 100, "hit_rate_at_10": 1.0, "mrr": 0.99,
               "technical_score": 0.97}
        impact = simulator_equivalent_impact(old, new, 417, 202)
        self.assertEqual(impact["additional_hit_sessions_per_10k"], 1200)
        self.assertEqual(impact["fewer_agent_calls_per_session"], 2.15)
        self.assertIn("simulator_equivalent", impact["scope"])

    def test_trace_marks_target_rank_without_changing_recommendations(self) -> None:
        trace = [{
            "turn": 1,
            "transformed_message": "I need a blue scarf.",
            "response": {
                "message": "Here are options.",
                "ask_attribute": "other",
                "recommendations": [
                    {"parent_asin": "A"}, {"parent_asin": "TARGET"}
                ],
            },
        }]
        story = trace_for_story(trace, "TARGET")
        self.assertEqual(story[0]["recommendations"], ["A", "TARGET"])
        self.assertEqual(story[0]["target_rank"], 2)


if __name__ == "__main__":
    unittest.main()
