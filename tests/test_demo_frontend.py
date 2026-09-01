from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from demo.server import DemoRuntime, bundle_freshness, make_handler
from tools.build_demo_bundle import (
    ROOT,
    build_bundle,
    catalog_snapshot,
    sha256_file,
    source_fingerprint,
)


class DemoBundleTest(unittest.TestCase):
    def test_bundle_copies_measured_story_and_marks_target_as_evaluator_only(self) -> None:
        bundle = build_bundle(
            ROOT,
            generated_at="2026-08-31T00:00:00+00:00",
        )
        story = bundle["story"]
        self.assertEqual(story["sample_id"], "public_0029")
        self.assertEqual(story["submitted"]["outcome"]["best_rank"], 1)
        self.assertIn("never_sent_to_agent", story["target_visibility"])
        self.assertTrue(
            story["certificate"]["minimal_counterfactual_explanation"]["faithful"]
        )
        self.assertEqual(bundle["explore_trace"]["sample_id"], "public_0187")
        self.assertEqual(len(bundle["explore_trace"]["turns"]), 4)
        self.assertEqual(bundle["explore_trace"]["outcome"]["first_hit_turn"], 4)
        self.assertEqual(bundle["proof"]["public"]["technical_score"], 0.9804)
        replays = bundle["session_replays"]
        self.assertEqual(replays["sample_count"], 200)
        self.assertEqual(replays["summary"]["rank_distribution"], {"1": 200})
        self.assertEqual(replays["target_join"], "post_response_evaluator_only")

    def test_catalog_snapshot_keeps_only_story_products_and_counts_real_shelf(self) -> None:
        products = [
            {
                "parent_asin": "TARGET",
                "title": "Target tunic",
                "categories": ["Clothing", "Women", "Tops, Tees & Blouses", "Tunics"],
            },
            {
                "parent_asin": "REPLACEMENT",
                "title": "Replacement tunic",
                "categories": ["Clothing", "Women", "Tops, Tees & Blouses", "Tunics"],
            },
            {
                "parent_asin": "OTHER",
                "title": "Other product",
                "categories": ["Clothing", "Women", "Dresses"],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text(
                "".join(json.dumps(product) + "\n" for product in products),
                encoding="utf-8",
            )
            catalog, selected = catalog_snapshot(
                path, {"TARGET", "REPLACEMENT"}, "TARGET"
            )
        self.assertEqual(catalog["row_count"], 3)
        self.assertEqual(catalog["shelf_count"], 2)
        self.assertEqual(set(selected), {"TARGET", "REPLACEMENT"})

    def test_freshness_detects_a_changed_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("first", encoding="utf-8")
            hashes = {"source.txt": sha256_file(source)}
            bundle = {
                "meta": {
                    "source_hashes": hashes,
                    "source_fingerprint": source_fingerprint(hashes),
                }
            }
            fresh, changed, _ = bundle_freshness(root, bundle)
            self.assertTrue(fresh)
            self.assertEqual(changed, [])
            source.write_text("second", encoding="utf-8")
            fresh, changed, _ = bundle_freshness(root, bundle)
            self.assertFalse(fresh)
            self.assertEqual(changed, ["source.txt"])


class DemoServerTest(unittest.TestCase):
    def test_runtime_reloads_a_rebuilt_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle_path = Path(directory) / "bundle.json"
            bundle_path.write_text(
                json.dumps({"version": 1, "padding": "old"}), encoding="utf-8"
            )
            runtime = DemoRuntime(bundle_path)
            self.assertEqual(runtime.replay_bundle()["version"], 1)
            bundle_path.write_text(
                json.dumps({"version": 2, "padding": "new-and-longer"}),
                encoding="utf-8",
            )
            self.assertEqual(runtime.replay_bundle()["version"], 2)

    def test_replay_and_health_endpoints_work_without_catalog(self) -> None:
        bundle = build_bundle(ROOT, generated_at="2026-08-31T00:00:00+00:00")
        with tempfile.TemporaryDirectory() as directory:
            bundle_path = Path(directory) / "bundle.json"
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
            runtime = DemoRuntime(bundle_path)
            try:
                server = ThreadingHTTPServer(
                    ("127.0.0.1", 0), make_handler(runtime)
                )
            except PermissionError:
                self.skipTest("local sockets are disabled by the current sandbox")
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                origin = f"http://127.0.0.1:{server.server_port}"
                with urllib.request.urlopen(f"{origin}/api/health", timeout=5) as response:
                    health = json.load(response)
                with urllib.request.urlopen(f"{origin}/api/demo", timeout=5) as response:
                    replay = json.load(response)
                with urllib.request.urlopen(f"{origin}/", timeout=5) as response:
                    html = response.read().decode()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
        self.assertEqual(health["status"], "ok")
        self.assertTrue(health["replay_fresh"])
        self.assertFalse(health["live_available"])
        self.assertEqual(replay["story"]["sample_id"], "public_0029")
        self.assertIn("Ask, Rank, Commit", html)

    def test_static_frontend_has_no_external_runtime_dependencies(self) -> None:
        index = (ROOT / "demo/static/index.html").read_text(encoding="utf-8")
        app = (ROOT / "demo/static/app.js").read_text(encoding="utf-8")
        components = (ROOT / "demo/static/components.js").read_text(encoding="utf-8")
        styles = (ROOT / "demo/static/styles.css").read_text(encoding="utf-8")
        combined = "\n".join((index, app, components))
        self.assertNotIn("https://", combined)
        self.assertNotIn("http://", combined)
        self.assertIn('id="main-content"', index)
        self.assertIn('"walk-4"', components)
        self.assertIn('id: "method", label: "Method"', (ROOT / "demo/static/state.js").read_text(encoding="utf-8"))
        self.assertIn("Algorithm 1", components)
        self.assertIn("ANSWERABLE_MVOI", components)
        self.assertIn("Answerability-aware Metric Value of Information", components)
        self.assertIn("Keep &lt;none&gt; honest", components)
        self.assertNotIn('label: "Counterfactual"', (ROOT / "demo/static/state.js").read_text(encoding="utf-8"))
        self.assertNotIn('label: "Ablation"', (ROOT / "demo/static/state.js").read_text(encoding="utf-8"))
        self.assertNotIn('label: "Robustness"', (ROOT / "demo/static/state.js").read_text(encoding="utf-8"))
        self.assertIn("Structured output · sent with this reply", components)
        self.assertIn("SUBMITTED", components)
        self.assertIn("Clue combination match", components)
        self.assertIn("it cannot override clues", components)
        self.assertNotIn('["explore", "Decision trace"]', components)
        self.assertIn("prefers-reduced-motion", (ROOT / "demo/static/styles.css").read_text(encoding="utf-8"))


class PlaygroundTest(unittest.TestCase):
    """The playground drives the real Agent, so it must fail loudly and safely."""

    def test_catalog_facts_name_every_submitted_product_once(self) -> None:
        class _Catalog:
            title = {"A": "leather penny loafer"}
            shelf_of = {"A": "Shoes Loafers & Slip-Ons"}
            price = {"A": 48.5}
            rating = {"A": 4.4}
            rating_count = {"A": 132}

        class _Agent:
            catalog = _Catalog()

        response = {
            "recommendations": [
                {"parent_asin": "A"}, {"parent_asin": "A"}, {"parent_asin": "GONE"}
            ]
        }
        facts = DemoRuntime._catalog_facts(_Agent(), response)
        self.assertEqual(set(facts), {"A", "GONE"})
        self.assertEqual(facts["A"]["title"], "leather penny loafer")
        self.assertEqual(facts["A"]["rating_count"], 132)
        # An ASIN the index does not know degrades to empty display text rather
        # than raising inside a live demo.
        self.assertEqual(facts["GONE"]["title"], "")
        self.assertIsNone(facts["GONE"]["price"])

    def test_live_endpoints_refuse_when_the_engine_is_not_running(self) -> None:
        bundle = build_bundle(ROOT, generated_at="2026-08-31T00:00:00+00:00")
        with tempfile.TemporaryDirectory() as directory:
            bundle_path = Path(directory) / "bundle.json"
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
            runtime = DemoRuntime(bundle_path)
            try:
                server = ThreadingHTTPServer(
                    ("127.0.0.1", 0), make_handler(runtime)
                )
            except PermissionError:
                self.skipTest("local sockets are disabled by the current sandbox")
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                origin = f"http://127.0.0.1:{server.server_port}"
                request = urllib.request.Request(
                    f"{origin}/api/live/reset",
                    data=json.dumps({"session_id": "playground-test"}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(request, timeout=5)
                error = json.loads(raised.exception.read())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
        self.assertEqual(error["status"], "error")
        self.assertIn("live engine", error["error"])

    def test_playground_page_is_wired_to_the_live_api(self) -> None:
        state = (ROOT / "demo/static/state.js").read_text(encoding="utf-8")
        app = (ROOT / "demo/static/app.js").read_text(encoding="utf-8")
        components = (ROOT / "demo/static/components.js").read_text(encoding="utf-8")
        self.assertIn('id: "sessions", label: "Sessions"', state)
        for endpoint in ("/api/live/reset", "/api/live/respond", "/api/live/explain"):
            self.assertIn(endpoint, app)
        # The static export has no engine, so the page must offer the command
        # instead of a dead input box.
        self.assertIn("health.live_available", components)
        self.assertIn(
            "python3 demo/server.py --catalog data/catalog.jsonl --live --prewarm",
            components,
        )
        # The turn horizon is a protocol rule, not a UI preference.
        self.assertIn("playground.turn >= 10", app)


class SessionExplorerTest(unittest.TestCase):
    """The static explorer must show evaluator truth without leaking it to ARC."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.replays = json.loads(
            (ROOT / "results/public_session_replays.json").read_text(encoding="utf-8")
        )

    def test_all_public_outcomes_have_a_matching_post_response_trace(self) -> None:
        sessions = self.replays["sessions"]
        self.assertEqual(len(sessions), 200)
        self.assertEqual(len({row["sample_id"] for row in sessions}), 200)
        non_rank_one = []
        for session in sessions:
            self.assertIn("never_sent_to_agent", session["target_visibility"])
            hits = [turn for turn in session["turns"] if turn["hit_this_turn"]]
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0]["turn"], session["outcome"]["first_hit_turn"])
            self.assertEqual(hits[0]["target_rank"], session["outcome"]["best_rank"])
            if session["outcome"]["best_rank"] != 1:
                non_rank_one.append(session["sample_id"])
        self.assertEqual(non_rank_one, [])

    def test_session_explorer_is_static_first_and_live_optional(self) -> None:
        state = (ROOT / "demo/static/state.js").read_text(encoding="utf-8")
        app = (ROOT / "demo/static/app.js").read_text(encoding="utf-8")
        components = (ROOT / "demo/static/components.js").read_text(encoding="utf-8")
        styles = (ROOT / "demo/static/styles.css").read_text(encoding="utf-8")
        self.assertIn('mode: requestedSessionMode', state)
        self.assertIn('id="session-filter"', components)
        self.assertIn('id="session-select"', components)
        self.assertIn("Pick any session. Replay every turn.", components)
        self.assertIn("Ground truth is joined only after", components)
        self.assertIn("bundle update required", components)
        self.assertIn('data-action="session-mode-live"', components)
        self.assertIn('case "session-random"', app)
        self.assertIn('case "session-set-turn"', app)
        self.assertIn(".section-label > span:not(.badge)", styles)
        self.assertIn("white-space: nowrap", styles)


if __name__ == "__main__":
    unittest.main()
