from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.build_static_site import DEFAULT_BUNDLE, build

ROOT = Path(__file__).resolve().parent.parent


class StaticSiteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.output = Path(self.tmp.name) / "site"
        # Freshness is asserted by the demo tests; here the export only has to
        # stay structurally correct while sources are being edited.
        self.health = build(DEFAULT_BUNDLE, self.output, allow_stale=True)
        self.addCleanup(self.tmp.cleanup)

    def test_export_is_self_contained(self) -> None:
        for name in (
            "index.html",
            "app.js",
            "components.js",
            "state.js",
            "styles.css",
            "favicon.svg",
            ".nojekyll",
            "data/demo_bundle.json",
            "data/health.json",
        ):
            self.assertTrue((self.output / name).is_file(), name)

    def test_index_marks_the_static_deployment_and_uses_relative_assets(self) -> None:
        index = (self.output / "index.html").read_text(encoding="utf-8")
        self.assertIn('data-deploy="static"', index)
        self.assertIn('src="./app.js"', index)
        self.assertIn('href="./styles.css"', index)
        self.assertNotIn('src="/app.js"', index)
        self.assertNotIn('href="/styles.css"', index)

    def test_frontend_reads_files_when_the_document_is_marked_static(self) -> None:
        app = (self.output / "app.js").read_text(encoding="utf-8")
        self.assertIn('dataset.deploy === "static"', app)
        self.assertIn('"./data/demo_bundle.json"', app)
        self.assertIn('"./data/health.json"', app)

    def test_exported_health_reports_a_replay_without_a_live_engine(self) -> None:
        health = json.loads((self.output / "data" / "health.json").read_text())
        self.assertEqual(health, self.health)
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["deployment"], "static")
        self.assertFalse(health["live_available"])
        self.assertFalse(health["catalog_present"])
        self.assertEqual(
            health["replay_status"], "verified" if health["replay_fresh"] else "stale"
        )

    def test_exported_bundle_matches_the_source_bundle(self) -> None:
        exported = json.loads(
            (self.output / "data" / "demo_bundle.json").read_text(encoding="utf-8")
        )
        self.assertEqual(exported, json.loads(DEFAULT_BUNDLE.read_text(encoding="utf-8")))

    def test_stale_replay_is_not_exported_by_default(self) -> None:
        bundle = json.loads(DEFAULT_BUNDLE.read_text(encoding="utf-8"))
        bundle["meta"]["source_hashes"]["agent.py"] = "0" * 64
        stale_bundle = Path(self.tmp.name) / "stale_bundle.json"
        stale_bundle.write_text(json.dumps(bundle), encoding="utf-8")
        with self.assertRaises(SystemExit):
            build(stale_bundle, Path(self.tmp.name) / "stale_site")


if __name__ == "__main__":
    unittest.main()
