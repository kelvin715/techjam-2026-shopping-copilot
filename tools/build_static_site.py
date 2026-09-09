"""Publish the Live Decision Lab replay as the GitHub Pages homepage.

The export includes only the Decision Lab frontend, with read-only API
responses written to disk. The source-manifest check runs at build time;
a stale replay fails the build. The previous presentation is not published.

The live engine is never exported.  It needs the 50,000-row catalog, which the
data terms keep out of this repository.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.server import bundle_freshness  # noqa: E402
from tools.build_demo_bundle import sha256_file  # noqa: E402

STATIC_DIR = ROOT / "demo" / "static"
DEFAULT_BUNDLE = ROOT / "demo" / "data" / "demo_bundle.json"
DEFAULT_OUTPUT = ROOT / "site"
DOCUMENT_TAG = '<html lang="en">'
STATIC_DOCUMENT_TAG = '<html lang="en" data-deploy="static">'
LAB_ASSETS = ("lab.js", "lab-state.js", "lab.css", "lab-replays.json", "favicon.svg")


def static_health(bundle: dict, fresh: bool, changed: list[str], current: str) -> dict:
    """The /api/health payload the frontend expects, resolved at build time."""
    meta = bundle.get("meta", {})
    return {
        "status": "ok",
        "replay_status": "verified" if fresh else "stale",
        "replay_fresh": fresh,
        "changed_sources": changed,
        "source_commit": meta.get("source_commit"),
        "source_fingerprint": meta.get("source_fingerprint"),
        "current_fingerprint": current or None,
        "live_requested": False,
        "live_available": False,
        "live_ready": False,
        "live_error": None,
        "catalog_present": False,
        "catalog_verified": False,
        "catalog_sha256": bundle.get("catalog", {}).get("sha256"),
        "deployment": "static",
        "verified_at_build": datetime.now(timezone.utc).isoformat(),
    }


def build(bundle_path: Path, output: Path, allow_stale: bool = False) -> dict:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    fresh, changed, current = bundle_freshness(ROOT, bundle)
    if not fresh and not allow_stale:
        raise SystemExit(
            "refusing to export a stale replay; rebuild the bundle first "
            f"(changed: {', '.join(changed) or 'source fingerprint mismatch'})"
        )

    if output.exists():
        shutil.rmtree(output)
    (output / "data").mkdir(parents=True)

    # Publish only this frontend: copying the whole directory would bring the
    # retired presentation and session explorer back into each deployment.
    for name in LAB_ASSETS:
        shutil.copy2(STATIC_DIR / name, output / name)

    index = (STATIC_DIR / "lab.html").read_text(encoding="utf-8")
    if DOCUMENT_TAG not in index:
        raise SystemExit(f"cannot mark the static build: {DOCUMENT_TAG} not in lab.html")
    index = index.replace(DOCUMENT_TAG, STATIC_DOCUMENT_TAG, 1)
    (output / "index.html").write_text(index, encoding="utf-8")
    # Keep existing links to the new demo working; both URLs open the lab.
    (output / "lab.html").write_text(index, encoding="utf-8")

    health = static_health(bundle, fresh, changed, current)
    (output / "data" / "demo_bundle.json").write_text(
        json.dumps(bundle, separators=(",", ":")), encoding="utf-8"
    )
    (output / "data" / "health.json").write_text(
        json.dumps(health, separators=(",", ":")), encoding="utf-8"
    )
    # GitHub Pages skips Jekyll processing when this file is present.
    (output / ".nojekyll").write_text("", encoding="utf-8")
    return health


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help="export even when the replay no longer matches the working tree",
    )
    args = parser.parse_args()
    health = build(args.bundle, args.output.resolve(), args.allow_stale)
    files = sorted(p for p in args.output.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"Static site: {args.output} ({len(files)} files, {total / 1024:.0f} KB)")
    print(f"Replay: {health['replay_status'].upper()} | Live engine: not exported")
    print(f"Bundle sha256: {sha256_file(args.bundle)[:16]}…")
    print("Preview: python3 -m http.server 8080 --directory " + str(args.output))


if __name__ == "__main__":
    main()
