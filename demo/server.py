"""Dependency-free local server for the judge-facing decision demo."""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "demo" / "static"
DEFAULT_BUNDLE = ROOT / "demo" / "data" / "demo_bundle.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_demo_bundle import sha256_file, source_fingerprint

SESSION_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")
MAX_REQUEST_BYTES = 64 * 1024


def bundle_freshness(root: Path, bundle: dict) -> tuple[bool, list[str], str]:
    expected = bundle.get("meta", {}).get("source_hashes", {})
    if not isinstance(expected, dict) or not expected:
        return False, ["source manifest missing"], ""
    changed: list[str] = []
    current: dict[str, str] = {}
    for relative, old_hash in sorted(expected.items()):
        path = root / relative
        if not path.is_file():
            changed.append(f"{relative} (missing)")
            continue
        new_hash = sha256_file(path)
        current[relative] = new_hash
        if new_hash != old_hash:
            changed.append(relative)
    fingerprint = source_fingerprint(current) if len(current) == len(expected) else ""
    expected_fingerprint = str(
        bundle.get("meta", {}).get("source_fingerprint") or ""
    )
    return not changed and fingerprint == expected_fingerprint, changed, fingerprint


class DemoRuntime:
    """Shared read-only replay state plus an optional, locked live Agent."""

    def __init__(
        self,
        bundle_path: Path = DEFAULT_BUNDLE,
        catalog_path: Path | None = None,
        live_requested: bool = False,
        prewarm: bool = False,
    ) -> None:
        self.bundle_path = bundle_path.resolve()
        self.bundle = json.loads(self.bundle_path.read_text(encoding="utf-8"))
        bundle_stat = self.bundle_path.stat()
        self._bundle_signature = (bundle_stat.st_mtime_ns, bundle_stat.st_size)
        self._bundle_lock = threading.RLock()
        self.catalog_path = catalog_path.resolve() if catalog_path else None
        self.live_requested = bool(live_requested)
        self._agent: Any = None
        self._agent_error: str | None = None
        self._agent_lock = threading.RLock()
        self._sessions: set[str] = set()
        self.replay_fresh, self.changed_sources, self.current_fingerprint = (
            bundle_freshness(ROOT, self.bundle)
        )
        self.catalog_present = bool(
            self.catalog_path and self.catalog_path.is_file()
        )
        expected_catalog_hash = self.bundle.get("catalog", {}).get("sha256")
        self.catalog_hash = (
            sha256_file(self.catalog_path) if self.catalog_present else None
        )
        self.catalog_verified = bool(
            self.catalog_hash
            and expected_catalog_hash
            and self.catalog_hash == expected_catalog_hash
        )
        if prewarm and self.live_available:
            self.ensure_agent()

    def replay_bundle(self) -> dict:
        """Return the latest complete bundle, reloading a rebuilt file safely."""
        with self._bundle_lock:
            try:
                bundle_stat = self.bundle_path.stat()
                signature = (bundle_stat.st_mtime_ns, bundle_stat.st_size)
                if signature != self._bundle_signature:
                    candidate = json.loads(
                        self.bundle_path.read_text(encoding="utf-8")
                    )
                    if not isinstance(candidate, dict):
                        raise ValueError("demo bundle must be a JSON object")
                    self.bundle = candidate
                    self._bundle_signature = signature
            except (OSError, json.JSONDecodeError, ValueError):
                # A build may be between truncate and write when a request
                # arrives. Keep serving the previous complete bundle; because
                # the signature is not advanced, the next request retries.
                pass
            return self.bundle

    @property
    def live_available(self) -> bool:
        return self.live_requested and self.catalog_present

    def ensure_agent(self) -> Any:
        if not self.live_available:
            raise RuntimeError("live engine requires --live and a readable --catalog")
        with self._agent_lock:
            if self._agent is None and self._agent_error is None:
                try:
                    from agent import Agent

                    self._agent = Agent(self.catalog_path)
                except Exception as exc:  # pragma: no cover - defensive UI path
                    self._agent_error = f"{type(exc).__name__}: {exc}"
            if self._agent is None:
                raise RuntimeError(self._agent_error or "live engine failed to start")
            return self._agent

    def health(self) -> dict:
        bundle = self.replay_bundle()
        # Re-evaluate on every health request so edits made while the local
        # server is running immediately remove the green verified state.
        self.replay_fresh, self.changed_sources, self.current_fingerprint = (
            bundle_freshness(ROOT, bundle)
        )
        meta = bundle.get("meta", {})
        expected_catalog_hash = bundle.get("catalog", {}).get("sha256")
        self.catalog_verified = bool(
            self.catalog_hash
            and expected_catalog_hash
            and self.catalog_hash == expected_catalog_hash
        )
        return {
            "status": "ok",
            "replay_status": "verified" if self.replay_fresh else "stale",
            "replay_fresh": self.replay_fresh,
            "changed_sources": self.changed_sources,
            "source_commit": meta.get("source_commit"),
            "source_fingerprint": meta.get("source_fingerprint"),
            "current_fingerprint": self.current_fingerprint or None,
            "live_requested": self.live_requested,
            "live_available": self.live_available,
            "live_ready": self._agent is not None,
            "live_error": self._agent_error,
            "catalog_present": self.catalog_present,
            "catalog_verified": self.catalog_verified,
            "catalog_sha256": self.catalog_hash,
        }

    def live_reset(self, payload: dict) -> dict:
        session_id = _session_id(payload)
        profile = payload.get("user_profile", {"preference_tags": []})
        if not isinstance(profile, dict):
            raise ValueError("user_profile must be an object")
        with self._agent_lock:
            self.ensure_agent().reset(session_id, profile)
            self._sessions.add(session_id)
        return {"status": "ready", "session_id": session_id}

    def live_respond(self, payload: dict) -> dict:
        session_id = _session_id(payload)
        message = payload.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        if len(message) > 4000:
            raise ValueError("message must be at most 4000 characters")
        turn = payload.get("turn")
        top_k = payload.get("top_k", 10)
        if not isinstance(turn, int) or not 1 <= turn <= 10:
            raise ValueError("turn must be an integer from 1 to 10")
        if not isinstance(top_k, int) or not 1 <= top_k <= 10:
            raise ValueError("top_k must be an integer from 1 to 10")
        with self._agent_lock:
            if session_id not in self._sessions:
                raise ValueError("reset the live session before responding")
            agent = self.ensure_agent()
            response = agent.respond(session_id, message.strip(), turn, top_k)
            # ``response`` is returned untouched: it is exactly what the
            # evaluator receives. Display facts travel in a separate field so
            # the playground can name a product without editing the contract.
            facts = self._catalog_facts(agent, response)
        return {
            "status": "ok",
            "session_id": session_id,
            "response": response,
            "catalog_facts": facts,
        }

    @staticmethod
    def _catalog_facts(agent: Any, response: dict) -> dict:
        catalog = getattr(agent, "catalog", None)
        if catalog is None:
            return {}
        facts: dict[str, dict] = {}
        for item in response.get("recommendations", []):
            asin = item.get("parent_asin") if isinstance(item, dict) else None
            if not isinstance(asin, str) or asin in facts:
                continue
            facts[asin] = {
                "title": catalog.title.get(asin, ""),
                "shelf": catalog.shelf_of.get(asin, ""),
                "price": catalog.price.get(asin),
                "rating": catalog.rating.get(asin, 0.0),
                "rating_count": catalog.rating_count.get(asin, 0),
            }
        return facts

    def live_explain(self, session_id: str) -> dict:
        if not SESSION_ID.fullmatch(session_id):
            raise ValueError("invalid session_id")
        with self._agent_lock:
            if session_id not in self._sessions:
                raise ValueError("unknown live session")
            certificate = self.ensure_agent().explain_last_decision(session_id)
        return {
            "status": "ok",
            "session_id": session_id,
            "certificate": certificate,
        }


def _session_id(payload: dict) -> str:
    value = payload.get("session_id")
    if not isinstance(value, str) or not SESSION_ID.fullmatch(value):
        raise ValueError("session_id must be 1-96 safe identifier characters")
    return value


def make_handler(runtime: DemoRuntime) -> type[SimpleHTTPRequestHandler]:
    class DemoHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

        def end_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            super().end_headers()

        def _send_json(self, value: dict, status: int = HTTPStatus.OK) -> None:
            body = json.dumps(value, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _payload(self) -> dict:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("invalid Content-Length") from exc
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("request body must be 1-65536 bytes")
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ValueError("request JSON must be an object")
            return value

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
            parsed = urlsplit(self.path)
            if parsed.path == "/api/demo":
                self._send_json(runtime.replay_bundle())
                return
            if parsed.path == "/api/health":
                self._send_json(runtime.health())
                return
            if parsed.path == "/api/live/explain":
                try:
                    session_id = parse_qs(parsed.query).get("session_id", [""])[0]
                    self._send_json(runtime.live_explain(session_id))
                except (ValueError, RuntimeError) as exc:
                    self._send_json(
                        {"status": "error", "error": str(exc)},
                        HTTPStatus.BAD_REQUEST,
                    )
                return
            if parsed.path == "/":
                self.path = "/index.html"
            super().do_GET()

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
            parsed = urlsplit(self.path)
            actions = {
                "/api/live/reset": runtime.live_reset,
                "/api/live/respond": runtime.live_respond,
            }
            action = actions.get(parsed.path)
            if action is None:
                self._send_json(
                    {"status": "error", "error": "not found"},
                    HTTPStatus.NOT_FOUND,
                )
                return
            try:
                self._send_json(action(self._payload()))
            except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
                self._send_json(
                    {"status": "error", "error": str(exc)},
                    HTTPStatus.BAD_REQUEST,
                )

        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write(f"demo {self.address_string()} {format % args}\n")

    return DemoHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--prewarm", action="store_true")
    args = parser.parse_args()
    if args.prewarm and not args.live:
        parser.error("--prewarm requires --live")
    runtime = DemoRuntime(
        args.bundle,
        args.catalog,
        live_requested=args.live,
        prewarm=args.prewarm,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))
    print(f"ARC · Ask · Rank · Commit: http://{args.host}:{server.server_port}")
    print(
        f"Replay: {'VERIFIED' if runtime.replay_fresh else 'STALE'} | "
        f"Live: {'READY' if runtime._agent is not None else 'OFF'}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping demo server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
