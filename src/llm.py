"""Optional OpenAI-compatible chat client built on the standard library.

ARC's scored path never needs this module: with ``config.LLM_MODE == "off"``
(the default) nothing here is imported at request time and every response
reports zero tokens. When a mode is enabled through the ``ARC_LLM_*``
environment variables, the client talks to any OpenAI-compatible endpoint
(a local vLLM/Ollama server or a hosted API) using ``urllib`` only, so the
runtime still carries no third-party dependency.

Every call is bounded by a timeout and wrapped so that a network failure
degrades to the deterministic behaviour instead of raising into ``respond``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import config

MODES = ("off", "ground", "assist")
_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


@dataclass
class LLMSettings:
    mode: str = "off"
    base_url: str = ""
    model: str = "gemma4"
    api_key: str = "unused"
    timeout: float = 20.0
    cache_path: str | None = None
    max_retries: int = 1

    @property
    def enabled(self) -> bool:
        return self.mode in ("ground", "assist") and bool(self.base_url)

    @property
    def says(self) -> bool:
        return self.mode == "assist" and bool(self.base_url)

    @classmethod
    def from_env(cls, environ: dict | None = None) -> "LLMSettings":
        env = os.environ if environ is None else environ
        mode = str(env.get("ARC_LLM_MODE", config.LLM_MODE) or "off").strip().lower()
        if mode not in MODES:
            mode = "off"
        timeout_raw = env.get("ARC_LLM_TIMEOUT", str(config.LLM_TIMEOUT_SECONDS))
        try:
            timeout = max(1.0, float(timeout_raw))
        except (TypeError, ValueError):
            timeout = float(config.LLM_TIMEOUT_SECONDS)
        return cls(
            mode=mode,
            base_url=str(env.get("ARC_LLM_BASE_URL", config.LLM_BASE_URL) or "").rstrip("/"),
            model=str(env.get("ARC_LLM_MODEL", config.LLM_MODEL) or "gemma4"),
            api_key=str(env.get("ARC_LLM_API_KEY", "unused") or "unused"),
            timeout=timeout,
            cache_path=env.get("ARC_LLM_CACHE") or None,
        )


@dataclass
class LLMReply:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def json(self) -> dict | None:
        """Best-effort extraction of one JSON object from a model reply."""
        if not self.text:
            return None
        candidate = self.text.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```[a-zA-Z]*\s*", "", candidate)
            candidate = re.sub(r"\s*```$", "", candidate)
        try:
            value = json.loads(candidate)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            pass
        match = _JSON_BLOCK.search(candidate)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None


@dataclass
class LLMUsage:
    """Per-``respond`` accounting so the contract's ``usage`` field is honest."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    latency_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    def absorb(self, reply: LLMReply) -> None:
        self.calls += 1
        self.prompt_tokens += max(0, int(reply.prompt_tokens))
        self.completion_tokens += max(0, int(reply.completion_tokens))
        self.latency_ms += max(0.0, float(reply.latency_ms))
        if reply.error:
            self.errors.append(reply.error)

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "calls": self.calls,
            "latency_ms": round(self.latency_ms, 1),
            "errors": list(self.errors),
        }


class ChatClient:
    """Minimal ``/v1/chat/completions`` client with an optional JSON cache.

    The cache makes research benchmarks reproducible: the same prompt at
    temperature zero is served from disk on the next run, and the recorded
    token counts are replayed so ``usage`` stays truthful.
    """

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self._cache: dict[str, dict] = {}
        self._cache_lock = threading.Lock()
        self._cache_dirty = False
        self._cache_path = Path(settings.cache_path) if settings.cache_path else None
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        if self._cache_path and self._cache_path.is_file():
            try:
                loaded = json.loads(self._cache_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._cache = loaded
            except (OSError, json.JSONDecodeError):
                self._cache = {}

    # -- cache ---------------------------------------------------------------
    def _key(self, messages: list[dict], max_tokens: int, temperature: float) -> str:
        payload = json.dumps(
            {
                "model": self.settings.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def flush(self) -> None:
        if not self._cache_path or not self._cache_dirty:
            return
        with self._cache_lock:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._cache_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self._cache, indent=0, sort_keys=True, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self._cache_path)
            self._cache_dirty = False

    # -- transport -----------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 200,
        temperature: float = 0.0,
        stream: bool = False,
    ) -> LLMReply:
        """Send one chat completion.

        ``stream`` matters only for a cold model: a shared server that has to
        load weights can take minutes before the first token, and proxies in
        front of such a server routinely drop a silent non-streaming request
        at 60 s. A streaming request keeps the connection alive through the
        load. The warm path stays non-streaming because it needs the whole
        JSON body anyway.
        """
        key = self._key(messages, max_tokens, temperature)
        with self._cache_lock:
            hit = self._cache.get(key)
        if hit is None and time.monotonic() < self._circuit_open_until:
            # The endpoint failed repeatedly; do not make the caller wait for
            # another timeout. The agent treats this exactly like any other
            # unavailable reply and stays on its deterministic path.
            return LLMReply(text="", error="circuit_open")
        if hit is not None:
            return LLMReply(
                text=str(hit.get("text", "")),
                prompt_tokens=int(hit.get("prompt_tokens", 0)),
                completion_tokens=int(hit.get("completion_tokens", 0)),
                latency_ms=0.0,
                cached=True,
            )
        body = json.dumps(
            {
                "model": self.settings.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": bool(stream),
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.settings.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.api_key}",
            },
            method="POST",
        )
        last_error = "no attempt"
        for attempt in range(self.settings.max_retries + 1):
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.settings.timeout) as handle:
                    if stream:
                        text, usage = _read_stream(handle)
                    else:
                        raw = json.loads(handle.read().decode("utf-8"))
                        choice = (raw.get("choices") or [{}])[0]
                        text = str(((choice.get("message") or {}).get("content")) or "")
                        usage = raw.get("usage") or {}
                elapsed = (time.perf_counter() - started) * 1000.0
                reply = LLMReply(
                    text=text,
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    latency_ms=elapsed,
                )
                with self._cache_lock:
                    self._cache[key] = {
                        "text": reply.text,
                        "prompt_tokens": reply.prompt_tokens,
                        "completion_tokens": reply.completion_tokens,
                    }
                    self._cache_dirty = True
                self._consecutive_failures = 0
                self._circuit_open_until = 0.0
                return reply
            except urllib.error.HTTPError as exc:
                last_error = f"http {exc.code}"
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.settings.max_retries:
                time.sleep(0.2)
        self._consecutive_failures += 1
        if self._consecutive_failures >= config.LLM_CIRCUIT_FAILURES:
            self._circuit_open_until = (
                time.monotonic() + config.LLM_CIRCUIT_COOLDOWN_SECONDS
            )
        return LLMReply(text="", error=last_error)


def _read_stream(handle) -> tuple[str, dict]:
    """Collect the text and usage of one server-sent-event completion."""
    pieces: list[str] = []
    usage: dict = {}
    for raw_line in handle:
        line = raw_line.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                pieces.append(str(delta["content"]))
    return "".join(pieces), usage


class FakeChatClient:
    """Deterministic stand-in for tests: maps a user message to a canned reply."""

    def __init__(self, replies: dict[str, str] | None = None, *, fail: bool = False) -> None:
        self.replies = dict(replies or {})
        self.fail = fail
        self.calls: list[list[dict]] = []
        self.settings = LLMSettings(mode="ground", base_url="fake://", model="fake")

    def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 200,
        temperature: float = 0.0,
        stream: bool = False,
    ) -> LLMReply:
        self.calls.append(messages)
        if self.fail:
            return LLMReply(text="", error="fake failure")
        user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        text = self.replies.get(user, self.replies.get("*", "{}"))
        return LLMReply(text=text, prompt_tokens=100, completion_tokens=20, latency_ms=1.0)

    def flush(self) -> None:  # pragma: no cover - interface parity
        return None
