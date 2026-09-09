"""Opt-in, request-local observation of decisions; never part of the response.

The evaluator installs no observer. The demo supplies a callback for one call
using a ContextVar, so concurrent threads cannot observe each other's turns.
An unavailable observer must never change an agent decision.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

_observer = ContextVar("arc_decision_observer", default=None)


@contextmanager
def observe(callback):
    token = _observer.set(callback)
    try:
        yield
    finally:
        _observer.reset(token)


def emit(stage: str, status: str, **data) -> None:
    callback = _observer.get()
    if callback is not None:
        try:
            callback({"stage": stage, "status": status, "data": data})
        except Exception:
            # Logging / disconnected presentation clients cannot alter policy.
            pass
