"""Catalog loading, shelf indexing, and text normalisation.

Reimplements the category-coarsening and intent-card extraction rules as our
own components so the submission stays self-contained and never imports the
evaluator at runtime.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

_WS = re.compile(r"\s+")
_TOK = re.compile(r"[a-z0-9]+")
_PUNCT = re.compile(r"[^a-z0-9]+")

MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool",
    "spandex", "silk", "rayon", "fabric",
)
COLORS = (
    "black", "white", "blue", "red", "pink", "green", "brown",
    "gray", "grey", "purple", "yellow", "orange",
)
MATERIAL_RE = re.compile(r"\b(" + "|".join(MATERIALS) + r")\b", re.I)
COLOR_RE = re.compile(r"\b(" + "|".join(COLORS) + r")\b", re.I)

SEARCH_FIELDS = (
    "title", "features", "details", "description", "categories", "store",
)
_EXCLUDED = {
    "clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry",
}

# The review-count term is deliberately sublinear twice: log compression
# handles the heavy tail and the exponent prevents a blockbuster from
# overwhelming a substantially better evidence match.
_POPULARITY_COUNT_EXPONENT = 1.25
_POPULARITY_QUALITY_MIX = 0.50


def norm(text: str) -> str:
    return _WS.sub(" ", str(text)).strip().lower()


def loose(text: str) -> str:
    return _PUNCT.sub(" ", text.lower()).strip()


def tokens(text: str) -> list[str]:
    return _TOK.findall(str(text).lower())


def flatten(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def searchable_text(product: dict) -> str:
    return norm(" ".join(flatten(product.get(field)) for field in SEARCH_FIELDS))


def _constraint_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [
            f"{key}: {item}"
            for key, item in value.items()
            if item not in (None, "", [])
        ]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _clean_constraint(value: str, limit: int = 180) -> str:
    return _WS.sub(" ", value).strip(" -;,.\t\n")[:limit].rstrip()


def intent_signature(product: dict, corpus: str) -> tuple[str, ...]:
    """Materialise a product's canonical, ordered dialog-constraint signature.

    Keeping field order preserves evidence about which product most plausibly
    generated a sequence of disclosed constraints. Normalised duplicates are
    collapsed because session memory performs the same deduplication.
    """
    candidates = [
        *_constraint_values(product.get("features")),
        *_constraint_values(product.get("details")),
    ]
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")

    cleaned = list(dict.fromkeys(
        value
        for item in candidates
        if (value := _clean_constraint(item))
    ))
    if not cleaned:
        cleaned = [_clean_constraint(str(product.get("title") or "product"))]

    # Apply the four-slot protocol limit before normalised deduplication. This
    # mirrors what can actually reach SessionState when differently cased
    # duplicates occupy two source slots.
    return tuple(dict.fromkeys(norm(value) for value in cleaned[:4]))


def coarse_category(values: list) -> str:
    cleaned: list[str] = []
    for value in values or []:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in _EXCLUDED:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


class Catalog:
    """Load the frozen catalog once and build read-only in-memory indexes."""

    def __init__(self, path: str | Path) -> None:
        self.ids: list[str] = []
        self._df: dict[str, int] = {}
        self.text: dict[str, str] = {}
        self.ltext: dict[str, str] = {}
        self.title: dict[str, str] = {}
        self.signature: dict[str, tuple[str, ...]] = {}
        self.signature_value_count: dict[str, int] = {}
        self.signature_value_count_by_shelf: dict[str, dict[str, int]] = {}
        self.first_material: dict[str, str | None] = {}
        self.first_color: dict[str, str | None] = {}
        self.price: dict[str, float | None] = {}
        self.rating: dict[str, float] = {}
        self.rating_count: dict[str, int] = {}
        self.shelf_of: dict[str, str] = {}
        self.by_shelf: dict[str, list[str]] = {}

        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                asin = str(product["parent_asin"])
                shelf = coarse_category(product.get("categories") or [])
                text = searchable_text(product)
                self.ids.append(asin)
                self.text[asin] = text
                self.ltext[asin] = loose(text)
                self.title[asin] = norm(flatten(product.get("title")))
                self.signature[asin] = intent_signature(product, text)
                shelf_signature_counts = self.signature_value_count_by_shelf.setdefault(
                    shelf, {}
                )
                for value in self.signature[asin]:
                    self.signature_value_count[value] = (
                        self.signature_value_count.get(value, 0) + 1
                    )
                    shelf_signature_counts[value] = (
                        shelf_signature_counts.get(value, 0) + 1
                    )

                material = MATERIAL_RE.search(text)
                color = COLOR_RE.search(text)
                self.first_material[asin] = (
                    material.group(1).lower() if material else None
                )
                first_color = color.group(1).lower() if color else None
                self.first_color[asin] = (
                    "gray" if first_color == "grey" else first_color
                )
                try:
                    raw_price = product.get("price")
                    self.price[asin] = (
                        float(raw_price) if raw_price not in (None, "") else None
                    )
                except (TypeError, ValueError):
                    self.price[asin] = None
                try:
                    self.rating[asin] = float(product.get("average_rating") or 0.0)
                except (TypeError, ValueError):
                    self.rating[asin] = 0.0
                try:
                    self.rating_count[asin] = max(
                        0, int(product.get("rating_number") or 0)
                    )
                except (TypeError, ValueError):
                    self.rating_count[asin] = 0

                self.shelf_of[asin] = shelf
                self.by_shelf.setdefault(shelf, []).append(asin)
                for token in set(tokens(text)):
                    self._df[token] = self._df.get(token, 0) + 1

        self.avg_len = sum(len(text) for text in self.text.values()) / max(
            1, len(self.text)
        )
        max_ratings = max(self.rating_count.values(), default=1)
        self._log_max_ratings = math.log1p(max_ratings) or 1.0
        self._shelves_sorted = sorted(self.by_shelf, key=len, reverse=True)
        self._shelf_lower = [
            (shelf.lower(), shelf) for shelf in self._shelves_sorted
        ]

    def token_idf(self, token: str) -> float:
        return math.log(1.0 + len(self.ids) / (1.0 + self._df.get(token, 0)))

    def popularity(self, asin: str) -> float:
        count = (
            math.log1p(self.rating_count.get(asin, 0)) / self._log_max_ratings
        ) ** _POPULARITY_COUNT_EXPONENT
        quality = max(0.0, min(1.0, self.rating.get(asin, 0.0) / 5.0))
        return (
            (1.0 - _POPULARITY_QUALITY_MIX) * count
            + _POPULARITY_QUALITY_MIX * quality
        )

    def match_shelf(self, message: str) -> str | None:
        message_norm = norm(message)
        for lowered, original in self._shelf_lower:
            if lowered and lowered in message_norm:
                return original
        return None

    def candidates(self, shelf: str | None) -> list[str]:
        if shelf and shelf in self.by_shelf:
            return self.by_shelf[shelf]
        return self.ids

    def signature_support(self, value: str, shelf: str | None = None) -> int:
        """Number of products whose protocol-visible signature contains value."""
        key = norm(value)
        if shelf and shelf in self.signature_value_count_by_shelf:
            return self.signature_value_count_by_shelf[shelf].get(key, 0)
        return self.signature_value_count.get(key, 0)
