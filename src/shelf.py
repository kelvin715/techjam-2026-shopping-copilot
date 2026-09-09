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

# Function words that carry no department meaning when a free-form sentence
# is matched against shelf names by token overlap (``Catalog.shelf_pool``).
# Only the free-form path uses this; protocol messages name a shelf exactly.
_SHELF_STOPWORDS = frozenset(
    "the and for with from all our your you her him his she its are was not "
    "but this that these those have has had who what when where how any some "
    "one ones per via into onto than then too very just only also can may "
    "will get got like want need looking buy something anything more most "
    "new top best off out down over under around about after before between "
    "within".split()
)

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


def category_parts(values: list) -> list[str]:
    """The category path with the catalog root removed, one part per node."""
    cleaned: list[str] = []
    for value in values or []:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in _EXCLUDED:
                cleaned.append(part)
    return cleaned


def coarse_category(values: list) -> str:
    cleaned = category_parts(values)
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
        # The (parent, leaf) nodes a shelf name was built from; a shelf whose
        # only parent is the catalog root's residual ("Shoes & Jewelry
        # Westlake") is a promotional or brand node, not a department.
        self.shelf_parts: dict[str, list[str]] = {}

        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                asin = str(product["parent_asin"])
                parts = category_parts(product.get("categories") or [])
                shelf = " ".join(parts[-2:]) if parts else "clothing item"
                if shelf not in self.shelf_parts:
                    self.shelf_parts[shelf] = parts[-2:]
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
        # Lazily built helpers for grounding free-form wording. They are
        # derived from the same read-only indexes and never touch labels.
        self._shelf_stems: dict[str, set[str]] | None = None
        self._shelf_stems_full: dict[str, set[str]] = {}
        self._root_residual_stems: frozenset[str] = frozenset()
        self._signature_token_index: dict[str, set[str]] | None = None

    def token_set(self, asin: str) -> set[str]:
        """Unique tokens of one product's text.

        Not cached: fifty thousand token sets cost more resident memory than
        the whole catalog index, and the ranker tests membership with a padded
        substring search on ``ltext`` instead.
        """
        return set(self.ltext[asin].split())

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

    @staticmethod
    def _stem(token: str) -> str:
        if len(token) >= 4 and token.endswith("es") and not token.endswith("ses"):
            return token[:-2]
        if len(token) >= 4 and token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
        return token

    def _stems(self, text: str) -> list[str]:
        """Content stems of a shelf name or a shopper sentence.

        Function words are dropped on both sides: a shelf called "Ugly
        Holiday Sweaters (and More) for the Family" must not win a shopper's
        "running shoes for the gym" on "for", "the" and "and".
        """
        return [
            self._stem(token) for token in tokens(text)
            if len(token) > 2 and token not in _SHELF_STOPWORDS
        ]

    def shelf_pool(self, category: str, min_products: int = 100) -> tuple[list[str], list[str]]:
        """Union of shelves sharing the most content tokens with ``category``.

        Free-form wording rarely names a coarse category exactly. Rather than
        gamble on one shelf, keep every shelf that ties for the best token
        overlap, so a wrong guess widens the pool instead of losing the target.
        """
        if self._shelf_stems is None:
            self._shelf_stems = self._department_stems()
        wanted = set(self._stems(category))
        if not wanted:
            return [], []
        shelf_stems = self._shelf_stems
        if self._root_residual_stems and self._root_residual_stems <= wanted:
            # A shopper who says the root's own words ("men's shoes and
            # jewelry") is naming the nodes filed under it; count them.
            shelf_stems = self._shelf_stems_full
        best = 0
        matched: list[str] = []
        for shelf, stems in shelf_stems.items():
            overlap = len(wanted & stems)
            if overlap > best:
                best, matched = overlap, [shelf]
            elif overlap and overlap == best:
                matched.append(shelf)
        if not matched:
            return [], []
        # A shopper's wording can over-specify the shelf ("tops and tunics"
        # matching one two-product shelf). When the best-overlap union is
        # tiny, widen to the next overlap level so a near miss still keeps
        # the target reachable.
        if best > 1 and min_products > 0:
            size = sum(len(self.by_shelf[shelf]) for shelf in matched)
            if size < min_products:
                matched.extend(
                    shelf for shelf, stems in self._shelf_stems.items()
                    if len(wanted & stems) == best - 1
                )
        matched.sort(key=lambda shelf: (-len(wanted & shelf_stems[shelf]), -len(self.by_shelf[shelf]), shelf))
        pool: list[str] = []
        seen: set[str] = set()
        for shelf in matched:
            for asin in self.by_shelf[shelf]:
                if asin not in seen:
                    seen.add(asin)
                    pool.append(asin)
        return matched, pool

    def _department_stems(self) -> dict[str, set[str]]:
        """Content stems of every shelf name, minus the catalog root's residual.

        ``coarse_category`` keeps the last two nodes of a product's category
        path. Products filed directly under the root ("Clothing, Shoes &
        Jewelry" > "Westlake") therefore get the root's residual as their
        parent, and every such promotional, brand or event node would answer
        a shopper's "shoes" or "jewelry". The residual is found as the parent
        shared by the most shelves (at least a tenth of them); its words do
        not count for the shelves it heads.
        """
        parents: dict[str, int] = {}
        for shelf, parts in self.shelf_parts.items():
            if len(parts) == 2:
                parents[parts[0]] = parents.get(parts[0], 0) + 1
        root_residual: str | None = None
        if parents:
            top = max(parents, key=lambda part: (parents[part], part))
            if parents[top] >= max(3, 0.1 * len(self.by_shelf)):
                root_residual = top
        stems: dict[str, set[str]] = {}
        self._shelf_stems_full = {
            shelf: set(self._stems(shelf)) for shelf in self.by_shelf
        }
        self._root_residual_stems = (
            frozenset(self._stems(root_residual)) if root_residual else frozenset()
        )
        for shelf in self.by_shelf:
            parts = self.shelf_parts.get(shelf) or [shelf]
            if root_residual is not None and len(parts) == 2 and parts[0] == root_residual:
                stems[shelf] = set(self._stems(parts[1]))
            else:
                stems[shelf] = self._shelf_stems_full[shelf]
        return stems

    def department_placement(self, text: str) -> tuple[int, int]:
        """How firmly a sentence places itself: (best stem overlap, pool size).

        ``(0, 0)`` when no shelf shares a content word with the sentence. A
        single shared word over a large union of shelves ("shoes" alone) is
        a weak placement; two or more shared words ("fashion sneakers") name
        a department.
        """
        shelves, pool = self.shelf_pool(text)
        if not shelves:
            return 0, 0
        if self._shelf_stems is None:
            self._shelf_stems = self._department_stems()
        wanted = set(self._stems(text))
        shelf_stems = self._shelf_stems
        if self._root_residual_stems and self._root_residual_stems <= wanted:
            shelf_stems = self._shelf_stems_full
        best = max(len(wanted & shelf_stems[shelf]) for shelf in shelves)
        return best, len(pool)

    def retrieval_index(self, max_df: int) -> dict[str, list[str]]:
        """Inverted index over tokens rarer than ``max_df``, built once.

        Used only when a grounded session has to look beyond its shelf
        pool. Common tokens are left out because they do not discriminate
        and their posting lists are the expensive ones to walk.
        """
        cached = getattr(self, "_retrieval_index", None)
        if cached is None or cached[0] != max_df:
            index: dict[str, list[str]] = {}
            df = self._df
            for asin in self.ids:
                for token in self.token_set(asin):
                    if df.get(token, 0) <= max_df:
                        index.setdefault(token, []).append(asin)
            cached = (max_df, index)
            self._retrieval_index = cached
        return cached[1]

    def retrieve(
        self,
        constraints: list[str],
        constraint_weights: list[float] | None,
        limit: int,
        max_df: int,
    ) -> list[str]:
        """Products sharing the rarest words with the constraints, best first.

        A rarity-weighted union of posting lists: each product accumulates
        the IDF of every discriminative constraint token it contains, scaled
        by that constraint's confidence. Ties keep catalog order, so the
        result is deterministic. Returns an empty list when no constraint
        carries a discriminative token, which tells the caller to fall back
        to the whole catalog.
        """
        if not constraints or limit <= 0:
            return []
        if constraint_weights is None or len(constraint_weights) != len(constraints):
            constraint_weights = [1.0] * len(constraints)
        index = self.retrieval_index(max_df)
        accumulated: dict[str, float] = {}
        for phrase, confidence in zip(constraints, constraint_weights):
            value = norm(phrase)
            if value.startswith("budget around"):
                continue
            wanted = [token for token in dict.fromkeys(tokens(value)) if len(token) > 2]
            if value.startswith("color: gray") or value == "gray":
                wanted.append("grey")
            for token in wanted:
                postings = index.get(token)
                if not postings:
                    continue
                gain = confidence * self.token_idf(token)
                for asin in postings:
                    accumulated[asin] = accumulated.get(asin, 0.0) + gain
        if not accumulated:
            return []
        order = {asin: position for position, asin in enumerate(self.ids)} if len(accumulated) > limit else None
        ranked = sorted(
            accumulated.items(),
            key=(lambda item: (-item[1], order[item[0]])) if order else (lambda item: -item[1]),
        )
        return [asin for asin, _ in ranked[:limit]]

    def signature_values_loose(self) -> dict[str, str]:
        """Punctuation-stripped form of every signature value, computed once."""
        cached = getattr(self, "_signature_loose", None)
        if cached is None:
            cached = {value: loose(value) for value in self.signature_value_count}
            self._signature_loose = cached
        return cached

    def shelf_words(self) -> frozenset[str]:
        """Content tokens and stems of every shelf name."""
        cached = getattr(self, "_shelf_words", None)
        if cached is None:
            words: set[str] = set()
            for shelf in self.by_shelf:
                for token in tokens(shelf):
                    if len(token) > 2:
                        words.add(token)
                        words.add(self._stem(token))
            cached = frozenset(words)
            self._shelf_words = cached
        return cached

    def signature_vocabulary(self) -> frozenset[str]:
        """Every content stem that occurs in some intent-signature value.

        The signature strings are the catalog's own attribute language, so
        this set is "words that can describe a product attribute" without
        any hand-written vocabulary.
        """
        cached = getattr(self, "_signature_vocabulary", None)
        if cached is None:
            words: set[str] = set()
            for value in self.signature_value_count:
                if value.startswith("budget around"):
                    continue
                for token in tokens(value):
                    if len(token) > 2 and not token.isdigit():
                        words.add(token)
                        words.add(self._stem(token))
            cached = frozenset(words)
            self._signature_vocabulary = cached
        return cached

    def signature_values_by_first_token(self) -> dict[str, list[str]]:
        """Signature values grouped by the first word of their loose form.

        Lets a verbatim-string matcher consider only values that can occur
        in a sentence instead of every string in the catalog.
        """
        cached = getattr(self, "_signature_by_first_token", None)
        if cached is None:
            cached = {}
            for value, loose_value in self.signature_values_loose().items():
                first = loose_value.split(" ", 1)[0] if loose_value else ""
                if first:
                    cached.setdefault(first, []).append(value)
            self._signature_by_first_token = cached
        return cached

    def top_signature_values(
        self, counts: dict[str, int], limit: int = 40, max_tokens: int = 5
    ) -> list[str]:
        """Most common short signature values in a pool, generic ones removed."""
        generic = {
            "imported", "is discontinued by manufacturer: no",
            "is discontinued by manufacturer: yes",
        }
        total = max(1, sum(counts.values()))
        ranked = []
        for value, count in counts.items():
            value_tokens = tokens(value)
            if (
                value in generic
                or value.startswith("budget around")
                or value.startswith("color:")
                or value in MATERIALS
                or not value_tokens
                or len(value_tokens) > max_tokens
                or all(token.isdigit() for token in value_tokens)
            ):
                continue
            # Single words are rarely discriminative; keep one only when it
            # is genuinely common in this pool.
            if len(value_tokens) == 1 and count < 0.02 * total:
                continue
            ranked.append((value, count))
        ranked.sort(key=lambda item: (-item[1], len(item[0]), item[0]))
        return [value for value, _ in ranked[:limit]]

    def signature_values_with_tokens(
        self, stems: list[str], shelf: str | None = None
    ) -> list[tuple[str, int]]:
        """Canonical signature values containing every given content stem.

        Returns ``(value, support)`` pairs, best first: highest support within
        the shelf (or globally), then the shortest value. Used to map a
        shopper's paraphrase such as ``"buckle"`` onto the catalog's own
        ``"buckle closure"`` before it is allowed to become evidence.
        """
        if not stems:
            return []
        if self._signature_token_index is None:
            index: dict[str, set[str]] = {}
            for value in self.signature_value_count:
                for stem in set(self._stems(value)):
                    index.setdefault(stem, set()).add(value)
            self._signature_token_index = index
        candidates: set[str] | None = None
        for stem in stems:
            values = self._signature_token_index.get(stem)
            if not values:
                return []
            candidates = set(values) if candidates is None else candidates & values
            if not candidates:
                return []
        assert candidates is not None
        scored = []
        for value in candidates:
            support = self.signature_support(value, shelf)
            if support > 0:
                scored.append((value, support))
        scored.sort(key=lambda item: (-item[1], len(item[0]), item[0]))
        return scored

    def signature_support(self, value: str, shelf: str | None = None) -> int:
        """Number of products whose protocol-visible signature contains value."""
        key = norm(value)
        if shelf and shelf in self.signature_value_count_by_shelf:
            return self.signature_value_count_by_shelf[shelf].get(key, 0)
        return self.signature_value_count.get(key, 0)
