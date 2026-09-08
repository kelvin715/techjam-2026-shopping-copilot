"""Dense (sentence-embedding) helpers for free-form wording (research only).

The scored path never imports this module, and nothing here runs on a
message the protocol parser recognised. It backs three research arms of the
language layer:

``dense`` (the original baseline)
    when the template parser learns nothing from a message, the message is
    appended to a free-text query and the cosine similarity between that
    query and each candidate product's text is added to the evidence score;

``DENSE_VALUE_EXPANSION`` (option 1: dense as a translator)
    an admitted feature phrase is expanded into the catalog signature values
    whose embedding lies within ``DENSE_VALUE_MIN_COSINE`` of it, and a phrase
    the catalog could not verify at all is mapped to its nearest supported
    signature value. The ranker credits an alternative at reduced weight, so
    a paraphrase of ``"water resistant"`` that the model rendered as
    ``"waterproof"`` still reaches the product that says the former;

``DENSE_SHELF_RECALL`` (option 2: dense as a recall channel)
    a category phrase that shares no token with any shelf name
    (``"women's denim"``) is mapped onto the shelves whose name embedding is
    close to it, and a session that has outgrown its pool also retrieves
    products by embedding similarity of the shopper's own sentences.

Requires ``sentence-transformers`` (a research dependency, not an agent
dependency) unless an ``embedder`` is injected, which the tests do with a
deterministic hashing embedder. Embeddings are cached under
``results/dense/`` per model and per vocabulary digest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import config


class SentenceTransformerEmbedder:
    """Thin wrapper so the index can be tested without the research dependency."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]):
        return self.model.encode(
            texts, batch_size=256, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ).astype("float32")


class HashEmbedder:
    """Deterministic bag-of-tokens embedding for tests: no model download.

    Two strings are close when they share tokens; ``synonyms`` lets a test
    declare that two different tokens mean the same thing.
    """

    model_name = "hash-test"

    def __init__(self, dim: int = 64, synonyms: dict[str, str] | None = None) -> None:
        self.dim = dim
        self.synonyms = synonyms or {}

    def encode(self, texts: list[str]):
        import numpy as np

        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for token in text.lower().replace("/", " ").replace(",", " ").replace(":", " ").split():
                token = self.synonyms.get(token, token)
                digest = hashlib.sha256(token.encode()).digest()
                out[row, int.from_bytes(digest[:4], "little") % self.dim] += 1.0
            norm = float(np.linalg.norm(out[row]))
            if norm:
                out[row] /= norm
        return out


class DenseIndex:
    def __init__(self, catalog, model_name: str | None = None, cache_dir: str | Path = "results/dense",
                 embedder=None) -> None:
        import numpy as np  # noqa: F401  (research dependency)

        self.model_name = model_name or config.DENSE_MODEL
        self.embedder = embedder or SentenceTransformerEmbedder(self.model_name)
        self.catalog = catalog
        self.ids = list(catalog.ids)
        self.position = {asin: index for index, asin in enumerate(self.ids)}
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.matrix = self._cached(
            "products", self.ids, lambda: [self._product_text(catalog, asin) for asin in self.ids],
            {"text": "title + features (first 400 chars)"},
        )
        self.path = str(self._path("products", self.ids))
        # Lazily built vocabularies for the value and shelf channels.
        self._signature_values: list[str] | None = None
        self._signature_matrix = None
        self._signature_position: dict[str, int] | None = None
        self._shelves: list[str] | None = None
        self._shelf_matrix = None
        self._query_cache: dict[str, object] = {}

    # ------------------------------------------------------------------ cache
    def _path(self, kind: str, keys: list[str]) -> Path:
        # The product matrix keeps the digest scheme of the original baseline
        # so the checked-in cache under results/dense/ is reused.
        salt = "" if kind == "products" else kind + "\n"
        digest = hashlib.sha256(
            (self.embedder.model_name + "\n" + salt + "\n".join(keys)).encode()
        ).hexdigest()[:16]
        return self.cache_dir / f"{kind}_{digest}.npy"

    def _cached(self, kind: str, keys: list[str], texts, meta: dict):
        import numpy as np

        path = self._path(kind, keys)
        if path.is_file():
            return np.load(path)
        matrix = self.embedder.encode(texts())
        if getattr(self.embedder, "model_name", "") != "hash-test":
            np.save(path, matrix)
            path.with_suffix(".json").write_text(json.dumps({
                "model": self.embedder.model_name, "kind": kind, "rows": len(keys), **meta,
            }))
        return matrix

    @staticmethod
    def _product_text(catalog, asin: str) -> str:
        return catalog.text[asin][:400]

    # ----------------------------------------------------------------- queries
    def query(self, text: str):
        """Embedding of one string, memoised: a phrase is asked about often."""
        vector = self._query_cache.get(text)
        if vector is None:
            vector = self.embedder.encode([text])[0]
            if len(self._query_cache) > 20000:
                self._query_cache.clear()
            self._query_cache[text] = vector
        return vector

    def rerank(self, scores: list[tuple[str, float]], query_vector, weight: float | None = None) -> list[tuple[str, float]]:
        """Add a min-max normalised cosine term to each candidate's score (the ``dense`` baseline)."""
        import numpy as np

        if not scores:
            return scores
        weight = config.DENSE_WEIGHT if weight is None else weight
        rows = np.array([self.position[asin] for asin, _ in scores])
        cosine = self.matrix[rows] @ query_vector
        low, high = float(cosine.min()), float(cosine.max())
        span = (high - low) or 1.0
        bumped = [(asin, score + weight * float((c - low) / span)) for (asin, score), c in zip(scores, cosine)]
        bumped.sort(key=lambda item: -item[1])
        return bumped

    # ------------------------------------------------------ option 1: values
    def _ensure_signatures(self) -> None:
        if self._signature_values is not None:
            return
        values = sorted(self.catalog.signature_value_count)
        self._signature_values = values
        self._signature_position = {value: index for index, value in enumerate(values)}
        self._signature_matrix = self._cached(
            "signatures", values, lambda: values, {"text": "intent-signature values"},
        )

    def similar_values(self, phrase: str, allowed=None, limit: int = 3,
                       min_cosine: float | None = None) -> list[tuple[str, float]]:
        """Signature values whose embedding is close to ``phrase``, best first.

        ``allowed`` restricts the search to the values a session can still
        use (those with support among its candidate products); the phrase
        itself is never returned as its own alternative.
        """
        import numpy as np

        self._ensure_signatures()
        threshold = config.DENSE_VALUE_MIN_COSINE if min_cosine is None else min_cosine
        if allowed is None:
            rows = None
            matrix = self._signature_matrix
            names = self._signature_values
        else:
            rows = [self._signature_position[value] for value in allowed if value in self._signature_position]
            if not rows:
                return []
            rows = np.array(rows)
            matrix = self._signature_matrix[rows]
            names = [self._signature_values[index] for index in rows]
        cosine = matrix @ self.query(phrase)
        order = np.argsort(-cosine)
        out: list[tuple[str, float]] = []
        for index in order[: max(limit + 1, 8)]:
            score = float(cosine[index])
            if score < threshold:
                break
            value = names[index]
            if value == phrase:
                continue
            out.append((value, round(score, 4)))
            if len(out) >= limit:
                break
        return out

    # ----------------------------------------------------- option 2: recall
    def _ensure_shelves(self) -> None:
        if self._shelves is not None:
            return
        shelves = list(self.catalog.by_shelf)
        self._shelves = shelves
        self._shelf_matrix = self._cached("shelves", shelves, lambda: shelves, {"text": "shelf names"})

    def similar_shelves(self, text: str, limit: int | None = None,
                        min_cosine: float | None = None) -> list[tuple[str, float]]:
        """Shelves whose name embedding is close to ``text``, best first."""
        import numpy as np

        self._ensure_shelves()
        limit = config.DENSE_SHELF_LIMIT if limit is None else limit
        threshold = config.DENSE_SHELF_MIN_COSINE if min_cosine is None else min_cosine
        cosine = self._shelf_matrix @ self.query(text)
        order = np.argsort(-cosine)[:limit]
        return [
            (self._shelves[index], round(float(cosine[index]), 4))
            for index in order if float(cosine[index]) >= threshold
        ]

    def search_products(self, text: str, limit: int) -> list[str]:
        """Products whose text embedding is closest to ``text``, best first."""
        import numpy as np

        if limit <= 0:
            return []
        cosine = self.matrix @ self.query(text)
        if limit < len(self.ids):
            top = np.argpartition(-cosine, limit - 1)[:limit]
            top = top[np.argsort(-cosine[top])]
        else:
            top = np.argsort(-cosine)
        return [self.ids[index] for index in top]
