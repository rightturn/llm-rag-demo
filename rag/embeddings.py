"""Embedding providers.

* ``HashingEmbedder``: offline and deterministic (keyword-style, no API key). Used for tests,
  CI and the free demo mode.
* ``OpenAIEmbedder``: real semantic embeddings via the OpenAI API.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i in is it its of on or "
    "that the their there this to was what when where which who why will with you your".split()
)
TOKEN_RE = re.compile(r"[a-z0-9]+")


SUFFIXES = ("ations", "ation", "ions", "ion", "ing", "ed", "es", "s")


def stem(token: str) -> str:
    """Very light suffix stripping so "encryption" and "encrypted" match."""
    for suffix in SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Bag-of-words embeddings using the hashing trick, L2-normalised."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _index(self, token: str) -> int:
        digest = hashlib.md5(token.encode(), usedforsecurity=False).digest()
        return int.from_bytes(digest[:4], "big") % self.dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            counts: dict[str, int] = {}
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                vec[self._index(token)] += 1.0 + math.log(count)  # sub-linear term frequency
            norm = math.sqrt(sum(v * v for v in vec))
            out.append([v / norm for v in vec] if norm else vec)
        return out


class OpenAIEmbedder:
    def __init__(self, client, model: str = "text-embedding-3-small") -> None:
        self.client = client
        self.model = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=list(texts))
        return [item.embedding for item in response.data]
