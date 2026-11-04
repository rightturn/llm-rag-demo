"""A tiny in-memory vector store using cosine similarity."""
from __future__ import annotations

import numpy as np

from .chunking import Chunk


class VectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks, self._matrix = [], None

    def sources(self) -> list[str]:
        return sorted({c.source for c in self._chunks})

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return
        new = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(new, axis=1, keepdims=True)
        new = new / np.where(norms == 0, 1, norms)
        self._matrix = new if self._matrix is None else np.vstack([self._matrix, new])
        self._chunks.extend(chunks)

    def remove_source(self, source: str) -> None:
        """Drop every chunk of ``source`` (used when a document is re-ingested)."""
        keep = [i for i, c in enumerate(self._chunks) if c.source != source]
        if len(keep) == len(self._chunks):
            return
        self._chunks = [self._chunks[i] for i in keep]
        self._matrix = self._matrix[keep] if keep and self._matrix is not None else None

    def search(self, vector: list[float], k: int = 4) -> list[tuple[Chunk, float]]:
        if self._matrix is None or k <= 0:
            return []
        q = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm == 0:
            return []
        scores = self._matrix @ (q / norm)
        top = np.argsort(-scores)[:k]
        return [(self._chunks[i], float(scores[i])) for i in top]
