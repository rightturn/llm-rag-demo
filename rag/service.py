"""Ingestion and question answering on top of the store, embedder and generator."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .chunking import Chunk, chunk_text
from .embeddings import Embedder
from .llm import NO_ANSWER, Generator
from .store import VectorStore


@dataclass
class Citation:
    source: str
    chunk_id: str
    score: float
    text: str


@dataclass
class Answer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    answered: bool = True


class RagService:
    def __init__(
        self,
        embedder: Embedder,
        generator: Generator,
        store: VectorStore | None = None,
        top_k: int = 4,
        min_score: float = 0.15,
    ) -> None:
        self.embedder = embedder
        self.generator = generator
        self.store = store or VectorStore()
        self.top_k = top_k
        self.min_score = min_score

    def ingest_text(self, source: str, text: str) -> int:
        chunks = chunk_text(source, text)
        self.store.remove_source(source)  # re-ingesting replaces the old version
        if chunks:
            self.store.add(chunks, self.embedder.embed([c.text for c in chunks]))
        return len(chunks)

    def ingest_directory(self, directory: str | Path) -> dict[str, int]:
        counts: dict[str, int] = {}
        for path in sorted(Path(directory).glob("*.md")):
            counts[path.name] = self.ingest_text(path.name, path.read_text(encoding="utf-8"))
        return counts

    def retrieve(self, question: str) -> list[tuple[Chunk, float]]:
        vector = self.embedder.embed([question])[0]
        hits = self.store.search(vector, self.top_k)
        return [(chunk, score) for chunk, score in hits if score >= self.min_score]

    def ask(self, question: str) -> Answer:
        hits = self.retrieve(question)
        if not hits:
            return Answer(NO_ANSWER, [], answered=False)
        text = self.generator.answer(question, [c for c, _ in hits])
        citations = [Citation(c.source, c.id, round(s, 4), c.text) for c, s in hits]
        return Answer(text, citations, answered=text != NO_ANSWER)
