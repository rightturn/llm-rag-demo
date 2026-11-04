"""Split documents into overlapping chunks that respect paragraph boundaries."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    source: str
    index: int
    text: str

    @property
    def id(self) -> str:
        return f"{self.source}#{self.index}"


def chunk_text(source: str, text: str, max_chars: int = 600, overlap: int = 100) -> list[Chunk]:
    """Greedily pack paragraphs into chunks of at most ``max_chars``.

    Paragraphs longer than ``max_chars`` are split on whitespace. Consecutive chunks
    share roughly ``overlap`` characters so an answer that straddles a boundary is
    still retrievable.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be >= 0 and < max_chars")

    pieces: list[str] = []
    for paragraph in (p.strip() for p in text.split("\n\n")):
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
            continue
        words, current = paragraph.split(), ""
        for word in words:
            if current and len(current) + 1 + len(word) > max_chars:
                pieces.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            pieces.append(current)

    chunks: list[Chunk] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 2 + len(piece) > max_chars:
            chunks.append(Chunk(source, len(chunks), current))
            tail = current[-overlap:] if overlap else ""
            # start the next chunk on a word boundary
            tail = tail.split(" ", 1)[1] if " " in tail and overlap else tail
            current = f"{tail}\n\n{piece}".strip() if tail else piece
        else:
            current = f"{current}\n\n{piece}".strip() if current else piece
    if current:
        chunks.append(Chunk(source, len(chunks), current))
    return chunks
