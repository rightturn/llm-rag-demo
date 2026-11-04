"""FastAPI application."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .embeddings import HashingEmbedder, OpenAIEmbedder
from .llm import ExtractiveGenerator, OpenAIGenerator
from .service import RagService

DOCS_DIR = Path(os.environ.get("DOCS_DIR", Path(__file__).resolve().parent.parent / "docs"))


def build_service() -> RagService:
    """Use OpenAI when OPENAI_API_KEY is set, otherwise the free offline mode."""
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI  # imported lazily: only needed for real mode

        client = OpenAI()
        return RagService(
            OpenAIEmbedder(client, os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")),
            OpenAIGenerator(client, os.environ.get("CHAT_MODEL", "gpt-4o-mini")),
        )
    return RagService(HashingEmbedder(), ExtractiveGenerator())


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class IngestRequest(BaseModel):
    source: str = Field(min_length=1, max_length=200, pattern=r"^[\w.\- ]+$")
    text: str = Field(min_length=1, max_length=200_000)


def create_app(service: RagService | None = None, load_samples: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or build_service()
        if load_samples and DOCS_DIR.is_dir():
            app.state.service.ingest_directory(DOCS_DIR)
        yield

    app = FastAPI(title="LLM RAG demo", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        svc: RagService = app.state.service
        mode = "openai" if type(svc.generator).__name__ == "OpenAIGenerator" else "offline"
        return {"status": "ok", "mode": mode, "chunks": len(svc.store), "sources": svc.store.sources()}

    @app.post("/ingest", status_code=201)
    def ingest(body: IngestRequest):
        added = app.state.service.ingest_text(body.source, body.text)
        if added == 0:
            raise HTTPException(422, "No text to index")
        return {"source": body.source, "chunks": added}

    @app.post("/ask")
    def ask(body: AskRequest):
        result = app.state.service.ask(body.question)
        return {
            "answer": result.answer,
            "answered": result.answered,
            "citations": [c.__dict__ for c in result.citations],
        }

    return app


app = create_app()
