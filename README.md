# llm-rag-demo

A small, readable **retrieval-augmented generation (RAG)** service in Python: ingest documents,
retrieve the most relevant passages, and answer questions **only from those passages, with citations**.

![CI](https://github.com/rightturn/llm-rag-demo/actions/workflows/ci.yml/badge.svg)

It runs in two modes:

| Mode | When | Embeddings | Answers |
|---|---|---|---|
| **Offline** (default) | no API key | hashing-trick bag-of-words (keyword-style) | extractive: the best matching sentence |
| **OpenAI** | `OPENAI_API_KEY` set | `text-embedding-3-small` | `gpt-4o-mini`, grounded prompt, `temperature=0` |

The offline mode makes the project runnable, testable and CI-friendly for free.
It is keyword-based, so it will not understand paraphrases the way real embeddings do.

## What it demonstrates
- **Chunking** with paragraph-aware splitting and overlap (`rag/chunking.py`)
- **Vector search** with cosine similarity over a NumPy matrix (`rag/store.py`)
- **Pluggable providers** behind two small interfaces: `Embedder` and `Generator`
- **Grounded answers with citations**; unsupported questions get *"I don't know"* instead of a guess,
  using a minimum similarity threshold
- **Prompt-injection awareness:** retrieved text is passed as numbered *data*, and the system prompt
  tells the model to ignore instructions found inside it
- **Re-ingestion** replaces the old version of a document
- **An evaluation script** that measures retrieval hit rate and answer correctness
- **API validation** (length limits, safe document names) and a Docker image with a health check

## Run it
```bash
pip install -r requirements.txt
uvicorn rag.api:app --port 8000
```
Three sample documents (a fictional backup product) are indexed at startup.

```bash
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question":"How long are deleted files recoverable on the Team plan?"}'
```
```json
{
  "answer": "Deleted files stay recoverable for 30 days on the free plan and for 180 days on the Team plan. [1]",
  "answered": true,
  "citations": [{"source": "restore-and-retention.md", "chunk_id": "restore-and-retention.md#0", "score": 0.4889, "text": "..."}]
}
```
A question the documents cannot answer returns `"answered": false` and no citations.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, mode (`offline` / `openai`), number of chunks and sources |
| POST | `/ingest` | `{"source": "name.md", "text": "..."}` adds or replaces a document |
| POST | `/ask` | `{"question": "..."}` returns the answer and its citations |

### With OpenAI
```bash
export OPENAI_API_KEY=sk-...        # your own key; never commit it
uvicorn rag.api:app --port 8000
```
Set `EMBEDDING_MODEL` and `CHAT_MODEL` to change models.

### With Docker
```bash
docker build -t llm-rag-demo .
docker run -p 8000:8000 llm-rag-demo                       # offline mode
docker run -p 8000:8000 -e OPENAI_API_KEY llm-rag-demo     # OpenAI mode
```

## Tests and evaluation
```bash
pip install -r requirements-dev.txt
python -m pytest -q          # 15 tests
python -m eval.run_eval      # retrieval hit rate and answer match rate
```
The evaluation set is small (9 questions over 3 documents that I wrote), so treat the scores
(offline mode: 100 % retrieval, 100 % answer match) as a regression check, not a benchmark.
The first version of the offline mode scored 89 % / 78 %; the misses showed that it needed light
stemming ("encryption" vs "encrypted") and rarity weighting of words, which are now in place.

The OpenAI adapters are unit-tested against stubbed clients (prompt contents, parameters, response
mapping). They have not been exercised against the live API in this repository's CI.

## Layout
```
rag/chunking.py     paragraph-aware chunking with overlap
rag/embeddings.py   HashingEmbedder (offline) and OpenAIEmbedder
rag/store.py        in-memory cosine-similarity store
rag/llm.py          ExtractiveGenerator (offline) and OpenAIGenerator, prompt
rag/service.py      ingest / retrieve / ask
rag/api.py          FastAPI app
docs/               sample documents
eval/               question set and evaluation script
tests/              pytest suite
```

## Limits and next steps
The store is in memory (everything is re-indexed at start-up). A production version would use a
persistent vector database, hybrid keyword+vector search, a re-ranker, streaming answers and
authentication on `/ingest`.

## License
MIT
