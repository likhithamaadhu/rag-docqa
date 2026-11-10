# RAG Doc Q&A

Upload PDFs, ask questions, get answers with page-level citations.
FastAPI · Postgres + pgvector · local embeddings · any OpenAI-compatible LLM.

## Run it (v0.1)

```bash
docker compose up -d                 # Postgres (pgvector) + Redis
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # add your LLM_API_KEY (Groq free tier works)
uvicorn app.main:app --reload
pytest -q                            # chunking tests
```

Open http://localhost:8000/docs and try:

1. `POST /documents` with a PDF → returns `id`, status `pending` (HTTP 202)
2. `GET /documents/{id}` until status is `ready`
3. `POST /ask` with `{"question": "..."}` → answer + citations

The first upload is slow (~30 s) because the embedding model downloads once.

## Architecture (v0.1)

```
client ──► FastAPI ──► Postgres (documents, chunks + HNSW vector index)
              │  ▲
              │  └── retrieve top-k chunks (cosine similarity)
              ├──► BackgroundTasks: PDF → pages → chunks → embeddings
              └──► LLM API (answer grounded in retrieved chunks)
```

Key decisions and why:

| Decision | Why | Tradeoff / revisit later |
|---|---|---|
| pgvector instead of a vector DB | One database for metadata and vectors; joins and transactions for free | Dedicated vector DBs scale further past tens of millions of vectors |
| HNSW index | Fast approximate search, no training step | Uses more memory than IVFFlat; approximate, not exact |
| Local `bge-small` embeddings | Free, private, 384 dims keeps index small | Lower quality than large API embedding models |
| Chunk per page, 800 chars, 150 overlap | Page numbers for citations; overlap avoids losing sentences at boundaries | Tune with evals (week 3) |
| 202 Accepted + status polling | Upload returns fast; ingestion is slow | Polling is simple; webhooks/SSE are nicer |
| OpenAI-compatible client | Swap Groq / Gemini / Ollama by changing env vars | — |

## Roadmap

Each version adds a feature *and* a system design concept.

- [x] **v0.1** — upload, chunk, embed, ask with citations · *API design, async processing basics*
- [ ] **v0.2** — move ingestion to a Redis-queue worker with retries · *message queues, idempotency, failure handling*
- [ ] **v0.3** — hybrid search (vector + full-text, RRF) and reranking · *indexing, search tradeoffs*
- [ ] **v0.4** — eval set + metrics (hit rate, faithfulness) · *measuring quality*
- [ ] **v0.5** — Redis answer cache, rate limiting, streaming responses · *caching, rate limiting*
- [ ] **v0.6** — Dockerfile, CI, deploy, structured logs + tracing · *observability, deployment*
