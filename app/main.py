import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db import close_pool, open_pool, pool
from app.ingest import ingest_document
from app.citations import cited_numbers
from app.rag import generate, retrieve

logging.basicConfig(level=logging.INFO)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(get_settings().upload_dir).mkdir(parents=True, exist_ok=True)
    await open_pool()
    yield
    await close_pool()


app = FastAPI(title="RAG Doc Q&A", version="0.1.0", lifespan=lifespan)


# ---------- schemas ----------
class DocumentOut(BaseModel):
    id: UUID
    filename: str
    status: str
    error: str | None = None
    chunk_count: int = 0


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    document_ids: list[UUID] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    number: int
    document_id: UUID
    filename: str
    page: int | None
    snippet: str
    score: float


class AskOut(BaseModel):
    answer: str
    citations: list[Citation]
    latency_ms: int


# ---------- routes ----------
@app.get("/health")
async def health():
    async with pool().connection() as conn:
        await conn.execute("SELECT 1")
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentOut, status_code=202)
async def upload_document(file: UploadFile, background: BackgroundTasks):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(415, "Only PDF files are supported for now.")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 20 MB).")

    doc_id = uuid.uuid4()
    path = Path(get_settings().upload_dir) / f"{doc_id}.pdf"
    path.write_bytes(data)

    async with pool().connection() as conn:
        await conn.execute(
            "INSERT INTO documents (id, filename) VALUES (%s, %s)", (doc_id, file.filename)
        )
    # 202 Accepted: we return immediately; ingestion continues in the background.
    background.add_task(ingest_document, doc_id, path)
    return DocumentOut(id=doc_id, filename=file.filename, status="pending")


@app.get("/documents/{doc_id}", response_model=DocumentOut)
async def get_document(doc_id: UUID):
    async with pool().connection() as conn:
        row = await (await conn.execute(
            """SELECT d.id, d.filename, d.status, d.error, count(c.id)
               FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
               WHERE d.id = %s GROUP BY d.id""",
            (doc_id,),
        )).fetchone()
    if row is None:
        raise HTTPException(404, "Document not found.")
    return DocumentOut(id=row[0], filename=row[1], status=row[2], error=row[3], chunk_count=row[4])


@app.post("/ask", response_model=AskOut)
async def ask(body: AskIn):
    started = time.perf_counter()
    top_k = body.top_k or get_settings().top_k
    passages = await retrieve(body.question, top_k, body.document_ids)
    if not passages:
        raise HTTPException(404, "No ready documents to search. Upload a PDF first.")

    answer = await generate(body.question, passages)
    citations = [
        Citation(
            number=n,
            document_id=passages[n - 1].document_id,
            filename=passages[n - 1].filename,
            page=passages[n - 1].page,
            snippet=passages[n - 1].content[:240],
            score=round(passages[n - 1].score, 4),
        )
        for n in cited_numbers(answer, len(passages))
    ]
    return AskOut(
        answer=answer,
        citations=citations,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
