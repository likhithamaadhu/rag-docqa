"""Ingestion pipeline: PDF -> pages -> chunks -> embeddings -> Postgres.

v1 runs this inside FastAPI BackgroundTasks (same process as the API).
That's fine for one user, but if the server restarts mid-ingest the job
is lost, and heavy embedding work slows API responses. In week 2 we move
this to a separate worker reading from a Redis queue — that's the
"async processing" topic from the system design roadmap.
"""
import asyncio
import logging
from pathlib import Path
from uuid import UUID

from pypdf import PdfReader

from app.chunking import chunk_pages
from app.config import get_settings
from app.db import pool
from app.embeddings import embed_documents

log = logging.getLogger(__name__)


def extract_pages(path: Path) -> list[str]:
    reader = PdfReader(path)
    return [page.extract_text() or "" for page in reader.pages]


async def set_status(doc_id: UUID, status: str, error: str | None = None) -> None:
    async with pool().connection() as conn:
        await conn.execute(
            "UPDATE documents SET status = %s, error = %s, updated_at = now() WHERE id = %s",
            (status, error, doc_id),
        )


async def ingest_document(doc_id: UUID, path: Path) -> None:
    settings = get_settings()
    try:
        await set_status(doc_id, "processing")

        # CPU-bound work goes to a thread so the event loop keeps serving requests.
        pages = await asyncio.to_thread(extract_pages, path)
        chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise ValueError("No extractable text (scanned PDF? OCR comes later).")
        vectors = await asyncio.to_thread(embed_documents, [c.content for c in chunks])

        async with pool().connection() as conn:
            async with conn.transaction():
                # Idempotent: re-running ingestion replaces old chunks.
                await conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
                async with conn.cursor() as cur:
                    await cur.executemany(
                        "INSERT INTO chunks (document_id, chunk_index, page, content, embedding) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        [(doc_id, c.index, c.page, c.content, v) for c, v in zip(chunks, vectors)],
                    )
        await set_status(doc_id, "ready")
        log.info("Ingested %s: %d chunks", doc_id, len(chunks))
    except Exception as exc:  # record the failure so the client can see it
        log.exception("Ingestion failed for %s", doc_id)
        await set_status(doc_id, "failed", str(exc)[:500])
