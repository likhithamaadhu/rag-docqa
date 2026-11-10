"""Retrieval + generation."""
from dataclasses import dataclass
from uuid import UUID

from openai import AsyncOpenAI

from app.config import get_settings
from app.db import pool
from app.embeddings import embed_query

SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages.
Cite passages inline like [1] or [2][3] after each claim they support.
If the context does not contain the answer, say "I couldn't find that in the documents."
Do not use outside knowledge."""


@dataclass
class Retrieved:
    chunk_id: int
    document_id: UUID
    filename: str
    page: int | None
    content: str
    score: float  # cosine similarity, 1.0 = identical


async def retrieve(question: str, top_k: int, document_ids: list[UUID] | None) -> list[Retrieved]:
    qvec = embed_query(question)
    sql = """
        SELECT c.id, c.document_id, d.filename, c.page, c.content,
               1 - (c.embedding <=> %(q)s) AS score
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE d.status = 'ready'
          AND (%(doc_ids)s::uuid[] IS NULL OR c.document_id = ANY(%(doc_ids)s::uuid[]))
        ORDER BY c.embedding <=> %(q)s
        LIMIT %(k)s
    """
    async with pool().connection() as conn:
        rows = await (await conn.execute(sql, {"q": qvec, "doc_ids": document_ids, "k": top_k})).fetchall()
    return [Retrieved(*row) for row in rows]


def build_context(passages: list[Retrieved]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {p.filename}, page {p.page})\n{p.content}"
        for i, p in enumerate(passages, start=1)
    )



async def generate(question: str, passages: list[Retrieved]) -> str:
    s = get_settings()
    client = AsyncOpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key)
    resp = await client.chat.completions.create(
        model=s.llm_model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{build_context(passages)}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content or ""
