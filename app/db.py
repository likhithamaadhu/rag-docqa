"""Postgres connection pool (psycopg 3) with pgvector support."""
from psycopg_pool import AsyncConnectionPool
from pgvector.psycopg import register_vector_async

from app.config import get_settings

_pool: AsyncConnectionPool | None = None


async def _configure(conn) -> None:
    await register_vector_async(conn)


async def open_pool() -> None:
    global _pool
    _pool = AsyncConnectionPool(
        get_settings().database_url,
        min_size=1,
        max_size=10,
        configure=_configure,
        open=False,
    )
    await _pool.open()


async def close_pool() -> None:
    if _pool is not None:
        await _pool.close()


def pool() -> AsyncConnectionPool:
    if _pool is None:
        raise RuntimeError("DB pool not opened; is the app lifespan running?")
    return _pool
