"""Local embeddings with sentence-transformers (free, runs on CPU).

The model is loaded once and reused: loading takes seconds, encoding a
batch takes milliseconds.
"""
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings

# bge models are trained to put this prefix on *queries* (not documents).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def _model() -> SentenceTransformer:
    return SentenceTransformer(get_settings().embedding_model)


def embed_documents(texts: list[str]) -> list[np.ndarray]:
    vectors = _model().encode(texts, batch_size=32, normalize_embeddings=True)
    return list(vectors)


def embed_query(text: str) -> np.ndarray:
    return _model().encode(QUERY_PREFIX + text, normalize_embeddings=True)
