"""Split page text into overlapping chunks.

Pure functions with no dependencies, so they are easy to unit test.

Why overlap? A sentence that answers a question may straddle a chunk
boundary. Repeating the tail of one chunk at the head of the next keeps
that sentence whole in at least one chunk.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    index: int
    page: int
    content: str


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into pieces of at most `size` characters with `overlap`.

    Prefers to break at a paragraph, sentence or word boundary inside the
    last 20% of the window so chunks don't cut words in half.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    text = " ".join(text.split())  # normalise whitespace
    if not text:
        return []

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window_floor = start + int(size * 0.8)
            for sep in (". ", "? ", "! ", "; ", " "):
                cut = text.rfind(sep, window_floor, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        # Don't start the next chunk mid-word: skip forward to a word start.
        if next_start < end and text[next_start - 1] != " ":
            space = text.find(" ", next_start, end)
            next_start = space + 1 if space != -1 else end
        start = next_start
    return pieces


def chunk_pages(pages: list[str], size: int, overlap: int) -> list[Chunk]:
    """Chunk each page separately so every chunk knows its page number
    (needed for citations like "see page 4")."""
    chunks: list[Chunk] = []
    for page_no, page_text in enumerate(pages, start=1):
        for piece in split_text(page_text, size, overlap):
            chunks.append(Chunk(index=len(chunks), page=page_no, content=piece))
    return chunks
