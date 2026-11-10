import pytest

from app.chunking import chunk_pages, split_text


def test_empty_text_gives_no_chunks():
    assert split_text("   \n ", 100, 10) == []


def test_short_text_is_one_chunk():
    assert split_text("Hello world.", 100, 10) == ["Hello world."]


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    pieces = split_text(text, 300, 60)
    assert len(pieces) > 1
    assert all(len(p) <= 300 for p in pieces)
    # consecutive chunks share some text (the overlap)
    for a, b in zip(pieces, pieces[1:]):
        assert b[:20] in a


def test_no_text_is_lost():
    words = [f"w{i}" for i in range(1000)]
    pieces = split_text(" ".join(words), 120, 20)
    seen = set(" ".join(pieces).split())
    assert set(words) <= seen


def test_does_not_cut_words_when_avoidable():
    text = "alpha beta gamma delta " * 50
    for p in split_text(text, 50, 10):
        for w in p.split():
            assert w in {"alpha", "beta", "gamma", "delta"}


def test_pages_keep_their_numbers_and_global_index():
    chunks = chunk_pages(["first page text", "", "third page text"], 100, 10)
    assert [(c.index, c.page) for c in chunks] == [(0, 1), (1, 3)]


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_params(size, overlap):
    with pytest.raises(ValueError):
        split_text("abc", size, overlap)
