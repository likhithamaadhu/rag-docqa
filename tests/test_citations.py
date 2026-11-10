from app.citations import cited_numbers


def test_square_brackets():
    assert cited_numbers("A [1]. B [3][2].", 5) == [1, 3, 2]


def test_full_width_brackets():
    assert cited_numbers("summary of work【1】.", 5) == [1]


def test_dagger_source_suffix():
    assert cited_numbers("claim【2†source】 and 【4†L10-L12】", 5) == [2, 4]


def test_lists_and_ranges():
    assert cited_numbers("x [1, 3] y [2-4]", 5) == [1, 3, 2, 4]


def test_out_of_range_and_duplicates_ignored():
    assert cited_numbers("[0] [1] [1] [9]", 5) == [1]


def test_non_citations_ignored():
    assert cited_numbers("see list[i] and [abc]", 5) == []
