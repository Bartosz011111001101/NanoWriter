"""Testy znakowego tokenizera (CharTokenizer)."""

import pytest

from nanowriter.tokenizer import CharTokenizer

POLSKI = "Zażółć gęślą jaźń"


def test_class_docstring_example():
    assert CharTokenizer(["a", "b", "c"]).encode("cab") == [2, 0, 1]


def test_init_ids_are_positions():
    t = CharTokenizer(["x", "a"])
    assert t.encode("ax") == [1, 0]
    assert t.chars == ["x", "a"]


def test_init_empty():
    t = CharTokenizer([])
    assert t.vocab_size == 0
    assert t.chars == []
    assert t.encode("") == []
    assert t.decode([]) == ""
    with pytest.raises(ValueError):
        t.encode("a")


def test_init_duplicate():
    with pytest.raises(ValueError):
        CharTokenizer(["a", "a"])
    with pytest.raises(ValueError, match="x"):
        CharTokenizer(["x", "y", "x"])


@pytest.mark.parametrize(
    ("chars", "fragment"),
    [
        (["ab"], "ab"),
        ([""], None),
        (["a", "bc"], "bc"),
    ],
)
def test_init_bad_length(chars, fragment):
    with pytest.raises(ValueError) as exc_info:
        CharTokenizer(chars)
    if fragment is not None:
        assert fragment in str(exc_info.value)


def test_init_copies_list():
    lst = ["a", "b"]
    t = CharTokenizer(lst)
    lst.append("c")
    lst[0] = "z"
    assert t.vocab_size == 2
    assert t.chars == ["a", "b"]
    assert t.encode("ab") == [0, 1]
    with pytest.raises(ValueError):
        t.encode("c")


def test_from_text_sorted():
    assert CharTokenizer.from_text("abca").chars == ["a", "b", "c"]
    assert CharTokenizer.from_text("żaba").chars == ["a", "b", "ż"]


def test_from_text_empty():
    t = CharTokenizer.from_text("")
    assert t.vocab_size == 0
    assert t.chars == []


def test_from_text_polish():
    t = CharTokenizer.from_text(POLSKI)
    assert t.chars == [
        " ", "Z", "a", "g", "j", "l", "ó", "ą", "ć", "ę", "ł", "ń", "ś", "ź", "ż",
    ]
    assert t.vocab_size == 15
    assert t.encode(POLSKI) == [1, 2, 14, 6, 10, 8, 0, 3, 9, 12, 5, 7, 0, 4, 2, 13, 11]


def test_from_text_deterministic():
    first = CharTokenizer.from_text(POLSKI).chars
    second = CharTokenizer.from_text(POLSKI).chars
    assert first == second


def test_vocab_size_equals_len_chars():
    assert CharTokenizer.from_text("abca").vocab_size == 3
    for text, expected in [("", 0), ("a", 1), (POLSKI, 15), ("\n\t", 2)]:
        t = CharTokenizer.from_text(text)
        assert t.vocab_size == len(t.chars)
        assert t.vocab_size == expected


def test_chars_returns_copy():
    t = CharTokenizer.from_text("abc")
    c = t.chars
    c.append("d")
    c[0] = "z"
    assert t.chars == ["a", "b", "c"]
    assert t.encode("a") == [0]
    assert t.chars is not t.chars


def test_encode_examples():
    assert CharTokenizer.from_text("abca").encode("cab") == [2, 0, 1]
    assert CharTokenizer.from_text("żaba").encode("żaba") == [2, 0, 1, 0]
    assert CharTokenizer.from_text("abc").encode("") == []


def test_encode_returns_list_of_int():
    ids = CharTokenizer.from_text("abc").encode("ab")
    assert type(ids) is list
    assert all(type(i) is int for i in ids)
    assert ids == [0, 1]


def test_encode_unknown_char():
    t = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError, match="d"):
        t.encode("d")
    with pytest.raises(ValueError):
        t.encode("abż")


def test_decode_examples():
    assert CharTokenizer.from_text("abca").decode([2, 0, 1]) == "cab"
    assert CharTokenizer.from_text("abc").decode([]) == ""


def test_decode_boundary_ids():
    t = CharTokenizer.from_text("abc")
    assert t.decode([0]) == "a"
    assert t.decode([2]) == "c"
    assert t.decode([2, 2, 0]) == "cca"


def test_decode_out_of_range():
    t = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError, match="3"):
        t.decode([3])
    with pytest.raises(ValueError, match="-1"):
        t.decode([-1])
    with pytest.raises(ValueError):
        t.decode([0, 5])


@pytest.mark.parametrize(
    "x",
    ["", POLSKI, "\n\t", "😀", "Ala ma kota 😀\nZażółć"],
)
def test_roundtrip(x):
    t = CharTokenizer.from_text(x)
    assert t.decode(t.encode(x)) == x


def test_ids_in_range():
    text = "Zażółć gęślą jaźń 😀\n\t"
    t = CharTokenizer.from_text(text)
    ids = t.encode(text)
    assert all(0 <= i < t.vocab_size for i in ids)
    assert len(ids) == len(text) == 21
