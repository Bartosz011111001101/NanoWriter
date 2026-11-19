"""Testy katalogu Wolnych Lektur, ograniczonego pobierania i czyszczenia tekstu (moduł data)."""

import dataclasses
import json
import re
import time
import unicodedata

import pytest

import nanowriter.data
from nanowriter.data import (
    PROSE_EPOCHS,
    PROSE_GENRES,
    WOLNELEKTURY_CATALOG,
    BookEntry,
    DownloadLimitError,
    LimitedFetch,
    book_api_url,
    catalog_api_url,
    clean_text,
    download_book,
    is_translation,
    list_books,
    select_books,
)

BOM = "﻿"
SOFT_HYPHEN = "­"
ZWSP = "​"

REAL = (
    "Anton Czechow\n\nAptekarzowa\ntłum. A. W.\n\nISBN 978-83-288-2166-8\n\n\n"
    "Miasteczko B…, składające się z dwóch ulic.\n\n*\n\n— Kot *ma* Alę[1].\n\n"
    "-----\nTa lektura, podobnie jak tysiące innych, dostępna jest na stronie "
    "wolnelektury.pl.\n\nUtwór opracowany został w ramach projektu Wolne Lektury.\n\n"
    "ISBN-978-83-288-2166-8\n"
)
REAL_CLEAN = "Miasteczko B…, składające się z dwóch ulic.\n\n— Kot ma Alę."


def make_fetch(responses):
    """Sztuczny fetch: bajty, słownik url -> bajty albo lista kolejnych odpowiedzi."""
    calls = []
    queue = list(responses) if isinstance(responses, list) else None

    def fetch(url):
        calls.append(url)
        if queue is not None:
            item = queue.pop(0)
        elif isinstance(responses, dict):
            item = responses[url]
        else:
            item = responses
        if isinstance(item, BaseException):
            raise item
        return item

    return fetch, calls


def make_sleep():
    calls = []

    def sleep(seconds):
        calls.append(seconds)

    return sleep, calls


def catalog_body(items):
    return json.dumps(items, ensure_ascii=False).encode("utf-8")


def entry(slug, genre, title=""):
    return BookEntry(slug, title, "", genre, "")


def slugs(entries):
    return [e.slug for e in entries]


# --- Stałe i BookEntry ---


def test_constants():
    assert WOLNELEKTURY_CATALOG == "https://wolnelektury.pl/api/epochs/"
    assert PROSE_EPOCHS == (
        "pozytywizm",
        "modernizm",
        "mloda-polska-modernizm",
        "dwudziestolecie-miedzywojenne",
    )
    assert PROSE_GENRES == ("powieść", "nowela", "opowiadanie")


def test_book_entry_example():
    e = BookEntry("a", "Lalka", "Bolesław Prus", "Powieść", "Pozytywizm")
    assert e.slug == "a"
    assert e == BookEntry(
        slug="a", title="Lalka", author="Bolesław Prus", genre="Powieść", epoch="Pozytywizm"
    )
    assert [f.name for f in dataclasses.fields(e)] == [
        "slug",
        "title",
        "author",
        "genre",
        "epoch",
    ]


def test_book_entry_frozen():
    e = BookEntry("a", "Lalka", "Bolesław Prus", "Powieść", "Pozytywizm")
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.slug = "b"
    same = BookEntry("a", "Lalka", "Bolesław Prus", "Powieść", "Pozytywizm")
    assert e == same
    assert hash(e) == hash(same)


# --- catalog_api_url ---


def test_catalog_api_url_example():
    assert (
        catalog_api_url("pozytywizm")
        == "https://wolnelektury.pl/api/epochs/pozytywizm/kinds/epika/books/"
    )


def test_catalog_api_url_kind():
    assert (
        catalog_api_url("modernizm", "liryka")
        == "https://wolnelektury.pl/api/epochs/modernizm/kinds/liryka/books/"
    )


@pytest.mark.parametrize("epoch", PROSE_EPOCHS)
def test_catalog_api_url_prose_epochs(epoch):
    assert catalog_api_url(epoch) == WOLNELEKTURY_CATALOG + epoch + "/kinds/epika/books/"


@pytest.mark.parametrize("epoch", ["Młoda Polska", "", "a/b", "Pozytywizm", "../x", "żaba"])
def test_catalog_api_url_invalid_epoch(epoch):
    with pytest.raises(ValueError):
        catalog_api_url(epoch)


@pytest.mark.parametrize("kind", ["", "Epika", "a/b", "epika "])
def test_catalog_api_url_invalid_kind(kind):
    with pytest.raises(ValueError):
        catalog_api_url("pozytywizm", kind)


# --- list_books ---


def test_list_books_example():
    body = (
        b'[{"slug": "a", "title": "A", "author": "X", '
        b'"genre": "Nowela", "epoch": "Pozytywizm"}]'
    )
    f, calls = make_fetch(body)
    result = list_books("pozytywizm", fetch=f)
    assert result == [BookEntry("a", "A", "X", "Nowela", "Pozytywizm")]
    assert isinstance(result, list)
    assert isinstance(result[0], BookEntry)
    assert calls == [catalog_api_url("pozytywizm")]


def test_list_books_kind_url():
    f, calls = make_fetch(b"[]")
    assert list_books("modernizm", fetch=f, kind="liryka") == []
    assert calls == ["https://wolnelektury.pl/api/epochs/modernizm/kinds/liryka/books/"]


def test_list_books_missing_fields():
    f, _ = make_fetch(catalog_body([{"slug": "b"}]))
    assert list_books("pozytywizm", fetch=f) == [BookEntry("b", "", "", "", "")]


def test_list_books_non_str_fields():
    items = [{"slug": "b", "title": 5, "author": None, "genre": ["Powieść"], "epoch": {"x": 1}}]
    f, _ = make_fetch(catalog_body(items))
    assert list_books("pozytywizm", fetch=f) == [BookEntry("b", "", "", "", "")]


def test_list_books_skips_invalid():
    items = [
        {"title": "bez slugu"},
        {"slug": "Zły slug"},
        {"slug": 5},
        "tekst",
        None,
        7,
        ["a"],
        {"slug": ""},
        {"slug": "a/b"},
        {"slug": "ok-1", "title": "T"},
    ]
    f, _ = make_fetch(catalog_body(items))
    assert list_books("pozytywizm", fetch=f) == [BookEntry("ok-1", "T", "", "", "")]


def test_list_books_order_and_duplicates():
    items = [
        {"slug": "c", "url": "u", "kind": "Epika"},
        {"slug": "a"},
        {"slug": "c", "title": "drugi"},
    ]
    f, _ = make_fetch(catalog_body(items))
    result = list_books("pozytywizm", fetch=f)
    assert slugs(result) == ["c", "a", "c"]
    assert result[2].title == "drugi"


def test_list_books_polish_utf8():
    body = json.dumps(
        [{"slug": "a", "author": "Bolesław Prus", "genre": "Powieść", "title": "Żółw"}],
        ensure_ascii=False,
    ).encode("utf-8")
    f, _ = make_fetch(body)
    assert list_books("pozytywizm", fetch=f) == [
        BookEntry("a", "Żółw", "Bolesław Prus", "Powieść", "")
    ]


def test_list_books_empty():
    f, _ = make_fetch(b"[]")
    assert list_books("pozytywizm", fetch=f) == []


@pytest.mark.parametrize("body", [b'{"detail": "Nie znaleziono."}', b'"tekst"', b"5", b"null"])
def test_list_books_not_a_list(body):
    f, _ = make_fetch(body)
    with pytest.raises(ValueError):
        list_books("pozytywizm", fetch=f)


@pytest.mark.parametrize("body", [b"nie json", b"", b"[{"])
def test_list_books_invalid_json(body):
    f, _ = make_fetch(body)
    with pytest.raises(ValueError):
        list_books("pozytywizm", fetch=f)


@pytest.mark.parametrize(
    "epoch, kind",
    [("Młoda Polska", "epika"), ("pozytywizm", "Epika")],
)
def test_list_books_invalid_args_no_fetch(epoch, kind):
    f, calls = make_fetch(b"[]")
    with pytest.raises(ValueError):
        list_books(epoch, fetch=f, kind=kind)
    assert calls == []


def test_list_books_default_fetch(monkeypatch):
    f, calls = make_fetch(b'[{"slug": "a"}]')
    monkeypatch.setattr(nanowriter.data, "_fetch_url", f)
    assert list_books("pozytywizm") == [BookEntry("a", "", "", "", "")]
    assert calls == [catalog_api_url("pozytywizm")]


# --- select_books ---


def test_select_books_example():
    entries = [
        entry("c", "Nowela"),
        entry("a", "Powieść"),
        entry("b", "Bajka"),
        entry("e", "powieść historyczna"),
    ]
    result = select_books(entries, seed=0)
    assert slugs(result) == ["a", "e", "c"]
    assert slugs(select_books(entries)) == ["a", "e", "c"]
    by_slug = {e.slug: e for e in entries}
    for item in result:
        assert isinstance(item, BookEntry)
        assert item == by_slug[item.slug]


def test_select_books_seed_one():
    entries = [entry(s, "Nowela") for s in ["e", "d", "c", "b", "a"]]
    assert slugs(select_books(entries, seed=1)) == ["c", "d", "e", "a", "b"]
    assert slugs(select_books(entries, seed=2)) == ["c", "b", "d", "e", "a"]


def test_select_books_input_order_irrelevant():
    ascending = [entry(f"b{i:02d}", "Nowela") for i in range(1, 21)]
    descending = list(reversed(ascending))
    swapped = list(ascending)
    swapped[0], swapped[-1] = swapped[-1], swapped[0]
    r1 = select_books(ascending, seed=3)
    r2 = select_books(descending, seed=3)
    r3 = select_books(swapped, seed=3)
    assert r1 == r2 == r3
    assert len(r1) == 20
    assert sorted(slugs(r1)) == [f"b{i:02d}" for i in range(1, 21)]


def test_select_books_seed_changes_order():
    entries = [entry(f"b{i:02d}", "Nowela") for i in range(1, 21)]
    assert select_books(entries, seed=0) == select_books(entries, seed=0)
    assert select_books(entries, seed=0) != select_books(entries, seed=1)


@pytest.mark.parametrize(
    "genre, selected",
    [
        ("Esej, Nowela", True),
        ("Powieść,Nowela", True),
        ("Esej, Felieton", False),
        ("Esej, ", False),
        (",", False),
    ],
)
def test_select_books_multi_genre(genre, selected):
    e = entry("a", genre)
    assert select_books([e]) == ([e] if selected else [])


@pytest.mark.parametrize(
    "genre", ["Bajka", "Poemat prozą", "", "Baśń", "Esej", "Powieściopisarz"]
)
def test_select_books_rejected_genres(genre):
    assert select_books([entry("a", genre)]) == []


@pytest.mark.parametrize(
    "genre",
    [
        "POWIEŚĆ",
        "powieść dla dzieci i młodzieży",
        "Opowiadanie",
        "nowela",
        " Nowela",
        "Esej,  Powieść obyczajowa",
    ],
)
def test_select_books_case_and_subgenre(genre):
    result = select_books([entry("a", genre)])
    assert slugs(result) == ["a"]


def test_select_books_nfd_genre():
    e = entry("a", unicodedata.normalize("NFD", "Powieść"))
    assert slugs(select_books([e])) == ["a"]


def test_select_books_normalized_genres_arg():
    entries = [entry("a", "Powieść")]
    nfd = (unicodedata.normalize("NFD", "powieść"),)
    assert slugs(select_books(entries, genres=nfd)) == ["a"]
    assert slugs(select_books(entries, genres=("POWIEŚĆ",))) == ["a"]
    assert select_books(entries, genres=("nowela",)) == []


def test_select_books_custom_genres():
    entries = [entry("a", "Bajka"), entry("b", "Powieść"), entry("c", "bajka ludowa")]
    result = select_books(entries, genres=("bajka",))
    assert sorted(slugs(result)) == ["a", "c"]


def test_select_books_duplicate_slug():
    entries = [entry("a", "Powieść", "pierwszy"), entry("a", "Nowela", "drugi")]
    result = select_books(entries)
    assert len(result) == 1
    assert result[0].title == "pierwszy"


def test_select_books_duplicate_after_filter():
    entries = [entry("a", "Bajka", "x"), entry("a", "Powieść", "y")]
    result = select_books(entries)
    assert len(result) == 1
    assert result[0].title == "y"


def test_select_books_input_unchanged():
    entries = [entry("c", "Nowela"), entry("b", "Powieść"), entry("a", "Opowiadanie")]
    copy = list(entries)
    result = select_books(entries, seed=0)
    assert entries == copy
    assert result is not entries


def test_select_books_empty():
    assert select_books([]) == []
    assert select_books([entry("a", "Bajka")]) == []


# --- LimitedFetch i DownloadLimitError ---


def test_limited_fetch_example():
    log = []

    def g(url):
        log.append(("fetch", url))
        return b"123456"

    def s(seconds):
        log.append(("sleep", seconds))

    f = LimitedFetch(10, pause=0.5, fetch=g, sleep=s)
    assert f.used == 0
    assert f("u1") == b"123456"
    assert f.used == 6
    assert log == [("fetch", "u1")]
    assert f("u2") == b"123456"
    assert f.used == 12
    assert log == [("fetch", "u1"), ("sleep", 0.5), ("fetch", "u2")]
    before = list(log)
    with pytest.raises(DownloadLimitError):
        f("u3")
    assert f.used == 12
    assert log == before
    with pytest.raises(DownloadLimitError):
        f("u4")
    assert log == before


def test_limited_fetch_limit_boundary():
    g, calls = make_fetch(b"123456")
    f = LimitedFetch(12, pause=0, fetch=g)
    assert f("1") == b"123456"
    assert f("2") == b"123456"
    with pytest.raises(DownloadLimitError):
        f("3")
    assert f.used == 12
    assert len(calls) == 2

    g, calls = make_fetch(b"123456")
    f = LimitedFetch(13, pause=0, fetch=g)
    assert f("1") == b"123456"
    assert f("2") == b"123456"
    assert f("3") == b"123456"
    assert f.used == 18
    with pytest.raises(DownloadLimitError):
        f("4")
    assert len(calls) == 3


def test_limited_fetch_single_response_over_limit():
    g, calls = make_fetch(b"x" * 100)
    f = LimitedFetch(1, pause=0, fetch=g)
    assert f("1") == b"x" * 100
    assert f.used == 100
    with pytest.raises(DownloadLimitError):
        f("2")
    assert len(calls) == 1


def test_limited_fetch_pause_zero():
    g, calls = make_fetch(b"a")
    s, sleeps = make_sleep()
    f = LimitedFetch(1000, pause=0, fetch=g, sleep=s)
    for i in range(3):
        f(f"u{i}")
    assert sleeps == []
    assert len(calls) == 3


def test_limited_fetch_default_pause():
    g, _ = make_fetch(b"a")
    s, sleeps = make_sleep()
    f = LimitedFetch(1000, fetch=g, sleep=s)
    for i in range(3):
        f(f"u{i}")
    assert sleeps == [0.5, 0.5]


def test_limited_fetch_default_sleep(monkeypatch):
    g, _ = make_fetch(b"a")
    s, sleeps = make_sleep()
    monkeypatch.setattr(time, "sleep", s)
    f = LimitedFetch(1000, pause=0.25, fetch=g)
    f("u1")
    f("u2")
    assert sleeps == [0.25]


@pytest.mark.parametrize(
    "args, kwargs",
    [((0,), {}), ((-5,), {}), ((10,), {"pause": -1}), ((10,), {"pause": -0.1})],
)
def test_limited_fetch_invalid_args(args, kwargs):
    with pytest.raises(ValueError):
        LimitedFetch(*args, **kwargs)


def test_limited_fetch_used_read_only():
    f = LimitedFetch(10)
    assert f.used == 0
    with pytest.raises(AttributeError):
        f.used = 5


def test_limited_fetch_error_propagates():
    g, calls = make_fetch([b"abc", OSError("x"), b"de"])
    s, sleeps = make_sleep()
    f = LimitedFetch(100, pause=1.0, fetch=g, sleep=s)
    assert f("u1") == b"abc"
    assert f.used == 3
    with pytest.raises(OSError):
        f("u2")
    assert f.used == 3
    assert sleeps == [1.0]
    assert f("u3") == b"de"
    assert f.used == 5
    assert sleeps == [1.0, 1.0]
    assert calls == ["u1", "u2", "u3"]


def test_limited_fetch_first_call_fails():
    g, _ = make_fetch([OSError("x"), b"ok"])
    s, sleeps = make_sleep()
    f = LimitedFetch(100, pause=0.5, fetch=g, sleep=s)
    with pytest.raises(OSError):
        f("u1")
    assert sleeps == []
    assert f.used == 0
    assert f("u2") == b"ok"
    assert sleeps == [0.5]
    assert f.used == 2


def test_limited_fetch_response_unchanged():
    g, _ = make_fetch([b"\x00\xff\r\n", b"", b"z"])
    s, sleeps = make_sleep()
    f = LimitedFetch(100, pause=0.5, fetch=g, sleep=s)
    assert f("a") == b"\x00\xff\r\n"
    assert f.used == 4
    assert f("b") == b""
    assert f.used == 4
    assert f("c") == b"z"
    assert f.used == 5
    assert sleeps == [0.5, 0.5]


def test_download_limit_error_hierarchy():
    assert issubclass(DownloadLimitError, RuntimeError)
    assert not issubclass(DownloadLimitError, ValueError)
    assert not issubclass(DownloadLimitError, OSError)


def test_download_limit_error_message():
    g, _ = make_fetch(b"12345678")
    f = LimitedFetch(7, pause=0, fetch=g)
    f("u1")
    with pytest.raises(DownloadLimitError) as excinfo:
        f("https://x/b.txt")
    message = str(excinfo.value)
    assert "https://x/b.txt" in message
    assert "7" in message


def test_limited_fetch_default_fetch_late_binding(monkeypatch):
    f = LimitedFetch(100, pause=0)
    g, calls = make_fetch(b"abc")
    monkeypatch.setattr(nanowriter.data, "_fetch_url", g)
    assert f("u") == b"abc"
    assert calls == ["u"]
    assert f.used == 3


def test_limited_fetch_with_download_book(tmp_path):
    meta = b'{"txt": "https://x/a.txt"}'
    g, calls = make_fetch({book_api_url("abc"): meta, "https://x/a.txt": b"ala"})
    lf = LimitedFetch(10**6, pause=0, fetch=g)
    download_book("abc", tmp_path, fetch=lf)
    assert (tmp_path / "abc.txt").read_bytes() == b"ala"
    assert calls == [book_api_url("abc"), "https://x/a.txt"]
    assert lf.used == len(meta) + 3


def test_limited_fetch_with_list_books():
    body = b'[{"slug": "a"}]'
    g, _ = make_fetch(body)
    lf = LimitedFetch(10**6, pause=0, fetch=g)
    assert list_books("pozytywizm", fetch=lf) == [BookEntry("a", "", "", "", "")]
    assert lf.used == len(body)


# --- clean_text ---


@pytest.mark.parametrize(
    "raw, expected",
    [
        (
            (
                "Autor\n\nTytuł\n\nISBN 1\n\n\nAla  ma\tkota.\n\n\n\n"
                "Kot *ma* Alę[1].\n\n-----\nStopka\n"
            ),
            "Ala ma kota.\n\nKot ma Alę.",
        ),
        ("Ala\r\nma\r\n", "Ala\nma"),
        ("Café", "Café"),
        ("Pan ***  przyszedł", "Pan przyszedł"),
        ("Ala ma 😀", "Ala ma 😀"),
        ("", ""),
        ("\n\t \n", ""),
    ],
)
def test_clean_text_docstring_examples(raw, expected):
    assert clean_text(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Zażółć gęślą jaźń", "Zażółć gęślą jaźń"),
        ("Ala 😀 ma 🐈", "Ala 😀 ma 🐈"),
        ("\n\t", ""),
    ],
)
def test_clean_text_edge_text(raw, expected):
    result = clean_text(raw)
    assert isinstance(result, str)
    assert result == expected


def test_clean_text_line_endings():
    assert clean_text("a\rb\r\n\r\nc") == "a\nb\n\nc"


def test_clean_text_invisible_chars():
    raw = BOM + "Ala ma" + SOFT_HYPHEN + " ko" + ZWSP + "ta"
    assert clean_text(raw) == "Ala ma kota"


def test_clean_text_header_line_19():
    raw = "\n".join(["h"] * 19 + ["ISBN 1", "Treść"])
    assert clean_text(raw) == "Treść"


def test_clean_text_header_line_20():
    raw = "\n".join(["h"] * 20 + ["ISBN 1", "Treść"])
    assert clean_text(raw) == raw


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Autor\n  ISBN 978\nTreść", "Treść"),
        ("A\nISBN 1\nB\nISBN 2\nC", "B\nISBN 2\nC"),
        ("A\nisbn 1\nB", "A\nisbn 1\nB"),
        ("Autor\n\nTytuł\n\nISBN 1", ""),
        (BOM + "ISBN 1\nTreść", "Treść"),
    ],
)
def test_clean_text_header_variants(raw, expected):
    assert clean_text(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Treść\n-----\nŚrodek\n-----\nStopka", "Treść\n-----\nŚrodek"),
        ("Treść\n----- \nStopka", "Treść"),
        ("Treść\n -----\nStopka", "Treść\n-----\nStopka"),
        ("Treść\n------\nStopka", "Treść\n------\nStopka"),
        ("-----\nStopka", ""),
    ],
)
def test_clean_text_footer_variants(raw, expected):
    assert clean_text(raw) == expected


def test_clean_text_footer_before_header():
    assert clean_text("Tekst\n-----\nISBN-978\nStopka") == "Tekst"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("a[12]b [a] [1a] [] [ 1]", "ab [a] [1a] [] [ 1]"),
        ("x[١]y [１]", "x[١]y [１]"),
        ("[1]", ""),
    ],
)
def test_clean_text_footnotes(raw, expected):
    assert clean_text(raw) == expected


@pytest.mark.parametrize("raw, expected", [("*a* **b** ***", "a b"), ("*", "")])
def test_clean_text_stars(raw, expected):
    assert clean_text(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("a\tb\x0bc\x0cd e f　g", "a b c d e f g"),
        ("a b\x85c\x1cd", "a b c d"),
    ],
)
def test_clean_text_whitespace_kinds(raw, expected):
    assert clean_text(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  Ala  \n\t ma \n", "Ala\nma"),
        ("a\n\n\nb\n \n\n \nc", "a\n\nb\n\nc"),
        ("a\n\nb", "a\n\nb"),
        ("a\nb", "a\nb"),
    ],
)
def test_clean_text_lines_and_blank_lines(raw, expected):
    assert clean_text(raw) == expected


def test_clean_text_keeps_words():
    x = "Ala  ma\n\n\n kota,\ta pies — psa."
    assert clean_text(x) == "Ala ma\n\nkota, a pies — psa."
    assert clean_text(x).split() == x.split()


def test_clean_text_real_book():
    assert clean_text(REAL) == REAL_CLEAN
    assert clean_text(REAL.replace("\n", "\r\n")) == REAL_CLEAN


@pytest.mark.parametrize(
    "raw",
    [
        REAL,
        "Autor\n\nTytuł\n\nISBN 1\n\n\nAla  ma\tkota.\n\n\n\nKot *ma* Alę[1].\n\n-----\nStopka\n",
        BOM + "  *Ala*" + SOFT_HYPHEN + "\t\t ma [3] kota\r\n\r\n\r\n\r\n  Pies" + ZWSP
        + " \x0b\n\n\n",
        "Café \n \n \n\tx  ",
    ],
)
def test_clean_text_invariants(raw):
    r = clean_text(raw)
    for forbidden in ["\r", "\t", BOM, SOFT_HYPHEN, ZWSP, "*", "  ", "\n\n\n", " \n", "\n "]:
        assert forbidden not in r
    assert re.search(r"\[[0-9]+\]", r) is None
    assert r == r.strip()
    assert unicodedata.normalize("NFC", r) == r
    for c in r:
        if c.isspace():
            assert c in (" ", "\n")


# --- is_translation ---


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Anton Czechow\n\nAptekarzowa\ntłum. A. W.\n\nISBN 978\n\n\nTekst", True),
        ("Eliza Orzeszkowa\n\nDobra pani\n\nISBN 978\n\n\nTekst", False),
        ("Autor\nPRZEŁ. X\nISBN 1\n\nTekst", True),
        ("Autor\n\nISBN 1\n\ntłum. w treści", False),
        ("tłum. bez nagłówka", False),
    ],
)
def test_is_translation_docstring_examples(raw, expected):
    assert is_translation(raw) is expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Autor\nprzeł. Jan Kowalski\nISBN 1\nTekst", True),
        ("Autor\n   tłum. X\nISBN 1\nTekst", True),
        ("Autor\nTŁUM. X\nISBN 1\nTekst", True),
        ("Anton Czechow\r\n\r\nAptekarzowa\r\ntłum. A. W.\r\n\r\nISBN 978\r\n\r\n\r\nTekst", True),
        ("Autor\rtłum. X\rISBN 1\rTekst", True),
        (BOM + "Autor\ntłum. X\nISBN 1\nTekst", True),
        ("Autor\ntłumaczenie X\nISBN 1\nTekst", False),
        ("", False),
    ],
)
def test_is_translation_variants(raw, expected):
    result = is_translation(raw)
    assert isinstance(result, bool)
    assert result is expected


def test_is_translation_header_limit():
    inside = "\n".join(["Autor", "tłum. X"] + ["h"] * 17 + ["ISBN 1", "Tekst"])
    outside = "\n".join(["Autor", "tłum. X"] + ["h"] * 18 + ["ISBN 1", "Tekst"])
    assert is_translation(inside) is True
    assert is_translation(outside) is False


def test_is_translation_ignores_footer():
    assert is_translation("Tekst\n-----\ntłum. X\nISBN-978") is False


def test_is_translation_real_book():
    assert is_translation(REAL) is True
    assert is_translation(REAL.replace("tłum. A. W.\n", "")) is False


# --- Ogólne ---


def test_no_output_no_files(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    f, _ = make_fetch(b'[{"slug": "a", "genre": "Nowela"}]')
    g, _ = make_fetch(b"abc")
    s, _ = make_sleep()

    catalog_api_url("pozytywizm")
    listed = list_books("pozytywizm", fetch=f)
    select_books(listed)
    limited = LimitedFetch(100, pause=0.5, fetch=g, sleep=s)
    limited("u1")
    limited("u2")
    clean_text(REAL)
    is_translation(REAL)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert list(tmp_path.iterdir()) == []
