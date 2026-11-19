"""Testy etapu 2: budowa korpusu, podział train/val i skrypt prepare_data.py."""

import dataclasses
import json
import runpy
import sys
import time
import urllib.error
from collections.abc import Callable
from pathlib import Path

import pytest

import nanowriter.data
from nanowriter.data import (
    PROSE_EPOCHS,
    BookEntry,
    CorpusReport,
    DownloadLimitError,
    LimitedFetch,
    SplitResult,
    book_api_url,
    build_corpus,
    clean_text,
    list_books,
    load_text,
    select_books,
    split_books,
    split_corpus,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"


def body(s: str) -> str:
    return ("Tekst ksiazki " + s) * 50


def raw(tekst: str) -> str:
    return "A\n\nT\n\nISBN 1\n\n\n" + tekst + "\n\n-----\nS\n"


def raw_tr(tekst: str) -> str:
    return (
        "Anton Czechow\n\nAptekarzowa\ntłum. A. W.\n\nISBN 978\n\n\n" + tekst + "\n\n-----\nS\n"
    )


RAW_EMPTY = "A\n\nT\n\nISBN 1\n\n\n\n\n-----\nS\n"


def entries(*slugs: str) -> list[BookEntry]:
    return [BookEntry(s, "", "", "Nowela", "") for s in slugs]


def make_api(
    catalog: dict[str, list[str]] | None = None,
    texts: dict[str, str] | None = None,
    *,
    no_txt: tuple[str, ...] = (),
    bad_json: tuple[str, ...] = (),
    errors: dict[str, Exception] | None = None,
) -> tuple[Callable[[str], bytes], list[str]]:
    catalog = catalog or {}
    texts = texts or {}
    errors = errors or {}
    calls: list[str] = []
    known = set(texts) | set(no_txt) | set(bad_json) | set(errors)
    by_url = {book_api_url(s): s for s in known}

    def fetch(url: str) -> bytes:
        calls.append(url)
        if "/epochs/" in url:
            epoch = url.split("/epochs/")[1].split("/")[0]
            if epoch not in catalog:
                return b'{"detail": "Nie znaleziono."}'
            items = [{"slug": s, "genre": "Nowela"} for s in catalog[epoch]]
            return json.dumps(items).encode()
        if url in by_url:
            slug = by_url[url]
            if slug in errors:
                raise errors[slug]
            if slug in no_txt:
                return b'{"txt": ""}'
            if slug in bad_json:
                return b"nie json"
            return json.dumps({"txt": f"https://x/{slug}.txt"}).encode()
        if url.startswith("https://x/") and url.endswith(".txt"):
            return texts[url[len("https://x/") : -len(".txt")]].encode("utf-8")
        raise AssertionError(f"nieoczekiwany URL: {url}")

    return fetch, calls


def log_book() -> tuple[Callable[[str, str], None], list[tuple[str, str]]]:
    log: list[tuple[str, str]] = []

    def on_book(slug: str, status: str) -> None:
        log.append((slug, status))

    return on_book, log


def texts_for(*slugs: str) -> dict[str, str]:
    return {s: raw(body(s)) for s in slugs}


def build(tmp_path: Path, slugs: tuple[str, ...], texts: dict[str, str], target: int = 10**9,
          **kwargs):
    fetch = kwargs.pop("fetch", None)
    calls: list[str] = []
    if fetch is None:
        fetch, calls = make_api(texts=texts, **kwargs.pop("api", {}))
    report = build_corpus(
        entries(*slugs), tmp_path / "raw", tmp_path / "clean", target, fetch=fetch, **kwargs
    )
    return report, calls


def write_clean(clean_dir: Path, texts: dict[str, str]) -> None:
    clean_dir.mkdir(parents=True, exist_ok=True)
    for s, t in texts.items():
        (clean_dir / f"{s}.txt").write_text(t, encoding="utf-8", newline="\n")


def abcd_clean(tmp_path: Path) -> Path:
    clean_dir = tmp_path / "clean"
    write_clean(clean_dir, {s: body(s) for s in "abcd"})
    return clean_dir


# --- Typy ---


def test_corpus_report_defaults():
    r = CorpusReport()
    assert r.books == []
    assert r.clean_bytes == 0
    assert r.skipped == {}
    assert r.limit_reached is False
    assert dataclasses.is_dataclass(CorpusReport)
    assert [f.name for f in dataclasses.fields(CorpusReport)] == [
        "books", "clean_bytes", "skipped", "limit_reached",
    ]


def test_corpus_report_independent_defaults():
    r1, r2 = CorpusReport(), CorpusReport()
    r1.books.append("a")
    r1.skipped["b"] = "pusty"
    r1.clean_bytes = 5
    assert r2.books == []
    assert r2.skipped == {}
    assert r1.clean_bytes == 5


def test_split_result_fields(tmp_path):
    p, q = tmp_path / "t.txt", tmp_path / "v.txt"
    r = SplitResult(train=["a"], val=["b"], train_path=p, val_path=q, train_chars=751,
                    val_chars=751)
    assert r.train == ["a"]
    assert r.val == ["b"]
    assert r.train_path == p
    assert r.val_path == q
    assert r.train_chars == 751
    assert r.val_chars == 751
    assert [f.name for f in dataclasses.fields(SplitResult)] == [
        "train", "val", "train_path", "val_path", "train_chars", "val_chars",
    ]
    with pytest.raises(TypeError):
        SplitResult()


# --- build_corpus ---


def test_build_corpus_docstring_example(tmp_path):
    report, calls = build(tmp_path, ("a", "b", "c"), texts_for("a", "b", "c"), 1000)
    assert isinstance(report, CorpusReport)
    assert report.books == ["a", "b"]
    assert report.clean_bytes == 1500
    assert report.skipped == {}
    assert report.limit_reached is False
    assert (tmp_path / "clean" / "a.txt").exists()
    assert (tmp_path / "clean" / "b.txt").exists()
    assert not (tmp_path / "clean" / "c.txt").exists()
    assert book_api_url("c") not in calls
    assert "https://x/c.txt" not in calls


def test_build_corpus_target_reached_exactly(tmp_path):
    report, calls = build(tmp_path / "s1", ("a", "b", "c"), texts_for("a", "b", "c"), 1500)
    assert report.books == ["a", "b"]
    assert book_api_url("c") not in calls
    report, _ = build(tmp_path / "s2", ("a", "b", "c"), texts_for("a", "b", "c"), 1501)
    assert report.books == ["a", "b", "c"]
    assert report.clean_bytes == 2250


def test_build_corpus_target_in_utf8_bytes(tmp_path):
    texts = {"a": raw("ąę" * 300), "b": raw(body("b"))}
    report, calls = build(tmp_path, ("a", "b"), texts, 1000)
    assert report.books == ["a"]
    assert report.clean_bytes == 1200
    assert book_api_url("b") not in calls
    assert (tmp_path / "clean" / "a.txt").read_bytes() == ("ąę" * 300).encode("utf-8")


def test_build_corpus_clean_file_bytes(tmp_path):
    surowy = raw("Zażółć gęślą jaźń.\n\nDrugi  akapit 😀 *wyróżnienie*[1].")
    report, _ = build(tmp_path, ("a",), {"a": surowy})
    expected = clean_text(surowy).encode("utf-8")
    data = (tmp_path / "clean" / "a.txt").read_bytes()
    assert data == expected
    assert b"\r" not in data
    assert not data.endswith(b"\n")
    assert report.clean_bytes == len(expected)
    assert "ISBN 1" in (tmp_path / "raw" / "a.txt").read_text(encoding="utf-8")


def test_build_corpus_clean_bytes_sum(tmp_path):
    texts = {
        "a": raw(body("a")),
        "b": raw("Krótki tekst."),
        "c": raw("Łódź " * 10 + "koniec"),
        "d": raw("x"),
    }
    report, _ = build(tmp_path, ("a", "b", "c", "d"), texts)
    assert report.books == ["a", "b", "c", "d"]
    total = sum(len((tmp_path / "clean" / f"{s}.txt").read_bytes()) for s in report.books)
    assert report.clean_bytes == total


def test_build_corpus_missing_txt(tmp_path):
    on_book, log = log_book()
    report, _ = build(
        tmp_path, ("a", "b", "c"), texts_for("b", "c"), on_book=on_book, api={"no_txt": ("a",)}
    )
    assert report.skipped == {"a": "brak-txt"}
    assert report.books == ["b", "c"]
    assert report.clean_bytes == 1500
    assert log == [("a", "brak-txt"), ("b", "ok"), ("c", "ok")]
    assert not (tmp_path / "raw" / "a.txt").exists()
    assert not (tmp_path / "clean" / "a.txt").exists()


def test_build_corpus_bad_json(tmp_path):
    report, _ = build(tmp_path, ("a", "b"), texts_for("b"), api={"bad_json": ("a",)})
    assert report.skipped == {"a": "brak-txt"}
    assert report.books == ["b"]


@pytest.mark.parametrize(
    "exc",
    [
        OSError("x"),
        urllib.error.URLError("x"),
        TimeoutError("x"),
        urllib.error.HTTPError("https://x", 500, "x", None, None),
    ],
    ids=["oserror", "urlerror", "timeout", "httperror"],
)
def test_build_corpus_network_errors(tmp_path, exc):
    report, _ = build(tmp_path, ("a", "b"), texts_for("b"), api={"errors": {"a": exc}})
    assert report.skipped == {"a": "blad-sieci"}
    assert report.books == ["b"]
    assert report.limit_reached is False


@pytest.mark.parametrize("exc_type", [RuntimeError, KeyError])
def test_build_corpus_other_errors_propagate(tmp_path, exc_type):
    fetch, _ = make_api(errors={"a": exc_type("x")})
    with pytest.raises(exc_type) as info:
        build_corpus(entries("a"), tmp_path / "raw", tmp_path / "clean", 1000, fetch=fetch)
    assert not isinstance(info.value, DownloadLimitError)


def test_build_corpus_translation_skipped(tmp_path):
    texts = {"t": raw_tr(body("t")), "a": raw(body("a"))}
    report, _ = build(tmp_path, ("t", "a"), texts)
    assert report.skipped == {"t": "tlumaczenie"}
    assert report.books == ["a"]
    assert report.clean_bytes == 750
    assert not (tmp_path / "clean" / "t.txt").exists()
    assert (tmp_path / "raw" / "t.txt").exists()


def test_build_corpus_translation_included(tmp_path):
    texts = {"t": raw_tr(body("t")), "a": raw(body("a"))}
    report, _ = build(tmp_path, ("t", "a"), texts, include_translations=True)
    assert report.books == ["t", "a"]
    assert report.skipped == {}
    assert (tmp_path / "clean" / "t.txt").read_bytes().decode("utf-8") == body("t")


def test_build_corpus_translation_from_cached_raw(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "t.txt").write_text(raw_tr(body("t")), encoding="utf-8", newline="\n")
    fetch, calls = make_api(texts={"t": raw_tr(body("t"))})
    report, _ = build(tmp_path, ("t",), {}, fetch=fetch)
    assert report.skipped == {"t": "tlumaczenie"}
    assert calls == []


def test_build_corpus_empty_text(tmp_path):
    texts = {"e": RAW_EMPTY, "a": raw(body("a"))}
    report, _ = build(tmp_path, ("e", "a"), texts)
    assert report.skipped == {"e": "pusty"}
    assert report.books == ["a"]
    assert not (tmp_path / "clean" / "e.txt").exists()


def test_build_corpus_duplicate_content(tmp_path):
    texts = {
        "a": raw(body("a")),
        "b": "Inny Autor\n\nInny tytuł\n\nISBN 2\n\n\n" + body("a") + "\n\n-----\nInna stopka\n",
    }
    report, _ = build(tmp_path, ("a", "b"), texts)
    assert report.books == ["a"]
    assert report.skipped == {"b": "duplikat"}
    assert report.clean_bytes == 750
    assert not (tmp_path / "clean" / "b.txt").exists()


def test_build_corpus_repeated_slug(tmp_path):
    report, calls = build(tmp_path, ("a", "a", "b"), texts_for("a", "b"))
    assert report.books == ["a", "b"]
    assert report.skipped == {"a": "duplikat"}
    assert calls.count("https://x/a.txt") == 1


def test_build_corpus_cached_raw_no_fetch(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "a.txt").write_text(raw(body("a")), encoding="utf-8", newline="\n")
    report, calls = build(tmp_path, ("a", "b"), texts_for("a", "b"))
    assert report.books == ["a", "b"]
    assert book_api_url("a") not in calls
    assert "https://x/a.txt" not in calls
    assert book_api_url("b") in calls


def _limit_n() -> int:
    fetch, _ = make_api(texts=texts_for("a"))
    return len(fetch(book_api_url("a"))) + len(raw(body("a")).encode("utf-8"))


def test_build_corpus_limit_between_books(tmp_path):
    g, calls = make_api(texts=texts_for("a", "b", "c"))
    lf = LimitedFetch(_limit_n(), pause=0, fetch=g)
    on_book, log = log_book()
    report, _ = build(tmp_path, ("a", "b", "c"), {}, fetch=lf, on_book=on_book)
    assert report.books == ["a"]
    assert report.limit_reached is True
    assert report.skipped == {}
    assert log == [("a", "ok"), ("b", "limit")]
    assert not any(u.endswith(("/b/", "b.txt", "/c/", "c.txt")) for u in calls)
    assert not (tmp_path / "raw" / "b.txt").exists()


def test_build_corpus_limit_inside_book(tmp_path):
    g, calls = make_api(texts=texts_for("a", "b", "c"))
    lf = LimitedFetch(_limit_n() + 1, pause=0, fetch=g)
    on_book, log = log_book()
    report, _ = build(tmp_path, ("a", "b", "c"), {}, fetch=lf, on_book=on_book)
    assert report.books == ["a"]
    assert report.limit_reached is True
    assert report.skipped == {}
    assert log == [("a", "ok"), ("b", "limit")]
    assert book_api_url("b") in calls
    assert "https://x/b.txt" not in calls
    assert not (tmp_path / "raw" / "b.txt").exists()
    assert book_api_url("c") not in calls


def test_build_corpus_limit_error_direct(tmp_path):
    def fetch(url: str) -> bytes:
        raise DownloadLimitError("x")

    on_book, log = log_book()
    report, _ = build(tmp_path, ("a", "b"), {}, fetch=fetch, on_book=on_book)
    assert report.books == []
    assert report.clean_bytes == 0
    assert report.limit_reached is True
    assert report.skipped == {}
    assert log == [("a", "limit")]


def test_build_corpus_on_book_order(tmp_path):
    texts = {
        "a": raw(body("a")),
        "t": raw_tr(body("t")),
        "e": RAW_EMPTY,
        "b": raw(body("a")),
        "c": raw(body("c")),
    }
    on_book, log = log_book()
    report, _ = build(
        tmp_path, ("a", "t", "e", "b", "x", "c"), texts, on_book=on_book, api={"no_txt": ("x",)}
    )
    assert log == [
        ("a", "ok"), ("t", "tlumaczenie"), ("e", "pusty"), ("b", "duplikat"),
        ("x", "brak-txt"), ("c", "ok"),
    ]
    assert report.books == ["a", "c"]
    assert report.skipped == {
        "t": "tlumaczenie", "e": "pusty", "b": "duplikat", "x": "brak-txt",
    }


def test_build_corpus_on_book_none(tmp_path, capsys):
    report, _ = build(tmp_path, ("a", "b"), texts_for("a", "b"))
    assert report.books == ["a", "b"]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_build_corpus_creates_clean_dir(tmp_path):
    clean_dir = tmp_path / "x" / "y" / "clean"
    fetch, _ = make_api(texts=texts_for("a"))
    build_corpus(entries("a"), tmp_path / "raw", clean_dir, 10**9, fetch=fetch)
    assert (clean_dir / "a.txt").exists()


def test_build_corpus_empty_entries(tmp_path):
    fetch, calls = make_api()
    result = build_corpus([], tmp_path / "raw", tmp_path / "clean", 1000, fetch=fetch)
    assert result == CorpusReport()
    assert calls == []


@pytest.mark.parametrize("target", [0, -1])
def test_build_corpus_invalid_target(tmp_path, target):
    fetch, calls = make_api(texts=texts_for("a"))
    with pytest.raises(ValueError):
        build_corpus(entries("a"), tmp_path / "raw", tmp_path / "clean", target, fetch=fetch)
    assert calls == []
    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "clean").exists()


def test_build_corpus_keyword_only(tmp_path):
    with pytest.raises(TypeError):
        build_corpus([], tmp_path / "raw", tmp_path / "clean", 1000, True)  # type: ignore[misc]


def test_build_corpus_default_fetch(tmp_path, monkeypatch):
    f, calls = make_api(texts=texts_for("a", "b"))
    monkeypatch.setattr(nanowriter.data, "_fetch_url", f)
    report = build_corpus(entries("a", "b"), tmp_path / "raw", tmp_path / "clean", 10**9)
    assert report.books == ["a", "b"]
    assert book_api_url("a") in calls
    assert "https://x/b.txt" in calls


# --- split_books ---

TEN = {s: 100 for s in "abcdefghij"}


def test_split_books_ten_equal():
    result = split_books(dict(TEN), 0.1, seed=0)
    assert result == (["a", "b", "c", "d", "e", "f", "g", "i", "j"], ["h"])
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], list)
    assert isinstance(result[1], list)


def test_split_books_n_minus_one_two_books():
    assert split_books({"a": 50, "b": 950}, 0.1, seed=0) == (["b"], ["a"])


def test_split_books_half():
    assert split_books({"a": 10, "b": 10, "c": 10}, 0.5, seed=0) == (["b"], ["a", "c"])


def test_split_books_n_minus_one_bound():
    assert split_books({"a": 1, "b": 1, "c": 1}, 0.99, seed=0) == (["b"], ["a", "c"])
    assert split_books({"a": 10, "b": 10}, 0.99, seed=0) == (["b"], ["a"])


def test_split_books_threshold_boundary():
    train, val = split_books(dict(TEN), 0.2, 0)
    assert val == ["h", "i"]
    assert train == ["a", "b", "c", "d", "e", "f", "g", "j"]
    train, val = split_books(dict(TEN), 0.21, 0)
    assert val == ["b", "h", "i"]
    assert train == ["a", "c", "d", "e", "f", "g", "j"]


def test_split_books_other_seed():
    result = split_books(dict(TEN), 0.1, seed=1)
    assert result == (["a", "b", "c", "d", "e", "f", "h", "i", "j"], ["g"])


def test_split_books_defaults_and_determinism():
    assert split_books(dict(TEN)) == split_books(dict(TEN), 0.1, 0)
    assert split_books(dict(TEN), 0.3, 5) == split_books(dict(TEN), 0.3, 5)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_split_books_key_order_irrelevant(seed):
    reversed_sizes = {s: 100 for s in "jihgfedcba"}
    assert split_books(dict(TEN), 0.1, seed) == split_books(reversed_sizes, 0.1, seed)


def test_split_books_input_unchanged_new_lists():
    sizes = dict(TEN)
    copy = dict(sizes)
    train, val = split_books(sizes)
    assert sizes == copy
    train.append("z")
    val.clear()
    assert split_books(sizes) == (["a", "b", "c", "d", "e", "f", "g", "i", "j"], ["h"])


@pytest.mark.parametrize("seed", range(10))
def test_split_books_properties(seed):
    sizes = {"a": 5, "b": 300, "c": 40, "d": 77, "e": 1, "f": 900, "g": 12, "h": 64,
             "i": 250, "j": 3}
    train, val = split_books(sizes, 0.1, seed)
    assert train == sorted(train)
    assert val == sorted(val)
    assert train
    assert val
    assert not set(train) & set(val)
    assert sorted(train + val) == sorted(sizes)
    assert sum(sizes[s] for s in val) >= 0.1 * sum(sizes.values()) or len(val) == 9


@pytest.mark.parametrize(
    "args",
    [
        ({},),
        ({"a": 10},),
        ({"a": 10, "b": 10}, 0),
        ({"a": 10, "b": 10}, 1),
        ({"a": 10, "b": 10}, -0.1),
        ({"a": 10, "b": 10}, 1.5),
        ({"a": 10, "b": 0},),
        ({"a": 10, "b": -5},),
    ],
)
def test_split_books_errors(args):
    with pytest.raises(ValueError):
        split_books(*args)


# --- split_corpus ---


def test_split_corpus_docstring_example(tmp_path):
    clean_dir = abcd_clean(tmp_path)
    out = tmp_path / "out"
    r = split_corpus(clean_dir, ["c", "a", "b", "d"], out, seed=0)
    assert isinstance(r, SplitResult)
    assert r.train == ["a", "b", "d"]
    assert r.val == ["c"]
    assert r.train_path == out / "train.txt"
    assert r.val_path == out / "val.txt"
    assert r.train_chars == 2255
    assert r.val_chars == 751


def test_split_corpus_file_contents(tmp_path):
    clean_dir = tmp_path / "clean"
    write_clean(clean_dir, {"a": "Ala\nma", "b": "Bolek", "c": "Cezary\n\nCelina", "d": "Dom"})
    out = tmp_path / "out"
    r = split_corpus(clean_dir, ["d", "c", "b", "a"], out, seed=0)
    assert r.train == ["a", "b", "d"]
    assert r.val == ["c"]
    assert (out / "train.txt").read_bytes() == b"Ala\nma\n\nBolek\n\nDom\n"
    assert (out / "val.txt").read_bytes() == b"Cezary\n\nCelina\n"
    assert r.train_chars == 19
    assert r.val_chars == 15


def test_split_corpus_split_json(tmp_path):
    clean_dir = abcd_clean(tmp_path)
    out = tmp_path / "out"
    split_corpus(clean_dir, ["c", "a", "b", "d"], out, val_fraction=0.1, seed=0)
    text = (out / "split.json").read_bytes().decode("utf-8")
    expected = json.dumps(
        {"seed": 0, "val_fraction": 0.1, "train": ["a", "b", "d"], "val": ["c"]},
        ensure_ascii=False,
        indent=2,
    ) + "\n"
    assert text == expected
    assert "\r" not in text
    assert list(json.loads(text)) == ["seed", "val_fraction", "train", "val"]


def test_split_corpus_args_passed(tmp_path):
    clean_dir = tmp_path / "clean"
    write_clean(clean_dir, {s: s * 100 for s in "abcdefghij"})
    books = list("abcdefghij")
    r = split_corpus(clean_dir, books, tmp_path / "o1", val_fraction=0.1, seed=1)
    assert r.val == ["g"]
    meta = json.loads((tmp_path / "o1" / "split.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 1
    assert meta["val_fraction"] == 0.1
    assert meta["val"] == ["g"]
    r = split_corpus(clean_dir, books, tmp_path / "o2", val_fraction=0.2, seed=0)
    assert r.val == ["h", "i"]
    meta = json.loads((tmp_path / "o2" / "split.json").read_text(encoding="utf-8"))
    assert meta["val_fraction"] == 0.2


def test_split_corpus_sizes_in_bytes_lengths_in_chars(tmp_path):
    clean_dir = tmp_path / "clean"
    write_clean(clean_dir, {"a": "ą" * 100, "b": "b" * 100, "c": "c" * 100})
    out = tmp_path / "out"
    r = split_corpus(clean_dir, ["a", "b", "c"], out, val_fraction=0.5, seed=0)
    assert r.train == ["b", "c"]
    assert r.val == ["a"]
    assert r.train_chars == 203
    assert r.val_chars == 101
    assert (out / "val.txt").read_bytes().decode("utf-8") == "ą" * 100 + "\n"


def test_split_corpus_deterministic(tmp_path):
    clean_dir = abcd_clean(tmp_path)
    o1, o2 = tmp_path / "o1", tmp_path / "o2"
    r1 = split_corpus(clean_dir, ["c", "a", "b", "d"], o1, seed=0)
    r2 = split_corpus(clean_dir, ["c", "a", "b", "d"], o2, seed=0)
    for name in ("train.txt", "val.txt", "split.json"):
        assert (o1 / name).read_bytes() == (o2 / name).read_bytes()
    assert (r1.train, r1.val, r1.train_chars, r1.val_chars) == (
        r2.train, r2.val, r2.train_chars, r2.val_chars,
    )


def test_split_corpus_creates_out_dir(tmp_path):
    clean_dir = abcd_clean(tmp_path)
    out = tmp_path / "x" / "y"
    split_corpus(clean_dir, ["c", "a", "b", "d"], out, seed=0)
    for name in ("train.txt", "val.txt", "split.json"):
        assert (out / name).exists()


def test_split_corpus_duplicate_slug(tmp_path):
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        split_corpus(clean_dir, ["a", "a", "b"], out)
    assert not out.exists()


def test_split_corpus_single_book(tmp_path):
    clean_dir = tmp_path / "clean"
    write_clean(clean_dir, {"a": body("a")})
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        split_corpus(clean_dir, ["a"], out)
    assert not (out / "train.txt").exists()


def test_split_corpus_missing_file(tmp_path):
    clean_dir = tmp_path / "clean"
    write_clean(clean_dir, {"a": body("a")})
    with pytest.raises(FileNotFoundError):
        split_corpus(clean_dir, ["a", "x"], tmp_path / "out")


def test_split_corpus_no_output_to_console(tmp_path, capsys):
    clean_dir = abcd_clean(tmp_path)
    split_corpus(clean_dir, ["c", "a", "b", "d"], tmp_path / "out", seed=0)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# --- Bramka etapu 2 ---

BOILERPLATE = "Ta lektura, podobnie jak tysiące innych, jest dostępna on-line."


def _gate_api() -> tuple[Callable[[str], bytes], list[str]]:
    slugs = [f"nowela-{i:02d}" for i in range(1, 11)]
    texts = {}
    for i, s in enumerate(slugs, start=1):
        akapity = "\n\n".join(
            f"Nowela numer {i:02d}, akapit {k}: zażółć gęślą jaźń." for k in range(20)
        )
        texts[s] = (
            "Autor\n\nTytuł noweli\n\nISBN 978-83-288-0000-0\n\n\n"
            + akapity
            + "\n\n-----\n"
            + BOILERPLATE
            + "\n"
        )
    return make_api(catalog={"pozytywizm": slugs}, texts=texts)


def _gate_run(d: Path):
    api, _ = _gate_api()
    fetcher = LimitedFetch(10**7, pause=0, fetch=api)
    catalog = list_books("pozytywizm", fetch=fetcher)
    selected = select_books(catalog, seed=0)
    report = build_corpus(selected, d / "raw", d / "clean", 10**9, fetch=fetcher)
    result = split_corpus(d / "clean", report.books, d, 0.1, 0)
    return report, result


def test_gate_stage2_split_disjoint(tmp_path):
    d = tmp_path / "run"
    report, result = _gate_run(d)
    assert len(report.books) == 10
    assert report.skipped == {}
    assert report.limit_reached is False
    for name in ("train.txt", "val.txt", "split.json"):
        assert (d / name).exists()
    assert result.train
    assert result.val
    assert not set(result.train) & set(result.val)
    assert sorted(result.train + result.val) == sorted(report.books)

    train_text = (d / "train.txt").read_text(encoding="utf-8")
    val_text = (d / "val.txt").read_text(encoding="utf-8")
    parts = {"train": (result.train, train_text, val_text), "val": (result.val, val_text,
                                                                    train_text)}
    for books, own, other in parts.values():
        for s in books:
            text = load_text(d / "clean" / f"{s}.txt")
            assert text in own
            assert text not in other

    meta = json.loads((d / "split.json").read_text(encoding="utf-8"))
    assert meta["train"] == result.train
    assert meta["val"] == result.val

    sizes = {s: (d / "clean" / f"{s}.txt").stat().st_size for s in report.books}
    assert sum(sizes[s] for s in result.val) >= 0.1 * sum(sizes.values())

    for text in (train_text, val_text):
        assert "ISBN" not in text
        assert "Ta lektura" not in text
        assert "-----" not in text
    assert result.train_chars == len(train_text)
    assert result.val_chars == len(val_text)


def test_gate_stage2_reproducible(tmp_path):
    r1, _ = _gate_run(tmp_path / "r1")
    r2, _ = _gate_run(tmp_path / "r2")
    for name in ("train.txt", "val.txt", "split.json"):
        assert (tmp_path / "r1" / name).read_bytes() == (tmp_path / "r2" / name).read_bytes()
    assert r1.books == r2.books


# --- Skrypt scripts/prepare_data.py ---


@pytest.fixture
def script_globals():
    return runpy.run_path(str(SCRIPT))


@pytest.fixture
def parse_args(script_globals):
    return script_globals["parse_args"]


@pytest.fixture
def main(script_globals):
    return script_globals["main"]


def run_scenario(
    main, monkeypatch, d, extra=(), *, catalog=None, texts=None, pause0=True, api=None
):
    if api is None:
        catalog = catalog or {"pozytywizm": list("abcd")}
        texts = texts or texts_for("a", "b", "c", "d")
        api = make_api(catalog=catalog, texts=texts)
    fetch, calls = api
    monkeypatch.setattr(nanowriter.data, "_fetch_url", fetch)
    argv = ["--data-dir", str(d)]
    if not any(x == "--epochs" for x in extra):
        argv += ["--epochs", *catalog_epochs(catalog)]
    argv += list(extra)
    if pause0:
        argv += ["--pause", "0"]
    return main(argv), calls


def catalog_epochs(catalog):
    return list(catalog) if catalog else ["pozytywizm"]


def stdout_lines(capsys) -> list[str]:
    return capsys.readouterr().out.splitlines()


def test_script_import_silent(capsys):
    g = runpy.run_path(str(SCRIPT))
    assert callable(g["parse_args"])
    assert callable(g["main"])
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_parse_args_defaults(parse_args):
    a = parse_args([])
    assert a.data_dir == Path("data")
    assert isinstance(a.data_dir, Path)
    assert a.epochs == list(PROSE_EPOCHS)
    assert isinstance(a.epochs, list)
    assert a.target_mb == 28.0
    assert a.max_download_mb == 200.0
    assert a.pause == 0.5
    assert a.seed == 0
    assert a.val_fraction == 0.1
    assert a.translations is False


def test_parse_args_epochs_translations(parse_args):
    a = parse_args(["--epochs", "pozytywizm", "--translations"])
    assert a.epochs == ["pozytywizm"]
    assert a.translations is True


def test_parse_args_no_translations_and_many_epochs(parse_args):
    assert parse_args(["--no-translations"]).translations is False
    assert parse_args(["--translations", "--no-translations"]).translations is False
    assert parse_args(["--epochs", "a", "b"]).epochs == ["a", "b"]


def test_parse_args_all_options_types(parse_args):
    a = parse_args([
        "--data-dir", "x", "--target-mb", "1.5", "--max-download-mb", "3", "--pause", "0",
        "--seed", "7", "--val-fraction", "0.25",
    ])
    assert a.data_dir == Path("x")
    assert isinstance(a.data_dir, Path)
    assert a.target_mb == 1.5
    assert a.max_download_mb == 3.0
    assert isinstance(a.max_download_mb, float)
    assert a.pause == 0.0
    assert isinstance(a.pause, float)
    assert a.seed == 7
    assert isinstance(a.seed, int)
    assert a.val_fraction == 0.25


@pytest.mark.parametrize("argv", [["--epochs"], ["--nieznana"]])
def test_parse_args_errors(parse_args, argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_parse_args_none_uses_argv(parse_args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prepare_data.py", "--seed", "5"])
    assert parse_args().seed == 5
    assert parse_args(None).seed == 5


def test_main_done_scenario(main, monkeypatch, tmp_path, capsys):
    d = tmp_path / "d"
    r, _ = run_scenario(main, monkeypatch, d)
    assert isinstance(r, SplitResult)
    assert r.train == ["a", "b", "d"]
    assert r.val == ["c"]
    assert r.train_chars == 2255
    assert r.val_chars == 751
    assert r.train_path == d / "train.txt"
    assert r.val_path == d / "val.txt"
    assert stdout_lines(capsys) == [
        "katalog: 4 pozycji, wybrane: 4",
        "ok           c",
        "ok           a",
        "ok           b",
        "ok           d",
        "ksiazki: 4, czysty tekst: 0.0 MB, pobrano: 0.0 MB, pominiete: 0",
        f"train: 3 ksiazek, 2255 znakow -> {d / 'train.txt'}",
        f"val: 1 ksiazek, 751 znakow -> {d / 'val.txt'}",
    ]


def test_main_files_layout(main, monkeypatch, tmp_path):
    d = tmp_path / "d"
    _, calls = run_scenario(main, monkeypatch, d)
    for s in "abcd":
        assert (d / "raw" / f"{s}.txt").exists()
        assert (d / "clean" / f"{s}.txt").exists()
    for name in ("train.txt", "val.txt", "split.json"):
        assert (d / name).exists()
    meta = json.loads((d / "split.json").read_text(encoding="utf-8"))
    assert meta == {"seed": 0, "val_fraction": 0.1, "train": ["a", "b", "d"], "val": ["c"]}
    assert sum(1 for u in calls if "/epochs/pozytywizm/" in u) == 1


def test_main_rerun_uses_raw_cache(main, monkeypatch, tmp_path):
    d = tmp_path / "d"
    api = make_api(catalog={"pozytywizm": list("abcd")}, texts=texts_for("a", "b", "c", "d"))
    run_scenario(main, monkeypatch, d, api=api, catalog={"pozytywizm": []})
    api[1].clear()
    r, calls = run_scenario(main, monkeypatch, d, api=api, catalog={"pozytywizm": []})
    assert len(calls) == 1
    assert "/epochs/" in calls[0]
    assert r.train == ["a", "b", "d"]
    assert r.val == ["c"]


def test_main_multiple_epochs(main, monkeypatch, tmp_path, capsys):
    d = tmp_path / "d"
    catalog = {"pozytywizm": ["a", "b", "c"], "modernizm": ["c", "d"]}
    r, calls = run_scenario(
        main, monkeypatch, d, ("--epochs", "pozytywizm", "modernizm"),
        catalog=catalog, texts=texts_for("a", "b", "c", "d"),
    )
    assert stdout_lines(capsys)[0] == "katalog: 5 pozycji, wybrane: 4"
    poz = next(i for i, u in enumerate(calls) if "/epochs/pozytywizm/" in u)
    mod = next(i for i, u in enumerate(calls) if "/epochs/modernizm/" in u)
    assert poz < mod
    assert r.train == ["a", "b", "d"]
    assert r.val == ["c"]


def _translation_scenario(main, monkeypatch, d, extra=()):
    texts = texts_for("a", "b", "c")
    texts["t"] = raw_tr(body("t"))
    return run_scenario(
        main, monkeypatch, d, extra, catalog={"pozytywizm": ["a", "b", "c", "t"]}, texts=texts
    )


def test_main_translations_default_skipped(main, monkeypatch, tmp_path, capsys):
    d = tmp_path / "d"
    r, _ = _translation_scenario(main, monkeypatch, d)
    lines = stdout_lines(capsys)
    assert lines[1:5] == [
        "ok           c", "ok           a", "ok           b", "tlumaczenie  t",
    ]
    assert "ksiazki: 3, czysty tekst: 0.0 MB, pobrano: 0.0 MB, pominiete: 1" in lines
    assert r.train == ["b", "c"]
    assert r.val == ["a"]
    assert r.train_chars == 1503
    assert r.val_chars == 751
    assert not (d / "clean" / "t.txt").exists()


def test_main_translations_flag(main, monkeypatch, tmp_path, capsys):
    d = tmp_path / "d"
    r, _ = _translation_scenario(main, monkeypatch, d, ("--translations",))
    lines = stdout_lines(capsys)
    assert "ok           t" in lines
    assert "ksiazki: 4, czysty tekst: 0.0 MB, pobrano: 0.0 MB, pominiete: 0" in lines
    assert r.train == ["a", "b", "t"]
    assert r.val == ["c"]


def test_main_target_mb(main, monkeypatch, tmp_path, capsys):
    d = tmp_path / "d"
    r, calls = run_scenario(main, monkeypatch, d, ("--target-mb", "0.001"))
    lines = stdout_lines(capsys)
    assert lines[1:3] == ["ok           c", "ok           a"]
    assert "ksiazki: 2, czysty tekst: 0.0 MB, pobrano: 0.0 MB, pominiete: 0" in lines
    assert not any(line.startswith("ok") and line.endswith((" b", " d")) for line in lines)
    assert r.train == ["c"]
    assert r.val == ["a"]
    assert book_api_url("b") not in calls
    assert book_api_url("d") not in calls


def test_main_megabytes_format(main, monkeypatch, tmp_path, capsys):
    d = tmp_path / "d"
    texts = {s: raw(("Tekst ksiazki " + s) * 9334) for s in "abcd"}
    run_scenario(main, monkeypatch, d, texts=texts)
    lines = stdout_lines(capsys)
    assert "ksiazki: 4, czysty tekst: 0.6 MB, pobrano: 0.6 MB, pominiete: 0" in lines


def test_main_download_limit(main, monkeypatch, tmp_path, capsys):
    d = tmp_path / "d"
    catalog = {"pozytywizm": list("abcd")}
    texts = texts_for("a", "b", "c", "d")
    measure, _ = make_api(catalog=catalog, texts=texts)
    n = len(measure("https://wolnelektury.pl/api/epochs/pozytywizm/kinds/epika/books/"))
    n += 2 * (len(measure(book_api_url("a"))) + len(raw(body("a")).encode("utf-8")))
    r, calls = run_scenario(
        main, monkeypatch, d, ("--max-download-mb", str(n / 1_000_000)),
        catalog=catalog, texts=texts,
    )
    lines = stdout_lines(capsys)
    assert lines[1:] == [
        "ok           c",
        "ok           a",
        "limit        b",
        "ksiazki: 2, czysty tekst: 0.0 MB, pobrano: 0.0 MB, pominiete: 0",
        "uwaga: osiagnieto limit pobierania",
        f"train: 1 ksiazek, 751 znakow -> {d / 'train.txt'}",
        f"val: 1 ksiazek, 751 znakow -> {d / 'val.txt'}",
    ]
    assert r.train == ["c"]
    assert r.val == ["a"]
    assert not any(u.endswith(("/d/", "d.txt")) for u in calls)


def test_main_no_limit_warning(main, monkeypatch, tmp_path, capsys):
    run_scenario(main, monkeypatch, tmp_path / "d")
    assert not any(line.startswith("uwaga") for line in stdout_lines(capsys))


def test_main_pause_single_fetcher(main, monkeypatch, tmp_path):
    pauses: list[float] = []
    monkeypatch.setattr(time, "sleep", pauses.append)
    run_scenario(main, monkeypatch, tmp_path / "d1", pause0=False)
    assert pauses == [0.5] * 8
    pauses.clear()
    run_scenario(main, monkeypatch, tmp_path / "d2", ("--pause", "0.2"), pause0=False)
    assert pauses == [0.2] * 8


def test_main_polish_text(main, monkeypatch, tmp_path):
    d = tmp_path / "d"
    texts = {s: raw((f"Zażółć gęślą jaźń {s}. " * 40).strip()) for s in "abcd"}
    r, _ = run_scenario(main, monkeypatch, d, texts=texts)
    train_text = (d / "train.txt").read_text(encoding="utf-8")
    val_text = (d / "val.txt").read_text(encoding="utf-8")
    assert "Zażółć gęślą jaźń" in train_text
    assert r.train_chars == len(train_text)
    assert r.val_chars == len(val_text)


def test_main_unknown_epoch(main, monkeypatch, tmp_path):
    d = tmp_path / "d"
    with pytest.raises(ValueError):
        run_scenario(
            main, monkeypatch, d, ("--epochs", "nieznana"),
            catalog={"pozytywizm": ["a", "b"]}, texts=texts_for("a", "b"),
        )
    assert not (d / "train.txt").exists()


def test_main_single_book(main, monkeypatch, tmp_path):
    d = tmp_path / "d"
    with pytest.raises(ValueError):
        run_scenario(
            main, monkeypatch, d, catalog={"pozytywizm": ["a"]}, texts=texts_for("a")
        )
    assert not (d / "train.txt").exists()


@pytest.mark.parametrize("option", ["--target-mb", "--max-download-mb"])
def test_main_invalid_sizes(main, monkeypatch, tmp_path, option):
    with pytest.raises(ValueError):
        run_scenario(main, monkeypatch, tmp_path / "d", (option, "0"))
