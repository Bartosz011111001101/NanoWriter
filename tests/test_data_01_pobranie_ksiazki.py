"""Testy pobierania książki i wczytywania tekstu (moduł data)."""

import json
import urllib.request
from pathlib import Path

import pytest

import nanowriter.data
from nanowriter.data import WOLNELEKTURY_API, book_api_url, download_book, load_text

TXT_URL = "https://x/a.txt"


def make_fetch(content: bytes = b"ala", api_json: dict | None = None):
    """Zwraca parę (sztuczny fetch, lista wywołań)."""
    if api_json is None:
        api_json = {"txt": TXT_URL}
    calls: list[str] = []
    api_url = book_api_url("abc")

    def fetch(url: str) -> bytes:
        calls.append(url)
        if url == api_url:
            return json.dumps(api_json).encode()
        if url == TXT_URL:
            return content
        raise AssertionError(f"nieoczekiwany URL: {url}")

    return fetch, calls


def test_book_api_url_example():
    assert (
        book_api_url("lalka-tom-pierwszy")
        == "https://wolnelektury.pl/api/books/lalka-tom-pierwszy/"
    )


def test_book_api_url_digits_and_constant():
    assert book_api_url("abc-123") == "https://wolnelektury.pl/api/books/abc-123/"
    assert WOLNELEKTURY_API == "https://wolnelektury.pl/api/books/"


@pytest.mark.parametrize("slug", ["", "../etc", "Lalka", "a b", "żaba", "a/b", "a.b"])
def test_book_api_url_invalid(slug):
    with pytest.raises(ValueError):
        book_api_url(slug)


def test_book_api_url_error_message_contains_slug():
    with pytest.raises(ValueError) as exc:
        book_api_url("Lalka")
    assert "Lalka" in str(exc.value)


def test_download_book_example(tmp_path):
    fetch, calls = make_fetch(b"ala\r\nma")
    result = download_book("abc", tmp_path, fetch=fetch)
    assert isinstance(result, Path)
    assert result == tmp_path / "abc.txt"
    assert result.read_bytes() == b"ala\nma"
    assert calls == [book_api_url("abc"), TXT_URL]


def test_download_book_polish_text(tmp_path):
    fetch, _ = make_fetch("Zażółć gęślą jaźń\r\n".encode())
    download_book("abc", tmp_path, fetch=fetch)
    assert (tmp_path / "abc.txt").read_bytes() == "Zażółć gęślą jaźń\n".encode()


def test_download_book_emoji_and_whitespace(tmp_path):
    fetch, _ = make_fetch("😀\r\n\t".encode())
    download_book("abc", tmp_path, fetch=fetch)
    assert (tmp_path / "abc.txt").read_bytes() == "😀\n\t".encode()


def test_download_book_strips_bom(tmp_path):
    fetch, _ = make_fetch(b"\xef\xbb\xbfala")
    download_book("abc", tmp_path, fetch=fetch)
    assert (tmp_path / "abc.txt").read_bytes() == b"ala"


def test_download_book_lone_cr(tmp_path):
    fetch, _ = make_fetch(b"a\rb")
    download_book("abc", tmp_path, fetch=fetch)
    assert (tmp_path / "abc.txt").read_bytes() == b"a\nb"


def test_download_book_mixed_line_endings(tmp_path):
    fetch, _ = make_fetch(b"a\r\nb\nc\rd")
    download_book("abc", tmp_path, fetch=fetch)
    assert (tmp_path / "abc.txt").read_bytes() == b"a\nb\nc\nd"


def test_download_book_existing_file_not_fetched(tmp_path):
    (tmp_path / "abc.txt").write_text("stare", encoding="utf-8")
    fetch, calls = make_fetch()
    result = download_book("abc", tmp_path, fetch=fetch)
    assert result == tmp_path / "abc.txt"
    assert (tmp_path / "abc.txt").read_text(encoding="utf-8") == "stare"
    assert calls == []


def test_download_book_invalid_slug_before_fetch(tmp_path):
    fetch, calls = make_fetch()
    with pytest.raises(ValueError):
        download_book("../etc", tmp_path, fetch=fetch)
    assert calls == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "api_json",
    [{}, {"txt": ""}, {"txt": None}, {"txt": 5}],
)
def test_download_book_missing_txt(tmp_path, api_json):
    fetch, calls = make_fetch(api_json=api_json)
    with pytest.raises(ValueError, match="abc"):
        download_book("abc", tmp_path, fetch=fetch)
    assert not (tmp_path / "abc.txt").exists()
    assert not (tmp_path / "abc.txt.part").exists()
    assert calls == [book_api_url("abc")]


def test_download_book_creates_nested_dir(tmp_path):
    out_dir = tmp_path / "a" / "b"
    fetch, _ = make_fetch(b"ala")
    result = download_book("abc", out_dir, fetch=fetch)
    assert result == tmp_path / "a" / "b" / "abc.txt"
    assert result.read_bytes() == b"ala"


def test_download_book_existing_dir_ok(tmp_path):
    fetch, _ = make_fetch(b"ala")
    download_book("abc", tmp_path, fetch=fetch)
    assert (tmp_path / "abc.txt").read_bytes() == b"ala"


def test_download_book_empty_text(tmp_path):
    fetch, _ = make_fetch(b"")
    result = download_book("abc", tmp_path, fetch=fetch)
    assert result.exists()
    assert result.read_bytes() == b""
    assert load_text(result) == ""


def test_download_book_no_part_left(tmp_path):
    fetch, _ = make_fetch(b"ala")
    download_book("abc", tmp_path, fetch=fetch)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["abc.txt"]


def test_download_book_overwrites_leftover_part(tmp_path):
    (tmp_path / "abc.txt.part").write_text("smieci", encoding="utf-8")
    fetch, _ = make_fetch(b"nowe")
    download_book("abc", tmp_path, fetch=fetch)
    assert (tmp_path / "abc.txt").read_bytes() == b"nowe"
    assert not (tmp_path / "abc.txt.part").exists()


def test_download_book_default_fetch_uses_fetch_url(tmp_path, monkeypatch):
    fetch, calls = make_fetch(b"ala")
    monkeypatch.setattr(nanowriter.data, "_fetch_url", fetch)
    download_book("abc", tmp_path)
    assert (tmp_path / "abc.txt").read_bytes() == b"ala"
    assert calls == [book_api_url("abc"), TXT_URL]


def test_fetch_url_request(monkeypatch):
    captured: dict = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self, *args):
            return b"tresc"

    def fake(req, *args, **kwargs):
        captured["req"] = req
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    result = nanowriter.data._fetch_url(TXT_URL)  # pylint: disable=protected-access
    assert result == b"tresc"
    req = captured["req"]
    assert isinstance(req, urllib.request.Request)
    assert req.full_url == TXT_URL
    assert req.get_method() == "GET"
    assert req.get_header("User-agent") == "nanowriter/0.1"
    if "timeout" in captured["kwargs"]:
        timeout = captured["kwargs"]["timeout"]
    else:
        timeout = captured["args"][1]
    assert timeout == 30


def test_download_book_prints_nothing(tmp_path, capsys):
    fetch, _ = make_fetch(b"ala\r\nma")
    download_book("abc", tmp_path, fetch=fetch)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b"ala\nma", "ala\nma"),
        (b"ala\r\nma", "ala\nma"),
        (b"", ""),
        (b"\n\t", "\n\t"),
        ("Zażółć gęślą jaźń".encode(), "Zażółć gęślą jaźń"),
        ("😀".encode(), "😀"),
    ],
)
def test_load_text_cases(tmp_path, raw, expected):
    path = tmp_path / "t.txt"
    path.write_bytes(raw)
    assert load_text(path) == expected


def test_load_text_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_text(tmp_path / "brak.txt")


def test_load_text_after_download(tmp_path):
    fetch, _ = make_fetch(b"a\r\nb\r\n")
    assert load_text(download_book("abc", tmp_path, fetch=fetch)) == "a\nb\n"
