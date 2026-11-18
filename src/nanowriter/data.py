"""Pobieranie i wczytywanie tekstu książek z Wolnych Lektur."""

import json
import re
import urllib.request
from collections.abc import Callable
from pathlib import Path

WOLNELEKTURY_API = "https://wolnelektury.pl/api/books/"


def book_api_url(slug: str) -> str:
    """Zwraca adres API Wolnych Lektur z opisem książki o danym slugu.

    Przykład: "lalka-tom-pierwszy" -> "https://wolnelektury.pl/api/books/lalka-tom-pierwszy/"
    Przykład: "" -> ValueError
    Przykład: "../etc" -> ValueError
    """
    if re.fullmatch(r"[a-z0-9-]+", slug) is None:
        raise ValueError(f"Niepoprawny slug książki: {slug!r}")
    return WOLNELEKTURY_API + slug + "/"


def download_book(
    slug: str, out_dir: Path, fetch: Callable[[str], bytes] | None = None
) -> Path:
    """Pobiera plik .txt książki do out_dir/<slug>.txt (UTF-8, końce linii \\n) i zwraca ścieżkę.

    Przykład: download_book("abc", d, fetch=f), gdzie f(book_api_url("abc")) -> b'{"txt": "https://x/a.txt"}',
        a f("https://x/a.txt") -> b"ala\\r\\nma" -> d/"abc.txt" o bajtach b"ala\\nma"
    Przykład: plik d/"abc.txt" już istnieje -> d/"abc.txt" bez wywołania fetch
    Przykład: JSON bez pola "txt" albo z pustym "txt" -> ValueError, plik nie powstaje
    """
    book_api_url(slug)
    target = out_dir / f"{slug}.txt"
    if target.exists():
        return target

    if fetch is None:
        fetch = _fetch_url

    meta = json.loads(fetch(book_api_url(slug)))
    txt_url = meta.get("txt") if isinstance(meta, dict) else None
    if not isinstance(txt_url, str) or not txt_url:
        raise ValueError(f"Brak adresu pliku .txt w opisie książki: {slug!r}")

    raw = fetch(txt_url).decode("utf-8-sig")
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    part = out_dir / f"{slug}.txt.part"
    part.write_text(text, encoding="utf-8", newline="\n")
    part.replace(target)
    return target


def load_text(path: Path) -> str:
    """Wczytuje plik tekstowy UTF-8 i zwraca jego treść z końcami linii \\n.

    Przykład: plik o bajtach b"ala\\nma" -> "ala\\nma"
    Przykład: plik o bajtach b"ala\\r\\nma" -> "ala\\nma"
    Przykład: pusty plik -> ""
    """
    return path.read_text(encoding="utf-8")


def _fetch_url(url: str) -> bytes:
    """Pobiera zawartość adresu HTTP(S) i zwraca ją jako bajty.

    Przykład: _fetch_url(book_api_url("lalka-tom-pierwszy")) ->
        b'{"title":"Lalka, tom pierwszy", ...}'
    """
    request = urllib.request.Request(url, headers={"User-Agent": "nanowriter/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()
