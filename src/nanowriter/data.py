"""Pobieranie, czyszczenie i podział tekstu książek z Wolnych Lektur."""

import hashlib
import json
import random
import re
import time
import unicodedata
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

WOLNELEKTURY_API = "https://wolnelektury.pl/api/books/"
WOLNELEKTURY_CATALOG = "https://wolnelektury.pl/api/epochs/"
PROSE_EPOCHS: tuple[str, ...] = (
    "pozytywizm",
    "modernizm",
    "mloda-polska-modernizm",
    "dwudziestolecie-miedzywojenne",
)
PROSE_GENRES: tuple[str, ...] = ("powieść", "nowela", "opowiadanie")


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


@dataclass(frozen=True)
class BookEntry:
    """Pozycja katalogu Wolnych Lektur.

    Przykład: BookEntry("a", "Lalka", "Bolesław Prus", "Powieść", "Pozytywizm").slug -> "a"
    """

    slug: str
    title: str
    author: str
    genre: str
    epoch: str


def catalog_api_url(epoch: str, kind: str = "epika") -> str:
    """Zwraca adres API z listą książek danej epoki i rodzaju literackiego.

    Przykład: "pozytywizm" -> "https://wolnelektury.pl/api/epochs/pozytywizm/kinds/epika/books/"
    Przykład: ("modernizm", "liryka") ->
        "https://wolnelektury.pl/api/epochs/modernizm/kinds/liryka/books/"
    Przykład: "Młoda Polska" albo "" -> ValueError
    Przykład: ("pozytywizm", "") -> ValueError
    """
    for name, value in (("epoka", epoch), ("rodzaj", kind)):
        if re.fullmatch(r"[a-z0-9-]+", value) is None:
            raise ValueError(f"Niepoprawna {name} katalogu: {value!r}")
    return WOLNELEKTURY_CATALOG + epoch + "/kinds/" + kind + "/books/"


def list_books(
    epoch: str, fetch: Callable[[str], bytes] | None = None, kind: str = "epika"
) -> list[BookEntry]:
    """Pobiera katalog książek epoki i rodzaju i zwraca jego pozycje w kolejności z API.

    Przykład: fetch(catalog_api_url("pozytywizm")) -> b'[{"slug": "a", "title": "A",
        "author": "X", "genre": "Nowela", "epoch": "Pozytywizm"}]'
        -> [BookEntry("a", "A", "X", "Nowela", "Pozytywizm")]
    Przykład: element {"slug": "b"} -> BookEntry("b", "", "", "", "")
    Przykład: element bez "slug", ze slugiem "Zły slug", ze slugiem 5 albo nie-obiekt -> pominięty
    Przykład: odpowiedź b'[]' -> []
    Przykład: odpowiedź b'{"detail": "Nie znaleziono."}' -> ValueError
    """
    url = catalog_api_url(epoch, kind)
    if fetch is None:
        fetch = _fetch_url

    data = json.loads(fetch(url))
    if not isinstance(data, list):
        raise ValueError(f"Odpowiedź katalogu nie jest listą: {url}")

    entries: list[BookEntry] = []
    for item in data:
        if not isinstance(item, dict) or not _is_slug(item.get("slug")):
            continue
        entries.append(
            BookEntry(
                item["slug"],
                _text_field(item, "title"),
                _text_field(item, "author"),
                _text_field(item, "genre"),
                _text_field(item, "epoch"),
            )
        )
    return entries


def select_books(
    entries: list[BookEntry], genres: tuple[str, ...] = PROSE_GENRES, seed: int = 0
) -> list[BookEntry]:
    """Wybiera pozycje o gatunku z genres, bez powtórzonych slugów, w kolejności tasowanej seedem.

    Przykład: (slug, genre) = (c, "Nowela"), (a, "Powieść"), (b, "Bajka"),
        (e, "powieść historyczna"), seed=0 -> slugi ["a", "e", "c"]
    Przykład: genre "Esej, Nowela" -> wybrana (pasuje druga część)
    Przykład: genre "Poemat prozą" -> pominięta (pierwsze słowo "poemat")
    Przykład: dwie pozycje ze slugiem "a" -> tylko pierwsza z wejścia
    Przykład: [] -> []
    """
    allowed = {_normalize(genre) for genre in genres}

    seen: set[str] = set()
    chosen: list[BookEntry] = []
    for entry in entries:
        if entry.slug in seen or not _genre_matches(entry.genre, allowed):
            continue
        seen.add(entry.slug)
        chosen.append(entry)

    chosen.sort(key=lambda entry: entry.slug)
    random.Random(seed).shuffle(chosen)
    return chosen


class DownloadLimitError(RuntimeError):
    """Limit bajtów pobierania został wyczerpany, kolejne pobranie się nie odbyło."""


class LimitedFetch:
    """Pobieranie url -> bytes z limitem bajtów i pauzą przed każdym pobraniem oprócz pierwszego.

    Przykład: f = LimitedFetch(10, pause=0.5, fetch=g, sleep=s), g zawsze zwraca b"123456":
        f("u1") -> b"123456", s niewywołane, f.used == 6
        f("u2") -> s(0.5), potem b"123456", f.used == 12
        f("u3") -> DownloadLimitError (12 >= 10), g i s niewywołane, f.used == 12
    Przykład: LimitedFetch(0) -> ValueError; LimitedFetch(10, pause=-1) -> ValueError
    """

    def __init__(
        self,
        max_bytes: int,
        pause: float = 0.5,
        fetch: Callable[[str], bytes] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError(f"Limit bajtów musi być dodatni: {max_bytes}")
        if pause < 0:
            raise ValueError(f"Pauza nie może być ujemna: {pause}")
        self._max_bytes = max_bytes
        self._pause = pause
        self._fetch = fetch
        self._sleep = sleep
        self._used = 0
        self._started = 0

    @property
    def used(self) -> int:
        """Zwraca liczbę bajtów pobranych dotąd przez ten obiekt.

        Przykład: zaraz po utworzeniu -> 0
        """
        return self._used

    def __call__(self, url: str) -> bytes:
        """Pobiera url z pilnowaniem limitu i pauzy i zwraca bajty odpowiedzi.

        Przykład: patrz opis klasy
        """
        if self._used >= self._max_bytes:
            raise DownloadLimitError(
                f"Wyczerpano limit {self._max_bytes} bajtów, pominięto: {url}"
            )

        if self._started > 0 and self._pause > 0:
            sleep = self._sleep if self._sleep is not None else time.sleep
            sleep(self._pause)
        self._started += 1

        fetch = self._fetch if self._fetch is not None else _fetch_url
        body = fetch(url)
        self._used += len(body)
        return body


def clean_text(raw: str) -> str:
    """Czyści tekst książki z Wolnych Lektur: nagłówek, stopka, znaczniki, NFC, białe znaki.

    Przykład: "Autor\\n\\nTytuł\\n\\nISBN 1\\n\\n\\nAla  ma\\tkota.\\n\\n\\n\\n"
        "Kot *ma* Alę[1].\\n\\n-----\\nStopka\\n" -> "Ala ma kota.\\n\\nKot ma Alę."
    Przykład: "Ala\\r\\nma\\r\\n" -> "Ala\\nma"
    Przykład: "Cafe\\u0301" -> "Café" (NFC)
    Przykład: "Pan ***  przyszedł" -> "Pan przyszedł"
    Przykład: "Ala\\u00a0ma 😀" -> "Ala ma 😀"
    Przykład: "" -> ""; "\\n\\t \\n" -> ""
    """
    lines = _split_lines(raw)
    lines = lines[_header_end(lines) :]

    text = "\n".join(lines)
    text = re.sub(r"\[[0-9]+\]", "", text)
    text = text.replace("*", "")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = "\n".join(line.strip(" ") for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(" \n")


def is_translation(raw: str) -> bool:
    """Mówi, czy nagłówek książki ma linię tłumacza ("tłum. …" albo "przeł. …").

    Przykład: "Anton Czechow\\n\\nAptekarzowa\\ntłum. A. W.\\n\\nISBN 978\\n\\n\\nTekst" -> True
    Przykład: "Eliza Orzeszkowa\\n\\nDobra pani\\n\\nISBN 978\\n\\n\\nTekst" -> False
    Przykład: "Autor\\nPRZEŁ. X\\nISBN 1\\n\\nTekst" -> True
    Przykład: "Autor\\n\\nISBN 1\\n\\ntłum. w treści" -> False (linia po ISBN to treść)
    Przykład: "tłum. bez nagłówka" -> False (brak linii ISBN, więc brak nagłówka)
    """
    lines = _split_lines(raw)
    header = lines[: _header_end(lines)]
    return any(
        line.strip().lower().startswith(("tłum.", "przeł.")) for line in header
    )


@dataclass
class CorpusReport:
    """Wynik budowy korpusu: przyjęte książki, rozmiar czystego tekstu, pominięte i powód.

    Przykład: CorpusReport() -> books=[], clean_bytes=0, skipped={}, limit_reached=False
    """

    books: list[str] = field(default_factory=list)
    clean_bytes: int = 0
    skipped: dict[str, str] = field(default_factory=dict)
    limit_reached: bool = False


def build_corpus(
    entries: list[BookEntry],
    raw_dir: Path,
    clean_dir: Path,
    target_bytes: int,
    *,
    include_translations: bool = False,
    fetch: Callable[[str], bytes] | None = None,
    on_book: Callable[[str, str], None] | None = None,
) -> CorpusReport:
    """Pobiera i czyści książki po kolei, aż czysty tekst osiągnie target_bytes, i zwraca raport.

    Przykład: wpisy a, b, c (każda książka po czyszczeniu ma 750 bajtów), target_bytes=1000
        -> books=["a", "b"], clean_bytes=1500, c nie jest pobierana; pliki clean_dir/a.txt, b.txt
    Przykład: książka bez pliku .txt (download_book -> ValueError) -> skipped[slug] == "brak-txt"
    Przykład: nagłówek z "tłum. X", include_translations=False -> skipped[slug] == "tlumaczenie"
    Przykład: fetch rzuca DownloadLimitError -> limit_reached=True, dalsze wpisy pominięte
    Przykład: target_bytes=0 -> ValueError
    """
    if target_bytes <= 0:
        raise ValueError(f"Cel rozmiaru musi być dodatni: {target_bytes}")

    report = CorpusReport()
    seen_hashes: set[str] = set()

    def skip(slug: str, reason: str) -> None:
        report.skipped[slug] = reason
        if on_book is not None:
            on_book(slug, reason)

    for entry in entries:
        if report.clean_bytes >= target_bytes:
            break
        slug = entry.slug

        try:
            raw_path = download_book(slug, raw_dir, fetch)
        except DownloadLimitError:
            report.limit_reached = True
            if on_book is not None:
                on_book(slug, "limit")
            break
        except ValueError:
            skip(slug, "brak-txt")
            continue
        except OSError:
            skip(slug, "blad-sieci")
            continue

        raw = load_text(raw_path)
        if not include_translations and is_translation(raw):
            skip(slug, "tlumaczenie")
            continue

        text = clean_text(raw)
        if text == "":
            skip(slug, "pusty")
            continue

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            skip(slug, "duplikat")
            continue

        clean_dir.mkdir(parents=True, exist_ok=True)
        (clean_dir / f"{slug}.txt").write_text(text, encoding="utf-8", newline="\n")
        seen_hashes.add(digest)
        report.books.append(slug)
        report.clean_bytes += len(text.encode("utf-8"))
        if on_book is not None:
            on_book(slug, "ok")

    return report


def split_books(
    sizes: dict[str, int], val_fraction: float = 0.1, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Dzieli książki (slug -> rozmiar w bajtach) na (train, val) według potasowanej kolejności.

    Przykład: {"a": 100, ..., "j": 100} (10 książek), 0.1, seed=0
        -> (["a", ..., "g", "i", "j"], ["h"])
    Przykład: {"a": 50, "b": 950}, 0.1, seed=0 -> (["b"], ["a"])
    Przykład: {"a": 10, "b": 10, "c": 10}, 0.5, seed=0 -> (["b"], ["a", "c"])
    Przykład: {"a": 10} -> ValueError; val_fraction 0 albo 1 -> ValueError; rozmiar 0 -> ValueError
    """
    if len(sizes) < 2:
        raise ValueError(f"Do podziału potrzeba co najmniej 2 książek: {len(sizes)}")
    if not 0 < val_fraction < 1:
        raise ValueError(f"Udział walidacji musi być w przedziale (0, 1): {val_fraction}")
    if any(size <= 0 for size in sizes.values()):
        raise ValueError("Rozmiar każdej książki musi być dodatni")

    order = sorted(sizes)
    random.Random(seed).shuffle(order)

    threshold = val_fraction * sum(sizes.values())
    max_val_count = len(order) - 1
    val: list[str] = []
    val_bytes = 0
    for slug in order:
        if not (val_bytes < threshold and len(val) < max_val_count):
            break
        val.append(slug)
        val_bytes += sizes[slug]

    val_set = set(val)
    train = [slug for slug in sizes if slug not in val_set]
    return sorted(train), sorted(val)


@dataclass
class SplitResult:
    """Wynik podziału: slugi obu części, ścieżki plików i ich długość w znakach.

    Przykład: SplitResult(train=["a"], val=["b"], train_path=p, val_path=q, train_chars=751,
        val_chars=751)
    """

    train: list[str]
    val: list[str]
    train_path: Path
    val_path: Path
    train_chars: int
    val_chars: int


def split_corpus(
    clean_dir: Path,
    books: list[str],
    out_dir: Path,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> SplitResult:
    """Dzieli czyste książki na train/val i zapisuje train.txt, val.txt i split.json w out_dir.

    Przykład: clean_dir z a, b, c, d (po 750 znaków ASCII), books=["c", "a", "b", "d"], seed=0
        -> train=["a", "b", "d"], val=["c"], train_chars=2255, val_chars=751
    Przykład: books z powtórzonym slugiem -> ValueError
    Przykład: books=["a"] -> ValueError
    """
    if len(set(books)) != len(books):
        raise ValueError(f"Powtórzony slug książki: {books}")

    sizes = {slug: (clean_dir / f"{slug}.txt").stat().st_size for slug in books}
    train, val = split_books(sizes, val_fraction, seed)

    train_text = _join_books(clean_dir, train)
    val_text = _join_books(clean_dir, val)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.txt"
    val_path = out_dir / "val.txt"
    train_path.write_text(train_text, encoding="utf-8", newline="\n")
    val_path.write_text(val_text, encoding="utf-8", newline="\n")

    payload = {"seed": seed, "val_fraction": val_fraction, "train": train, "val": val}
    (out_dir / "split.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return SplitResult(
        train=train,
        val=val,
        train_path=train_path,
        val_path=val_path,
        train_chars=len(train_text),
        val_chars=len(val_text),
    )


def _fetch_url(url: str) -> bytes:
    """Pobiera zawartość adresu HTTP(S) i zwraca ją jako bajty.

    Przykład: _fetch_url(book_api_url("lalka-tom-pierwszy")) ->
        b'{"title":"Lalka, tom pierwszy", ...}'
    """
    request = urllib.request.Request(url, headers={"User-Agent": "nanowriter/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _is_slug(value: object) -> bool:
    """Mówi, czy wartość jest napisem pasującym w całości do wzorca slugu.

    Przykład: "a-1" -> True
    Przykład: "Zły slug" albo 5 -> False
    """
    return isinstance(value, str) and re.fullmatch(r"[a-z0-9-]+", value) is not None


def _text_field(item: dict[str, object], key: str) -> str:
    """Zwraca pole tekstowe pozycji katalogu albo pusty napis, gdy pole nie jest napisem.

    Przykład: {"title": "A"}, "title" -> "A"
    Przykład: {}, "title" -> ""
    """
    value = item.get(key)
    return value if isinstance(value, str) else ""


def _normalize(text: str) -> str:
    """Normalizuje napis do NFC i małych liter, do porównań gatunków.

    Przykład: "Powieść" -> "powieść"
    Przykład: "Cafe\\u0301" -> "café"
    """
    return unicodedata.normalize("NFC", text).lower()


def _genre_matches(genre: str, allowed: set[str]) -> bool:
    """Mówi, czy któraś część gatunku (po przecinku) zaczyna się od dozwolonego słowa.

    Przykład: "Esej, Nowela", {"nowela"} -> True
    Przykład: "Poemat prozą", {"nowela"} -> False
    Przykład: "", {"nowela"} -> False
    """
    for part in _normalize(genre).split(","):
        words = part.split()
        if words and words[0] in allowed:
            return True
    return False


def _split_lines(raw: str) -> list[str]:
    """Ujednolica tekst (końce linii, znaki niewidoczne, NFC) i usuwa stopkę od linii -----.

    Przykład: "A\\r\\nB\\n-----\\nC" -> ["A", "B"]
    Przykład: "" -> []
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    for invisible in ("﻿", "­", "​"):
        text = text.replace(invisible, "")
    text = unicodedata.normalize("NFC", text)

    lines = text.split("\n") if text else []
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].rstrip() == "-----":
            return lines[:index]
    return lines


def _header_end(lines: list[str]) -> int:
    """Zwraca liczbę linii nagłówka (do linii ISBN włącznie) albo 0, gdy go nie ma.

    Przykład: ["Autor", "", "ISBN 1", "Tekst"] -> 3
    Przykład: ["Autor", "Tekst"] -> 0
    """
    for index, line in enumerate(lines[:20]):
        if line.lstrip().startswith("ISBN"):
            return index + 1
    return 0


def _join_books(clean_dir: Path, slugs: list[str]) -> str:
    """Łączy czyste teksty książek w kolejności slugów przez pustą linię i kończy \\n.

    Przykład: a="Ala", b="Ma" w clean_dir, ["a", "b"] -> "Ala\\n\\nMa\\n"
    Przykład: [] -> "\\n"
    """
    parts = [load_text(clean_dir / f"{slug}.txt") for slug in slugs]
    return "\n\n".join(parts) + "\n"
