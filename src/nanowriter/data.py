"""Pobieranie, czyszczenie i podział tekstu książek z Wolnych Lektur."""

import codecs
import hashlib
import json
import random
import re
import time
import unicodedata
import urllib.request
import zlib
from collections import Counter, deque
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO

WOLNELEKTURY_API = "https://wolnelektury.pl/api/books/"
WOLNELEKTURY_CATALOG = "https://wolnelektury.pl/api/epochs/"
PROSE_EPOCHS: tuple[str, ...] = (
    "pozytywizm",
    "modernizm",
    "mloda-polska-modernizm",
    "dwudziestolecie-miedzywojenne",
)
PROSE_GENRES: tuple[str, ...] = ("powieść", "nowela", "opowiadanie")
OPUS_SUBTITLES_PL_URL: str = (
    "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/pl.txt.gz"
)
PROFANITY_STEMS: tuple[str, ...] = (
    "kurw", "kurew", "chuj", "huj", "pierdol", "pierdal", "jeb", "pizd",
    "gówn", "dziwk", "cipk", "fiut", "sukinsyn", "fuck", "shit",
)
PROFANITY_WORDS: tuple[str, ...] = (
    "dupa", "dupy", "dupie", "dupę", "dupek", "dupku", "cholera", "cholerny",
    "cholerne", "cholerę", "kutas", "ciul",
)
ENGLISH_MARKERS: tuple[str, ...] = (
    "the", "and", "you", "your", "what", "that", "this", "with", "have", "why", "how",
    "are", "is", "i'm", "don't", "it's",
)

_ALLOWED_TEXT = re.compile(r"[0-9A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźżÀ-ÖØ-öø-ÿ .,!?:;'\"()\-–—…„”«»%]+")
_LEADING_DASHES = re.compile(r"^(?:[-–—]+\s*)+")
_WORD_RE = re.compile(r"[^\W\d_]+")
_CREDIT_RE = re.compile(
    r"\bnapisy\b|\bnapisów\b|\bnapisach\b|tłumaczeni|opensubtitles|subtitles"
    r"|synchroniz|korekta|www\.|http|\.com|\.pl\b|@|\bfps\b"
)
_PROFANITY_STEMS_RE = re.compile("|".join(re.escape(stem) for stem in PROFANITY_STEMS))
_PROFANITY_WORDS_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in PROFANITY_WORDS) + r")\b"
)
_ENGLISH_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in ENGLISH_MARKERS) + r")\b"
)
ASSISTANT_PHRASES: tuple[str, ...] = (
    "oczywiście! oto", "oto lista", "jako asystent", "jako model", "jako sztuczna inteligencja",
    "w czym mogę pomóc", "chętnie pomogę", "mogę ci pomóc",
)

_DIALOG_LIST_RE = re.compile(r"^(?:[-–—]\s|\d+[.)]\s)")


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


class SubtitleFilter:
    """Czyści i filtruje linie napisów jedna po drugiej, pamiętając ostatnio przyjęte.

    Przykład: accept("- Dzień dobry, panie Kowalski.") -> "Dzień dobry, panie Kowalski."
    Przykład: accept("Nie.") -> None (za krótka), counts["krotka"] == 1
    Przykład: accept("UWAŻAJ NA SIEBIE!") -> None (wielkie litery)
    Przykład: accept("Co za kurwa dzień.") -> None (przekleństwo)
    Przykład: accept("Why am I so happy?") -> None (angielski)
    Przykład: accept("♪ La la la ♪") -> None (znaki spoza zestawu)
    Przykład: accept("Napisy: michnik") -> None (podpis twórcy napisów)
    Przykład: accept("Jestem  zmęczony.") -> "Jestem zmęczony." (zwinięte spacje)
    Przykład: dwa razy z rzędu "Dzień dobry." -> "Dzień dobry.", potem None (powtórka)
    Przykład: SubtitleFilter(window=0, max_repeats=2), "Nie wiem, co robić." cztery razy
        -> "Nie wiem, co robić.", "Nie wiem, co robić.", None, None
    """

    counts: Counter[str]

    def __init__(self, min_words: int = 2, window: int = 20, max_repeats: int = 20) -> None:
        """Tworzy pusty filtr; min_words i max_repeats muszą być >= 1, window >= 0.

        Przykład: SubtitleFilter().counts -> Counter()
        Przykład: SubtitleFilter(min_words=0) albo window=-1 albo max_repeats=0 -> ValueError
        """
        if min_words < 1 or max_repeats < 1 or window < 0:
            raise ValueError(
                f"Niepoprawne progi: min_words={min_words}, window={window}, "
                f"max_repeats={max_repeats}"
            )
        self._min_words = min_words
        self._max_repeats = max_repeats
        self.counts = Counter()
        self._recent: deque[str] = deque(maxlen=window)
        self._short_seen: Counter[str] = Counter()

    def accept(self, line: str) -> str | None:
        """Zwraca oczyszczoną linię albo None, gdy linię trzeba odrzucić.

        Przykład: patrz opis klasy
        """
        cleaned = _clean_subtitle_line(line)
        if cleaned == "":
            self.counts["pusta"] += 1
            return None
        if _ALLOWED_TEXT.fullmatch(cleaned) is None:
            self.counts["znaki"] += 1
            return None

        words = _WORD_RE.findall(cleaned)
        if len(words) < self._min_words:
            self.counts["krotka"] += 1
            return None
        if cleaned.isupper():
            self.counts["wielkie"] += 1
            return None

        lowered = cleaned.lower()
        if _PROFANITY_STEMS_RE.search(lowered) or _PROFANITY_WORDS_RE.search(lowered):
            self.counts["przeklenstwa"] += 1
            return None
        if _CREDIT_RE.search(lowered):
            self.counts["znaczniki"] += 1
            return None
        if _ENGLISH_RE.search(lowered):
            self.counts["angielski"] += 1
            return None
        if lowered in self._recent:
            self.counts["powtorka"] += 1
            return None

        short = len(words) <= 5
        if short and self._short_seen[lowered] >= self._max_repeats:
            self.counts["powtorka"] += 1
            return None

        if short:
            self._short_seen[lowered] += 1
        self._recent.append(lowered)
        self.counts["przyjete"] += 1
        return cleaned


def remote_size(url: str, head: Callable[[str], int] | None = None) -> int:
    """Zwraca rozmiar pliku pod adresem w bajtach (zapytanie HEAD, bez pobierania treści).

    Przykład: head=lambda url: 5000 -> 5000
    Przykład: head zwraca 0 (serwer nie podał rozmiaru) albo liczbę ujemną -> ValueError
    head=None -> używa _head_url.
    """
    measure = head if head is not None else _head_url
    size = measure(url)
    if size <= 0:
        raise ValueError(f"Serwer nie podał rozmiaru pliku: {url}")
    return size


def download_prefix(
    url: str,
    dest: Path,
    max_bytes: int,
    opener: Callable[[str], IO[bytes]] | None = None,
    chunk_size: int = 1_048_576,
) -> int:
    """Zapisuje do dest co najwyżej max_bytes początkowych bajtów odpowiedzi i zwraca ich liczbę.

    Przykład: strumień z 10 bajtami, max_bytes=6, chunk_size=4 -> dest ma 6 pierwszych bajtów,
        zwraca 6 (nie czyta więcej niż 6)
    Przykład: strumień z 10 bajtami, max_bytes=100 -> dest ma 10 bajtów, zwraca 10
    Przykład: dest już istnieje -> zwraca jego rozmiar, opener niewywołany
    Przykład: max_bytes=0 -> ValueError
    opener=None -> używa _open_url.
    """
    if max_bytes <= 0:
        raise ValueError(f"Limit bajtów musi być dodatni: {max_bytes}")
    if dest.exists():
        return dest.stat().st_size

    open_stream = opener if opener is not None else _open_url
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    stream = open_stream(url)
    written = 0
    try:
        with part.open("wb") as out:
            while written < max_bytes:
                chunk = stream.read(min(chunk_size, max_bytes - written))
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    finally:
        stream.close()

    part.replace(dest)
    return written


def iter_gzip_lines(path: Path, chunk_size: int = 1_048_576) -> Iterator[str]:
    """Zwraca kolejne linie tekstu UTF-8 z pliku gzip, także ucięcie w środku strumienia.

    Przykład: gzip z "ala\\nma kota\\n" -> "ala", "ma kota"
    Przykład: gzip z "ala\\nma" (bez końcowego \\n, strumień kompletny) -> "ala", "ma"
    Przykład: gzip ucięty w środku danych -> tylko pełne linie, niedokończony koniec pominięty
    Przykład: gzip z "a\\r\\nb\\n" -> "a", "b"
    Przykład: pusty plik (0 bajtów) -> brak linii
    Przykład: plik, który nie jest gzip -> zlib.error
    """
    inflater = zlib.decompressobj(wbits=zlib.MAX_WBITS | 16)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    with path.open("rb") as handle:
        while not inflater.eof:
            block = handle.read(chunk_size)
            if not block:
                break
            pending += decoder.decode(inflater.decompress(block))
            *lines, pending = pending.split("\n")
            for line in lines:
                yield line.removesuffix("\r")
    if inflater.eof:
        pending += decoder.decode(b"", final=True)
        if pending:
            yield pending.removesuffix("\r")


@dataclass
class PretrainReport:
    """Wynik budowy korpusu treningu wstępnego.

    Przykład: PretrainReport() -> pola liczbowe 0, listy i słownik puste, target_reached=False
    """

    train_books: list[str] = field(default_factory=list)
    val_books: list[str] = field(default_factory=list)
    lines_read: int = 0
    train_lines: int = 0
    val_lines: int = 0
    subtitle_bytes: int = 0
    train_bytes: int = 0
    val_bytes: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    target_reached: bool = False


def build_pretrain_corpus(
    gz_path: Path,
    clean_dir: Path,
    out_dir: Path,
    target_bytes: int,
    *,
    block_lines: int = 20_000,
    val_every: int = 50,
    book_val_fraction: float = 0.1,
    seed: int = 0,
) -> PretrainReport:
    """Zapisuje train.txt, val.txt i report.json: książki, potem przefiltrowane linie napisów.

    Przykład: clean_dir z a.txt ("Ala ma kota.") i b.txt ("Ola ma psa."), gz z liniami
        "Dzien dobry panu.", "- Dzien dobry.", "Nie.", "Jak sie masz?", "Dobrze, dziekuje.",
        "Co slychac?", target_bytes=1000, block_lines=2, val_every=2, book_val_fraction=0.5,
        seed=0 -> train_books=["b"], val_books=["a"], lines_read=6, train_lines=3, val_lines=2,
        rejected={"krotka": 1}, target_reached=False,
        train.txt == "Ola ma psa.\\n\\nDzien dobry panu.\\nDzien dobry.\\nCo slychac?\\n",
        val.txt == "Ala ma kota.\\n\\nJak sie masz?\\nDobrze, dziekuje.\\n"
    Przykład: target_bytes=0, block_lines=0 albo val_every=1 -> ValueError
    Przykład: clean_dir z jedną książką -> ValueError (z split_books)
    """
    if target_bytes <= 0 or block_lines <= 0 or val_every < 2:
        raise ValueError(
            f"Niepoprawne parametry: target_bytes={target_bytes}, "
            f"block_lines={block_lines}, val_every={val_every}"
        )

    sizes = {path.stem: path.stat().st_size for path in clean_dir.glob("*.txt")}
    train_books, val_books = split_books(sizes, book_val_fraction, seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    report = PretrainReport(train_books=train_books, val_books=val_books)
    filtr = SubtitleFilter()
    train_path = out_dir / "train.txt"
    val_path = out_dir / "val.txt"
    with (
        train_path.open("w", encoding="utf-8", newline="\n") as train_file,
        val_path.open("w", encoding="utf-8", newline="\n") as val_file,
    ):
        train_file.write(_join_books(clean_dir, train_books) + "\n")
        val_file.write(_join_books(clean_dir, val_books) + "\n")

        lines = iter_gzip_lines(gz_path)
        try:
            for raw_line in lines:
                report.lines_read += 1
                cleaned = filtr.accept(raw_line)
                if cleaned is None:
                    continue
                block = (report.train_lines + report.val_lines) // block_lines
                if block % val_every == val_every - 1:
                    val_file.write(cleaned + "\n")
                    report.val_lines += 1
                else:
                    train_file.write(cleaned + "\n")
                    report.train_lines += 1
                report.subtitle_bytes += len(cleaned.encode("utf-8")) + 1
                if report.subtitle_bytes >= target_bytes:
                    report.target_reached = True
                    break
        finally:
            lines.close()

    report.rejected = {
        key: count for key, count in sorted(filtr.counts.items()) if key != "przyjete"
    }
    report.train_bytes = train_path.stat().st_size
    report.val_bytes = val_path.stat().st_size

    payload: dict[str, object] = asdict(report)
    payload["params"] = {
        "target_bytes": target_bytes,
        "block_lines": block_lines,
        "val_every": val_every,
        "book_val_fraction": book_val_fraction,
        "seed": seed,
    }
    _write_json(out_dir / "report.json", payload)
    return report


def split_dialog_blocks(text: str) -> list[list[str]]:
    """Dzieli surowy plik rozmów na bloki linii według linii "---".

    Przykład: "A: Hej.\\nB: Cześć.\\n---\\nA: Co?\\nB: Nic.\\n---\\n"
        -> [["A: Hej.", "B: Cześć."], ["A: Co?", "B: Nic."]]
    Przykład: "A:  Hej.  \\r\\nB:\\tOk." -> [["A: Hej.", "B: Ok."]]
    Przykład: "\\n\\nA: x\\n---\\n\\n" -> [["A: x"]]
    Przykład: "A: a\\n\\nB: b" -> [["A: a", "", "B: b"]] (pusta linia w środku zostaje)
    Przykład: "" albo "---\\n---\\n" -> []
    """
    cleaned = text.replace("﻿", "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = unicodedata.normalize("NFC", cleaned)

    blocks: list[list[str]] = []
    current: list[str] = []
    for raw_line in cleaned.split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if line == "---":
            blocks.append(current)
            current = []
        else:
            current.append(line)
    blocks.append(current)

    result: list[list[str]] = []
    for block in blocks:
        start = 0
        end = len(block)
        while start < end and block[start] == "":
            start += 1
        while end > start and block[end - 1] == "":
            end -= 1
        trimmed = block[start:end]
        if trimmed:
            result.append(trimmed)
    return result


def validate_dialog(lines: list[str], min_lines: int = 4, max_lines: int = 16) -> str | None:
    """Zwraca None dla poprawnej rozmowy, a dla błędnej kod pierwszego znalezionego powodu.

    Przykład: ["A: Hej.", "B: Cześć.", "A: Co tam?", "B: Dobrze."] -> None
    Przykład: [] -> "pusta"
    Przykład: ["A: a", "", "A: b", "B: c"] -> "pusta-linia"
    Przykład: ["A: a", "B:b", "A: c", "B: d"] -> "zla-linia"
    Przykład: ["A: a", "B: ", "A: c", "B: d"] -> "pusta-wypowiedz"
    Przykład: ["A: a", "B: *b*", "A: c", "B: d"] -> "znaki"
    Przykład: ["A: a", "B: 1. Idę.", "A: c", "B: d"] -> "lista"
    Przykład: ["A: a", "B: Oczywiście! Oto plan.", "A: c", "B: d"] -> "asystent"
    Przykład: ["B: a", "A: b", "A: c", "B: d"] -> "start"
    Przykład: ["A: a", "B: b", "B: c", "A: d"] -> "kolejnosc"
    Przykład: ["A: a", "B: b", "A: c", "B: d", "A: e"] -> "koniec"
    Przykład: ["A: a", "B: b"] -> "dlugosc" (2 linie < min_lines=4)
    """
    if not lines:
        return "pusta"
    if any(line == "" for line in lines):
        return "pusta-linia"
    if any(not line.startswith(("A: ", "B: ")) for line in lines):
        return "zla-linia"

    texts = [line[3:] for line in lines]
    if any(text.strip() == "" for text in texts):
        return "pusta-wypowiedz"
    if any(_ALLOWED_TEXT.fullmatch(text) is None for text in texts):
        return "znaki"
    if any(_DIALOG_LIST_RE.match(text) for text in texts):
        return "lista"
    for line in lines:
        if line.startswith("B: ") and any(
            phrase in line[3:].lower() for phrase in ASSISTANT_PHRASES
        ):
            return "asystent"
    if not lines[0].startswith("A: "):
        return "start"
    for index in range(1, len(lines)):
        if lines[index - 1][0] == lines[index][0]:
            return "kolejnosc"
    if not lines[-1].startswith("B: "):
        return "koniec"
    if not min_lines <= len(lines) <= max_lines:
        return "dlugosc"
    return None


@dataclass
class DialogReport:
    """Wynik wczytania rozmów: liczby i powody odrzuceń z identyfikatorami źródeł.

    Przykład: DialogReport() -> files=0, found=0, accepted=0, rejected={}, duplicates={}
    """

    files: int = 0
    found: int = 0
    accepted: int = 0
    rejected: dict[str, str] = field(default_factory=dict)
    duplicates: dict[str, str] = field(default_factory=dict)


def load_dialogs(
    src_dir: Path, min_lines: int = 4, max_lines: int = 16
) -> tuple[list[list[str]], DialogReport]:
    """Wczytuje rozmowy z src_dir/*.txt, odrzuca błędne i duplikaty, zwraca je z raportem.

    Przykład: 001.txt z trzema blokami: poprawna rozmowa, rozmowa z linią "C: x", ta sama
        poprawna rozmowa pisana innymi wielkimi literami -> (1 rozmowa, DialogReport(files=1,
        found=3, accepted=1, rejected={"001.txt#2": "zla-linia"},
        duplicates={"001.txt#3": "001.txt#1"}))
    Przykład: katalog bez plików *.txt -> ([], DialogReport())
    Przykład: katalog nie istnieje -> FileNotFoundError
    """
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Brak katalogu z rozmowami: {src_dir}")

    files = sorted(
        (path for path in src_dir.glob("*.txt") if path.is_file()),
        key=lambda path: path.name,
    )
    report = DialogReport(files=len(files))
    seen: dict[str, str] = {}
    dialogs: list[list[str]] = []
    for path in files:
        blocks = split_dialog_blocks(load_text(path))
        for number, block in enumerate(blocks, start=1):
            source = f"{path.name}#{number}"
            report.found += 1
            reason = validate_dialog(block, min_lines, max_lines)
            if reason is not None:
                report.rejected[source] = reason
                continue
            key = _dialog_key(block)
            if key in seen:
                report.duplicates[source] = seen[key]
                continue
            seen[key] = source
            dialogs.append(block)
            report.accepted += 1
    return dialogs, report


def split_dialogs(
    dialogs: list[list[str]], val_fraction: float = 0.05, seed: int = 0
) -> tuple[list[list[str]], list[list[str]]]:
    """Dzieli rozmowy na (train, val), zachowując w każdej części kolejność wejściową.

    Przykład: 4 rozmowy tej samej długości d0..d3, val_fraction=0.25, seed=0
        -> ([d0, d1, d3], [d2])
    Przykład: jedna rozmowa -> ValueError (z split_books)
    """
    sizes = {
        f"{index:06d}": sum(len(line) + 1 for line in dialog)
        for index, dialog in enumerate(dialogs)
    }
    train_keys, val_keys = split_books(sizes, val_fraction, seed)
    train_set = set(train_keys)
    val_set = set(val_keys)
    train = [dialog for index, dialog in enumerate(dialogs) if f"{index:06d}" in train_set]
    val = [dialog for index, dialog in enumerate(dialogs) if f"{index:06d}" in val_set]
    return train, val


def format_dialog_stream(dialogs: list[list[str]]) -> str:
    """Składa rozmowy w strumień: linie wypowiedzi, rozmowy rozdzielone jedną pustą linią.

    Przykład: [["A: Hej.", "B: Cześć."], ["A: Co?", "B: Nic."]]
        -> "A: Hej.\\nB: Cześć.\\n\\nA: Co?\\nB: Nic.\\n"
    Przykład: [["A: Hej.", "B: Cześć."]] -> "A: Hej.\\nB: Cześć.\\n"
    Przykład: [] -> ""
    """
    return "\n".join("\n".join(dialog) + "\n" for dialog in dialogs)


@dataclass
class DialogCorpusResult:
    """Wynik budowy korpusu rozmów: liczby rozmów w częściach, ścieżki plików i raport.

    Przykład: DialogCorpusResult(train=3, val=1, train_path=p, val_path=q, report=r)
    """

    train: int
    val: int
    train_path: Path
    val_path: Path
    report: DialogReport


def build_dialog_corpus(
    src_dir: Path,
    out_dir: Path,
    val_fraction: float = 0.05,
    seed: int = 0,
    min_lines: int = 4,
    max_lines: int = 16,
) -> DialogCorpusResult:
    """Wczytuje rozmowy, dzieli je i zapisuje train.txt, val.txt i report.json w out_dir.

    Przykład: src_dir z 001.txt z 4 różnymi poprawnymi rozmowami o tej samej długości,
        val_fraction=0.25, seed=0 -> train=3, val=1, pliki train.txt i val.txt w out_dir
    Przykład: katalog bez poprawnych rozmów -> ValueError (z split_dialogs)
    """
    dialogs, report = load_dialogs(src_dir, min_lines, max_lines)
    train, val = split_dialogs(dialogs, val_fraction, seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.txt"
    val_path = out_dir / "val.txt"
    train_path.write_text(format_dialog_stream(train), encoding="utf-8", newline="\n")
    val_path.write_text(format_dialog_stream(val), encoding="utf-8", newline="\n")

    reasons = dict(sorted(Counter(report.rejected.values()).items()))
    payload: dict[str, object] = {
        "seed": seed,
        "val_fraction": val_fraction,
        "min_lines": min_lines,
        "max_lines": max_lines,
        "files": report.files,
        "found": report.found,
        "accepted": report.accepted,
        "train": len(train),
        "val": len(val),
        "rejected": report.rejected,
        "duplicates": report.duplicates,
        "reasons": reasons,
    }
    _write_json(out_dir / "report.json", payload)
    return DialogCorpusResult(
        train=len(train),
        val=len(val),
        train_path=train_path,
        val_path=val_path,
        report=report,
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


def _clean_subtitle_line(line: str) -> str:
    """Normalizuje linię: NFC, bez wiodących myślników dialogowych, jedna spacja, bez brzegów.

    Przykład: "  - Co tam?  " -> "Co tam?"
    Przykład: "-- – Nie wiem  co robić." -> "Nie wiem co robić."
    Przykład: "Ala\\tma   kota" -> "Ala ma kota"
    Przykład: "Cafe\\u0301" -> "Café" (NFC)
    Przykład: "" -> ""
    """
    text = unicodedata.normalize("NFC", line).strip()
    text = _LEADING_DASHES.sub("", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _open_url(url: str) -> IO[bytes]:
    """Otwiera adres HTTP(S) i zwraca strumień bajtów do czytania metodą read(n).

    Przykład: _open_url(OPUS_SUBTITLES_PL_URL).read(4) -> b"\\x1f\\x8b\\x08\\x00"
    """
    request = urllib.request.Request(url, headers={"User-Agent": "nanowriter/0.1"})
    return urllib.request.urlopen(request, timeout=60)


def _head_url(url: str) -> int:
    """Wysyła zapytanie HEAD i zwraca wartość nagłówka Content-Length jako liczbę bajtów.

    Przykład: _head_url(OPUS_SUBTITLES_PL_URL) -> 1931316112
    Przykład: brak nagłówka Content-Length -> 0
    """
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "nanowriter/0.1"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        length = response.headers.get("Content-Length")
    return 0 if length is None else int(length)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Zapisuje payload jako JSON (UTF-8, bez ucieczki polskich znaków, wcięcie 2, końcowy \\n).

    Przykład: {"a": "ą"} -> plik o treści '{\\n  "a": "ą"\\n}\\n'
    """
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _dialog_key(lines: list[str]) -> str:
    """Zwraca klucz do wykrywania duplikatów: NFC, małe litery (casefold), jedna spacja.

    Przykład: ["A: Hej.", "B: Cześć."] -> "a: hej. | b: cześć."
    Przykład: ["A:  HEJ. ", "B: cześć."] -> "a: hej. | b: cześć."
    """
    parts: list[str] = []
    for line in lines:
        folded = unicodedata.normalize("NFC", line).casefold()
        parts.append(re.sub(r"\s+", " ", folded).strip())
    return " | ".join(parts)
