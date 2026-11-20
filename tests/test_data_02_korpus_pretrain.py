import dataclasses
import gzip
import io
import json
import re
import zlib
from collections.abc import Callable
from pathlib import Path

import pytest

import nanowriter.data as data
from nanowriter.data import (
    OPUS_SUBTITLES_PL_URL,
    PROFANITY_STEMS,
    PretrainReport,
    build_pretrain_corpus,
    download_prefix,
    iter_gzip_lines,
    remote_size,
)

URL = "http://example.invalid/pl.txt.gz"

PREFIKS_TRAIN = "Ola ma psa.\n\n"
PREFIKS_VAL = "Ala ma kota.\n\n"

PRZYKLAD_LINIE = [
    "Dzien dobry panu.",
    "- Dzien dobry.",
    "Nie.",
    "Jak sie masz?",
    "Dobrze, dziekuje.",
    "Co slychac?",
]

ODRZUCONE_LINIE = [
    "- Dzień dobry, panie Kowalski.",
    "UWAŻAJ NA SIEBIE!",
    "♪ La la la ♪",
    "Co za kurwa dzień.",
    "Nie.",
    "Why am I so happy?",
    "Napisy: michnik",
    "Jak się masz?",
    "Jak się masz?",
    "Mam się dobrze, dziękuję.",
    "",
    "Zażółć gęślą jaźń.",
]

LINIE_NUMEROWANE = [f"Linia numer {n:03d} tutaj." for n in range(10)]


# ----------------------------------------------------------------------------
# Pomocnicze funkcje testów
# ----------------------------------------------------------------------------


def _gz_z_linii(tmp_path: Path, linie: list[str], name: str = "pl.txt.gz") -> Path:
    path = tmp_path / name
    path.write_bytes(gzip.compress("\n".join(linie).encode("utf-8")))
    return path


def _gz_z_tekstu(tmp_path: Path, tekst: str, name: str = "pl.txt.gz") -> Path:
    path = tmp_path / name
    path.write_bytes(gzip.compress(tekst.encode("utf-8")))
    return path


def _zestaw_ab(tmp_path: Path) -> Path:
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    (clean_dir / "a.txt").write_text("Ala ma kota.", encoding="utf-8")
    (clean_dir / "b.txt").write_text("Ola ma psa.", encoding="utf-8")
    return clean_dir


def _wywolanie_ab(
    gz: Path,
    clean_dir: Path,
    out_dir: Path,
    target_bytes: int,
    **kwargs: object,
) -> PretrainReport:
    params: dict[str, object] = {
        "block_lines": 2,
        "val_every": 2,
        "book_val_fraction": 0.5,
        "seed": 0,
    }
    params.update(kwargs)
    return build_pretrain_corpus(gz, clean_dir, out_dir, target_bytes, **params)  # type: ignore[arg-type]


def _train(out_dir: Path) -> str:
    return (out_dir / "train.txt").read_bytes().decode("utf-8")


def _val(out_dir: Path) -> str:
    return (out_dir / "val.txt").read_bytes().decode("utf-8")


def _linie_napisow(tekst: str, prefiks: str) -> list[str]:
    assert tekst.startswith(prefiks)
    reszta = tekst[len(prefiks) :]
    if not reszta:
        return []
    assert reszta.endswith("\n")
    return reszta.removesuffix("\n").split("\n")


def _opener(stream: io.BytesIO) -> Callable[[str], io.BytesIO]:
    return lambda url: stream


def _przyklad_z_opisu(tmp_path: Path) -> tuple[PretrainReport, Path, Path]:
    clean_dir = _zestaw_ab(tmp_path)
    gz = _gz_z_linii(tmp_path, PRZYKLAD_LINIE)
    out_dir = tmp_path / "out"
    raport = _wywolanie_ab(gz, clean_dir, out_dir, 1000)
    return raport, out_dir, gz


class _StreamZLogiemOdczytow(io.BytesIO):
    def __init__(self, initial: bytes) -> None:
        super().__init__(initial)
        self.odczyty: list[int] = []

    def read(self, size: int | None = -1) -> bytes:
        self.odczyty.append(-1 if size is None else size)
        return super().read(size)


class _StreamZBledem(io.BytesIO):
    def __init__(self) -> None:
        super().__init__(b"")
        self.licznik = 0

    def read(self, size: int | None = -1) -> bytes:
        self.licznik += 1
        if self.licznik == 1:
            return b"abcd"
        raise OSError("zerwane")


# ----------------------------------------------------------------------------
# Stała i dataclass
# ----------------------------------------------------------------------------


def test_stala_url_opus():
    assert (
        OPUS_SUBTITLES_PL_URL
        == "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/pl.txt.gz"
    )


def test_pretrainreport_wartosci_domyslne():
    raport = PretrainReport()

    assert dataclasses.is_dataclass(raport)
    assert raport.train_books == []
    assert raport.val_books == []
    assert raport.lines_read == 0
    assert raport.train_lines == 0
    assert raport.val_lines == 0
    assert raport.subtitle_bytes == 0
    assert raport.train_bytes == 0
    assert raport.val_bytes == 0
    assert raport.rejected == {}
    assert raport.target_reached is False
    assert [f.name for f in dataclasses.fields(raport)] == [
        "train_books",
        "val_books",
        "lines_read",
        "train_lines",
        "val_lines",
        "subtitle_bytes",
        "train_bytes",
        "val_bytes",
        "rejected",
        "target_reached",
    ]


def test_pretrainreport_nie_dzieli_list_miedzy_obiektami():
    r1 = PretrainReport()
    r2 = PretrainReport()

    r1.train_books.append("x")
    r1.rejected["y"] = 1

    assert r2.train_books == []
    assert r2.rejected == {}


# ----------------------------------------------------------------------------
# remote_size
# ----------------------------------------------------------------------------


def test_remote_size_zwraca_wynik_head():
    assert remote_size(URL, head=lambda url: 5000) == 5000


def test_remote_size_przekazuje_url_do_head():
    argumenty: list[str] = []

    def head(url: str) -> int:
        argumenty.append(url)
        return 42

    assert remote_size("http://x/y", head=head) == 42
    assert argumenty == ["http://x/y"]


def test_remote_size_zero_i_ujemny():
    with pytest.raises(ValueError):
        remote_size(URL, head=lambda url: 0)
    with pytest.raises(ValueError):
        remote_size(URL, head=lambda url: -1)


def test_remote_size_domyslny_head_to_head_url(monkeypatch):
    argumenty: list[str] = []

    def fake(url: str) -> int:
        argumenty.append(url)
        return 1931316112

    monkeypatch.setattr(data, "_head_url", fake)

    assert remote_size(URL) == 1931316112
    assert argumenty == [URL]


# ----------------------------------------------------------------------------
# download_prefix
# ----------------------------------------------------------------------------


def test_download_prefix_obcina_do_max_bytes(tmp_path):
    dest = tmp_path / "x.gz"
    opener = _opener(io.BytesIO(b"0123456789"))

    wynik = download_prefix(URL, dest, 6, opener=opener, chunk_size=4)

    assert wynik == 6
    assert dest.read_bytes() == b"012345"


def test_download_prefix_nie_czyta_wiecej_niz_limit(tmp_path):
    stream = _StreamZLogiemOdczytow(b"0123456789")

    wynik = download_prefix(URL, tmp_path / "x.gz", 6, opener=_opener(stream), chunk_size=4)

    assert wynik == 6
    assert stream.odczyty == [4, 2]

    stream2 = _StreamZLogiemOdczytow(b"0123456789")

    wynik2 = download_prefix(URL, tmp_path / "y.gz", 10, opener=_opener(stream2), chunk_size=4)

    assert wynik2 == 10
    assert stream2.odczyty == [4, 4, 2]


def test_download_prefix_strumien_krotszy_niz_limit(tmp_path):
    dest = tmp_path / "x.gz"
    opener = _opener(io.BytesIO(b"0123456789"))

    wynik = download_prefix(URL, dest, 100, opener=opener, chunk_size=4)

    assert wynik == 10
    assert dest.read_bytes() == b"0123456789"


def test_download_prefix_pusty_strumien(tmp_path):
    dest = tmp_path / "x.gz"

    wynik = download_prefix(URL, dest, 100, opener=_opener(io.BytesIO(b"")))

    assert wynik == 0
    assert dest.exists()
    assert dest.stat().st_size == 0


def test_download_prefix_domyslna_porcja(tmp_path):
    dest = tmp_path / "x.gz"
    opener = _opener(io.BytesIO(b"x" * 1000))

    wynik = download_prefix(URL, dest, 5000, opener=opener)

    assert wynik == 1000
    assert dest.read_bytes() == b"x" * 1000


def test_download_prefix_dest_istnieje(tmp_path):
    dest = tmp_path / "x.gz"
    dest.write_bytes(b"stare")
    wywolania: list[str] = []

    def opener(url: str) -> io.BytesIO:
        wywolania.append(url)
        return io.BytesIO(b"nowe dane")

    wynik = download_prefix(URL, dest, 100, opener=opener)

    assert wynik == 5
    assert len(wywolania) == 0
    assert dest.read_bytes() == b"stare"


@pytest.mark.parametrize("limit", [0, -5])
def test_download_prefix_zly_limit(tmp_path, limit):
    dest = tmp_path / "x.gz"
    wywolania: list[str] = []

    def opener(url: str) -> io.BytesIO:
        wywolania.append(url)
        return io.BytesIO(b"0123456789")

    with pytest.raises(ValueError):
        download_prefix(URL, dest, limit, opener=opener)

    assert not dest.exists()
    assert len(wywolania) == 0


def test_download_prefix_zly_limit_przed_cache(tmp_path):
    dest = tmp_path / "x.gz"
    dest.write_bytes(b"stare")

    with pytest.raises(ValueError):
        download_prefix(URL, dest, 0, opener=_opener(io.BytesIO(b"0123456789")))


def test_download_prefix_tworzy_katalog_i_przekazuje_url(tmp_path):
    dest = tmp_path / "a" / "b" / "x.gz"
    argumenty: list[str] = []

    def opener(url: str) -> io.BytesIO:
        argumenty.append(url)
        return io.BytesIO(b"abc")

    wynik = download_prefix("http://x/y", dest, 10, opener=opener)

    assert wynik == 3
    assert dest.read_bytes() == b"abc"
    assert argumenty == ["http://x/y"]


def test_download_prefix_bez_pliku_part(tmp_path):
    opener = _opener(io.BytesIO(b"0123456789"))

    download_prefix(URL, tmp_path / "x.gz", 6, opener=opener)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.gz"]


def test_download_prefix_zamyka_strumien(tmp_path):
    stream = io.BytesIO(b"0123456789")

    download_prefix(URL, tmp_path / "x.gz", 6, opener=_opener(stream))

    assert stream.closed is True


def test_download_prefix_blad_w_trakcie(tmp_path):
    dest = tmp_path / "x.gz"
    stream = _StreamZBledem()

    with pytest.raises(OSError):
        download_prefix(URL, dest, 100, opener=_opener(stream), chunk_size=4)

    assert dest.exists() is False
    assert stream.closed is True


def test_download_prefix_domyslny_opener_to_open_url(tmp_path, monkeypatch):
    dest = tmp_path / "x.gz"
    argumenty: list[str] = []

    def fake(url: str) -> io.BytesIO:
        argumenty.append(url)
        return io.BytesIO(b"0123456789")

    monkeypatch.setattr(data, "_open_url", fake)

    wynik = download_prefix("http://x/y", dest, 6)

    assert wynik == 6
    assert dest.read_bytes() == b"012345"
    assert argumenty == ["http://x/y"]


# ----------------------------------------------------------------------------
# iter_gzip_lines
# ----------------------------------------------------------------------------


def test_iter_gzip_lines_przyklad_podstawowy(tmp_path):
    path = _gz_z_tekstu(tmp_path, "ala\nma kota\n")

    assert list(iter_gzip_lines(path)) == ["ala", "ma kota"]


def test_iter_gzip_lines_zwraca_iterator(tmp_path):
    path = _gz_z_tekstu(tmp_path, "ala\n")

    wynik = iter_gzip_lines(path)

    assert iter(wynik) is wynik
    assert next(wynik) == "ala"


def test_iter_gzip_lines_bez_koncowego_newline(tmp_path):
    path = _gz_z_tekstu(tmp_path, "ala\nma")

    assert list(iter_gzip_lines(path)) == ["ala", "ma"]


def test_iter_gzip_lines_crlf(tmp_path):
    path1 = _gz_z_tekstu(tmp_path, "a\r\nb\n", name="1.gz")
    path2 = _gz_z_tekstu(tmp_path, "a\r\nb\r", name="2.gz")

    assert list(iter_gzip_lines(path1)) == ["a", "b"]
    assert list(iter_gzip_lines(path2)) == ["a", "b"]


def test_iter_gzip_lines_puste_linie(tmp_path):
    path1 = _gz_z_tekstu(tmp_path, "a\n\nb\n", name="1.gz")
    path2 = _gz_z_tekstu(tmp_path, "a\n\n", name="2.gz")
    path3 = _gz_z_tekstu(tmp_path, "\n", name="3.gz")

    assert list(iter_gzip_lines(path1)) == ["a", "", "b"]
    assert list(iter_gzip_lines(path2)) == ["a", ""]
    assert list(iter_gzip_lines(path3)) == [""]


def test_iter_gzip_lines_pusty_gzip_i_pusty_plik(tmp_path):
    pusty_gzip = tmp_path / "pusty.gz"
    pusty_gzip.write_bytes(gzip.compress(b""))
    pusty_plik = tmp_path / "zero.gz"
    pusty_plik.write_bytes(b"")

    assert list(iter_gzip_lines(pusty_gzip)) == []
    assert list(iter_gzip_lines(pusty_plik)) == []


def _ucieta_paczka(tmp_path: Path) -> tuple[Path, Path]:
    tekst = "".join(f"linia numer {i}\n" for i in range(5000))
    pelny = gzip.compress(tekst.encode("utf-8"))
    ucieta_path = tmp_path / "ucieta.gz"
    ucieta_path.write_bytes(pelny[: len(pelny) // 2])
    pelny_path = tmp_path / "pelna.gz"
    pelny_path.write_bytes(pelny)
    return ucieta_path, pelny_path


def test_iter_gzip_lines_ucietyplik(tmp_path):
    ucieta_path, _ = _ucieta_paczka(tmp_path)

    lines = list(iter_gzip_lines(ucieta_path))

    assert 0 < len(lines) < 5000
    assert lines == [f"linia numer {i}" for i in range(len(lines))]


def test_iter_gzip_lines_ucietyplik_male_porcje(tmp_path):
    ucieta_path, _ = _ucieta_paczka(tmp_path)

    wyniki = [
        list(iter_gzip_lines(ucieta_path, chunk_size=64)),
        list(iter_gzip_lines(ucieta_path, chunk_size=7)),
        list(iter_gzip_lines(ucieta_path, chunk_size=1000)),
    ]

    assert wyniki[0] == wyniki[1] == wyniki[2]
    for lista in wyniki:
        assert len(lista) > 0
        assert lista == [f"linia numer {i}" for i in range(len(lista))]


def test_iter_gzip_lines_ucietyplik_to_prefiks_pelnego(tmp_path):
    ucieta_path, pelny_path = _ucieta_paczka(tmp_path)

    pelne = list(iter_gzip_lines(pelny_path))
    lines = list(iter_gzip_lines(ucieta_path))

    assert pelne == [f"linia numer {i}" for i in range(5000)]
    assert lines == pelne[: len(lines)]


@pytest.mark.parametrize("compresslevel", [0, None])
@pytest.mark.parametrize("chunk_size", [1, 3, 7, 1_048_576])
def test_iter_gzip_lines_polskie_znaki_granica_porcji(tmp_path, compresslevel, chunk_size):
    tekst = "zażółć gęślą jaźń\nŁódź ąę\nAla 😀 kot\n"
    path = tmp_path / "pl.gz"
    opcje = {} if compresslevel is None else {"compresslevel": compresslevel}
    path.write_bytes(gzip.compress(tekst.encode("utf-8"), **opcje))

    assert list(iter_gzip_lines(path, chunk_size=chunk_size)) == [
        "zażółć gęślą jaźń",
        "Łódź ąę",
        "Ala 😀 kot",
    ]


def test_iter_gzip_lines_bledne_utf8(tmp_path):
    path = tmp_path / "zle.gz"
    path.write_bytes(gzip.compress(b"ala\n\xff\xfe ok\nkot\n"))

    assert list(iter_gzip_lines(path)) == ["ala", "�� ok", "kot"]


def test_iter_gzip_lines_ignoruje_dane_po_koncu_strumienia(tmp_path):
    path = tmp_path / "smiec.gz"
    path.write_bytes(gzip.compress(b"ala\nkot\n") + b"to jest smiec")

    assert list(iter_gzip_lines(path)) == ["ala", "kot"]


def test_iter_gzip_lines_nie_gzip(tmp_path):
    path = tmp_path / "zwykly.gz"
    path.write_bytes(b"to nie jest gzip, tylko zwykly tekst\n")

    with pytest.raises(zlib.error):
        list(iter_gzip_lines(path))


def test_iter_gzip_lines_brak_cr_i_lf_na_koncu(tmp_path):
    path = _gz_z_tekstu(tmp_path, "a\r\nb\nc\r\n\r\nd")

    wynik = list(iter_gzip_lines(path))

    for linia in wynik:
        assert "\n" not in linia
        assert not linia.endswith("\r")
    assert wynik == ["a", "b", "c", "", "d"]


def test_iter_gzip_lines_zamkniecie_generatora_zwalnia_plik(tmp_path):
    path = _gz_z_tekstu(tmp_path, "a\nb\nc\nd\n")

    gen = iter_gzip_lines(path)
    next(gen)
    gen.close()
    path.unlink()

    assert path.exists() is False


# ----------------------------------------------------------------------------
# build_pretrain_corpus
# ----------------------------------------------------------------------------


def test_build_przyklad_z_opisu(tmp_path):
    raport, out_dir, _ = _przyklad_z_opisu(tmp_path)

    train_text = _train(out_dir)
    val_text = _val(out_dir)

    assert raport.train_books == ["b"]
    assert raport.val_books == ["a"]
    assert raport.lines_read == 6
    assert raport.train_lines == 3
    assert raport.val_lines == 2
    assert raport.rejected == {"krotka": 1}
    assert raport.target_reached is False
    assert raport.subtitle_bytes == 75
    assert train_text == "Ola ma psa.\n\nDzien dobry panu.\nDzien dobry.\nCo slychac?\n"
    assert val_text == "Ala ma kota.\n\nJak sie masz?\nDobrze, dziekuje.\n"
    assert raport.train_bytes == len(train_text.encode("utf-8"))
    assert raport.val_bytes == len(val_text.encode("utf-8"))


def test_build_zwraca_pretrainreport(tmp_path):
    raport, _, _ = _przyklad_z_opisu(tmp_path)

    assert isinstance(raport, PretrainReport)


def test_build_bez_cr_w_plikach(tmp_path):
    _, out_dir, _ = _przyklad_z_opisu(tmp_path)

    assert b"\r" not in (out_dir / "train.txt").read_bytes()
    assert b"\r" not in (out_dir / "val.txt").read_bytes()
    assert b"\r" not in (out_dir / "report.json").read_bytes()


def test_build_report_json_zgodny_z_raportem(tmp_path):
    raport, out_dir, _ = _przyklad_z_opisu(tmp_path)

    tekst = (out_dir / "report.json").read_text(encoding="utf-8")
    zawartosc = json.loads(tekst)
    params = zawartosc.pop("params")

    assert zawartosc == dataclasses.asdict(raport)
    assert params == {
        "target_bytes": 1000,
        "block_lines": 2,
        "val_every": 2,
        "book_val_fraction": 0.5,
        "seed": 0,
    }
    assert list(json.loads(tekst)) == [
        "train_books",
        "val_books",
        "lines_read",
        "train_lines",
        "val_lines",
        "subtitle_bytes",
        "train_bytes",
        "val_bytes",
        "rejected",
        "target_reached",
        "params",
    ]
    assert tekst.startswith('{\n  "train_books": [')
    assert tekst.endswith("\n")
    assert not tekst.endswith("\n\n")


def test_build_report_json_polskie_znaki_bez_ucieczki(tmp_path):
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    (clean_dir / "lodz.txt").write_text("Ala ma kota.", encoding="utf-8")
    (clean_dir / "łódź.txt").write_text("Ola ma psa.", encoding="utf-8")
    gz = _gz_z_linii(tmp_path, ["Dzien dobry panu.", "Jak sie masz?"])
    out_dir = tmp_path / "out"

    raport = build_pretrain_corpus(gz, clean_dir, out_dir, 1000, book_val_fraction=0.5)

    tekst = (out_dir / "report.json").read_text(encoding="utf-8")
    assert "łódź" in tekst
    assert "\\u" not in tekst
    assert sorted(raport.train_books + raport.val_books) == sorted(["lodz", "łódź"])


def test_build_params_przy_domyslnych_argumentach(tmp_path):
    clean_dir = _zestaw_ab(tmp_path)
    gz = _gz_z_linii(tmp_path, ["Dzien dobry panu.", "Jak sie masz?"])
    out_dir = tmp_path / "out"

    build_pretrain_corpus(gz, clean_dir, out_dir, 500)

    params = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))["params"]
    assert params == {
        "target_bytes": 500,
        "block_lines": 20000,
        "val_every": 50,
        "book_val_fraction": 0.1,
        "seed": 0,
    }


def test_build_domyslne_bloki_wszystko_w_train(tmp_path):
    clean_dir = _zestaw_ab(tmp_path)
    gz = _gz_z_linii(tmp_path, ["Dzien dobry panu.", "Jak sie masz?", "Co slychac?"])
    out_dir = tmp_path / "out"

    raport = build_pretrain_corpus(gz, clean_dir, out_dir, 1000)

    assert raport.train_lines == 3
    assert raport.val_lines == 0
    assert _val(out_dir) == "Ala ma kota.\n\n"
    assert _train(out_dir) == "Ola ma psa.\n\nDzien dobry panu.\nJak sie masz?\nCo slychac?\n"


def _dwanascie_linii() -> list[str]:
    return [f"Linia numer {n:02d} tutaj." for n in range(12)]


def test_build_bloki_val_every_3(tmp_path):
    clean_dir = _zestaw_ab(tmp_path)
    linie = _dwanascie_linii()
    gz = _gz_z_linii(tmp_path, linie)
    out_dir = tmp_path / "out"

    raport = _wywolanie_ab(gz, clean_dir, out_dir, 100000, block_lines=2, val_every=3)

    train_linie = _linie_napisow(_train(out_dir), PREFIKS_TRAIN)
    assert train_linie == [linie[n] for n in (0, 1, 2, 3, 6, 7, 8, 9)]
    assert _linie_napisow(_val(out_dir), PREFIKS_VAL) == [linie[n] for n in (4, 5, 10, 11)]
    assert raport.train_lines == 8
    assert raport.val_lines == 4
    assert raport.lines_read == 12
    assert raport.target_reached is False


def test_build_bloki_val_every_2_naprzemiennie(tmp_path):
    clean_dir = _zestaw_ab(tmp_path)
    linie = _dwanascie_linii()
    gz = _gz_z_linii(tmp_path, linie)
    out_dir = tmp_path / "out"

    _wywolanie_ab(gz, clean_dir, out_dir, 100000, block_lines=2, val_every=2)

    assert _linie_napisow(_train(out_dir), PREFIKS_TRAIN) == [linie[n] for n in (0, 1, 4, 5, 8, 9)]
    assert _linie_napisow(_val(out_dir), PREFIKS_VAL) == [linie[n] for n in (2, 3, 6, 7, 10, 11)]


def test_build_bloki_nieparzysta_liczba_linii(tmp_path):
    clean_dir = _zestaw_ab(tmp_path)
    linie = _dwanascie_linii()[:7]
    gz = _gz_z_linii(tmp_path, linie)
    out_dir = tmp_path / "out"

    raport = _wywolanie_ab(gz, clean_dir, out_dir, 100000, block_lines=3, val_every=2)

    assert _linie_napisow(_train(out_dir), PREFIKS_TRAIN) == [linie[n] for n in (0, 1, 2, 6)]
    assert _linie_napisow(_val(out_dir), PREFIKS_VAL) == [linie[n] for n in (3, 4, 5)]
    assert raport.train_lines == 4
    assert raport.val_lines == 3


def _cztery_ksiazki(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    tresc = {
        "a": "Ala ma kota.",
        "b": "Ola ma psa.",
        "c": "Ela ma rybę.",
        "d": "Ula ma ptaka.",
    }
    for slug, tekst in tresc.items():
        (clean_dir / f"{slug}.txt").write_text(tekst, encoding="utf-8")
    return clean_dir, tresc


def test_build_rozlacznosc(tmp_path):
    clean_dir, tresc = _cztery_ksiazki(tmp_path)
    gz = _gz_z_linii(tmp_path, _dwanascie_linii())
    out_dir = tmp_path / "out"

    raport = _wywolanie_ab(gz, clean_dir, out_dir, 1_000_000, block_lines=2, val_every=2)

    train_text = _train(out_dir)
    val_text = _val(out_dir)

    assert set(raport.train_books) & set(raport.val_books) == set()
    assert sorted(raport.train_books + raport.val_books) == ["a", "b", "c", "d"]
    assert raport.train_books == sorted(raport.train_books)
    assert raport.val_books == sorted(raport.val_books)
    assert raport.train_books
    assert raport.val_books

    napisy_train = {x for x in train_text.split("\n") if x.startswith("Linia numer")}
    napisy_val = {x for x in val_text.split("\n") if x.startswith("Linia numer")}
    assert napisy_train & napisy_val == set()
    assert len(napisy_train | napisy_val) == 12

    for slug in raport.train_books:
        assert tresc[slug] in train_text
        assert tresc[slug] not in val_text
    for slug in raport.val_books:
        assert tresc[slug] in val_text
        assert tresc[slug] not in train_text


def test_build_prefiks_ksiazek_to_join_posortowanych(tmp_path):
    clean_dir, tresc = _cztery_ksiazki(tmp_path)
    gz = _gz_z_linii(tmp_path, _dwanascie_linii())
    out_dir = tmp_path / "out"

    raport = _wywolanie_ab(gz, clean_dir, out_dir, 1_000_000, block_lines=2, val_every=2)

    train_prefiks = "\n\n".join(tresc[slug] for slug in raport.train_books) + "\n\n"
    val_prefiks = "\n\n".join(tresc[slug] for slug in raport.val_books) + "\n\n"
    train_text = _train(out_dir)
    val_text = _val(out_dir)

    assert train_text.startswith(train_prefiks)
    assert val_text.startswith(val_prefiks)
    assert train_text[len(train_prefiks) :].startswith("Linia numer")
    assert val_text[len(val_prefiks) :].startswith("Linia numer")


def test_build_ignoruje_pliki_inne_niz_txt(tmp_path):
    clean_dir = _zestaw_ab(tmp_path)
    (clean_dir / "notatki.md").write_text("nie ksiazka", encoding="utf-8")
    (clean_dir / "podkatalog").mkdir()
    gz = _gz_z_linii(tmp_path, PRZYKLAD_LINIE)
    out_dir = tmp_path / "out"

    raport = _wywolanie_ab(gz, clean_dir, out_dir, 1000)

    assert raport.train_books == ["b"]
    assert raport.val_books == ["a"]
    assert "nie ksiazka" not in _train(out_dir)
    assert "nie ksiazka" not in _val(out_dir)
    assert "nie ksiazka" not in (out_dir / "report.json").read_text(encoding="utf-8")


def test_build_napisy_nie_lamia_regul_filtra(tmp_path):
    clean_dir = _zestaw_ab(tmp_path)
    linie: list[str] = []
    for i in range(20):
        linie += [
            f"Dzień dobry, panie numer {i}.",
            f"- Tak jest, panie numer {i}.",
            f"UWAŻAJ NA SIEBIE {i}!",
            f"Co za kurwa dzień numer {i}.",
            f"♪ La la la {i} ♪",
            "Nie.",
        ]
    gz = _gz_z_linii(tmp_path, linie)
    out_dir = tmp_path / "out"

    raport = _wywolanie_ab(gz, clean_dir, out_dir, 1_000_000, block_lines=10, val_every=2)

    assert raport.lines_read == 120
    assert raport.train_lines == 20
    assert raport.val_lines == 20
    assert raport.rejected == {"krotka": 20, "przeklenstwa": 20, "wielkie": 20, "znaki": 20}
    assert raport.subtitle_bytes == 1100
    assert raport.target_reached is False
    przyjete_i_odrzucone = raport.train_lines + raport.val_lines + sum(raport.rejected.values())
    assert raport.lines_read == przyjete_i_odrzucone

    train_linie = _linie_napisow(_train(out_dir), PREFIKS_TRAIN)
    val_linie = _linie_napisow(_val(out_dir), PREFIKS_VAL)

    assert len(train_linie) == 20
    assert len(val_linie) == 20
    for linia in train_linie + val_linie:
        assert linia != ""
        assert not linia.startswith("-")
        assert "♪" not in linia
        assert "/" not in linia
        assert "#" not in linia
        assert not linia.isupper()
        male = linia.lower()
        for rdzen in PROFANITY_STEMS:
            assert rdzen not in male
        assert len(linia.split()) >= 2

    def oczekiwane(numery: list[int]) -> list[str]:
        wynik: list[str] = []
        for i in numery:
            wynik += [f"Dzień dobry, panie numer {i}.", f"Tak jest, panie numer {i}."]
        return wynik

    numery_train = list(range(0, 5)) + list(range(10, 15))
    numery_val = list(range(5, 10)) + list(range(15, 20))
    assert sorted(train_linie) == sorted(oczekiwane(numery_train))
    assert sorted(val_linie) == sorted(oczekiwane(numery_val))

    # Numer i wyciągnięty z tekstu linii: żadna linia train nie ma numeru z val i odwrotnie.
    numery_w_train = {int(re.search(r"numer (\d+)\.", x).group(1)) for x in train_linie}
    numery_w_val = {int(re.search(r"numer (\d+)\.", x).group(1)) for x in val_linie}
    assert numery_w_train == set(numery_train)
    assert numery_w_val == set(numery_val)
    assert numery_w_train & numery_w_val == set()


def test_build_linie_odrzucone_licza_sie_do_lines_read(tmp_path):
    clean_dir = _zestaw_ab(tmp_path)
    gz = _gz_z_linii(tmp_path, ODRZUCONE_LINIE)
    out_dir = tmp_path / "out"

    raport = _wywolanie_ab(gz, clean_dir, out_dir, 100000)

    train_text = _train(out_dir)
    val_text = _val(out_dir)

    assert raport.lines_read == 12
    assert raport.train_lines == 2
    assert raport.val_lines == 2
    assert raport.subtitle_bytes == 102
    assert raport.target_reached is False
    assert raport.rejected == {
        "angielski": 1,
        "krotka": 1,
        "powtorka": 1,
        "przeklenstwa": 1,
        "pusta": 1,
        "wielkie": 1,
        "znaczniki": 1,
        "znaki": 1,
    }
    assert list(raport.rejected) == sorted(raport.rejected)
    assert "przyjete" not in raport.rejected
    assert train_text == "Ola ma psa.\n\nDzień dobry, panie Kowalski.\nJak się masz?\n"
    assert val_text == "Ala ma kota.\n\nMam się dobrze, dziękuję.\nZażółć gęślą jaźń.\n"
    assert raport.train_bytes == len(train_text.encode("utf-8"))
    assert raport.val_bytes == len(val_text.encode("utf-8"))


def test_build_target_bytes_przerywa_w_srodku(tmp_path):
    clean_dir = _zestaw_ab(tmp_path)
    gz = _gz_z_linii(tmp_path, LINIE_NUMEROWANE)
    out_dir = tmp_path / "out"

    raport = _wywolanie_ab(gz, clean_dir, out_dir, 70)

    train_text = _train(out_dir)
    val_text = _val(out_dir)

    assert raport.target_reached is True
    assert raport.subtitle_bytes == 92
    assert raport.lines_read == 4
    assert raport.train_lines == 2
    assert raport.val_lines == 2
    assert _linie_napisow(train_text, PREFIKS_TRAIN) == LINIE_NUMEROWANE[0:2]
    assert _linie_napisow(val_text, PREFIKS_VAL) == LINIE_NUMEROWANE[2:4]
    for linia in LINIE_NUMEROWANE[4:]:
        assert linia not in train_text
        assert linia not in val_text


def test_build_target_bytes_granica(tmp_path):
    clean_dir = _zestaw_ab(tmp_path)
    gz = _gz_z_linii(tmp_path, LINIE_NUMEROWANE)

    r69 = _wywolanie_ab(gz, clean_dir, tmp_path / "o69", 69)
    assert r69.target_reached is True
    assert r69.subtitle_bytes == 69
    assert r69.lines_read == 3

    r70 = _wywolanie_ab(gz, clean_dir, tmp_path / "o70", 70)
    assert r70.lines_read == 4

    r1000 = _wywolanie_ab(gz, clean_dir, tmp_path / "o1000", 1000)
    assert r1000.target_reached is False
    assert r1000.lines_read == 10
    assert r1000.subtitle_bytes == 230
    assert r1000.train_lines + r1000.val_lines == 10


def test_build_target_bytes_z_liniami_odrzuconymi(tmp_path):
    clean_dir = _zestaw_ab(tmp_path)
    gz = _gz_z_linii(tmp_path, ODRZUCONE_LINIE)
    out_dir = tmp_path / "out"

    raport = _wywolanie_ab(gz, clean_dir, out_dir, 45)

    assert raport.target_reached is True
    assert raport.lines_read == 8
    assert raport.train_lines == 2
    assert raport.val_lines == 0
    assert raport.subtitle_bytes == 45
    assert raport.rejected == {
        "angielski": 1,
        "krotka": 1,
        "przeklenstwa": 1,
        "wielkie": 1,
        "znaczniki": 1,
        "znaki": 1,
    }
    assert _val(out_dir) == "Ala ma kota.\n\n"


def test_build_po_przerwaniu_mozna_usunac_gz(tmp_path):
    clean_dir = _zestaw_ab(tmp_path)
    gz = _gz_z_linii(tmp_path, LINIE_NUMEROWANE)

    _wywolanie_ab(gz, clean_dir, tmp_path / "out", 70)
    gz.unlink()

    assert gz.exists() is False


@pytest.mark.parametrize("rodzaj", ["pusty_gzip", "pusty_plik"])
def test_build_pusty_gzip_i_pusty_plik(tmp_path, rodzaj):
    clean_dir = _zestaw_ab(tmp_path)
    gz = tmp_path / "pl.txt.gz"
    gz.write_bytes(gzip.compress(b"") if rodzaj == "pusty_gzip" else b"")
    out_dir = tmp_path / "out"

    raport = _wywolanie_ab(gz, clean_dir, out_dir, 1000)

    assert raport.lines_read == 0
    assert raport.train_lines == 0
    assert raport.val_lines == 0
    assert raport.subtitle_bytes == 0
    assert raport.rejected == {}
    assert raport.target_reached is False
    assert _train(out_dir) == "Ola ma psa.\n\n"
    assert _val(out_dir) == "Ala ma kota.\n\n"
    assert raport.train_bytes == len("Ola ma psa.\n\n")
    assert raport.val_bytes == len("Ala ma kota.\n\n")


def test_build_gz_bez_koncowego_newline(tmp_path):
    clean_dir = _zestaw_ab(tmp_path)
    gz = _gz_z_tekstu(tmp_path, "Dzien dobry panu.\nJak sie masz?")
    out_dir = tmp_path / "out"

    raport = _wywolanie_ab(gz, clean_dir, out_dir, 1000, block_lines=1)

    assert raport.lines_read == 2
    assert raport.train_lines == 1
    assert raport.val_lines == 1
    assert _train(out_dir).endswith("Dzien dobry panu.\n")
    assert _val(out_dir).endswith("Jak sie masz?\n")


def test_build_ucietyplik_gz(tmp_path):
    clean_dir = _zestaw_ab(tmp_path)
    tekst = "".join(f"Linia numer {n:04d} tutaj.\n" for n in range(5000))
    pelny = gzip.compress(tekst.encode("utf-8"))
    gz = tmp_path / "ucieta.gz"
    gz.write_bytes(pelny[: len(pelny) // 2])
    out_dir = tmp_path / "out"

    raport = _wywolanie_ab(
        gz, clean_dir, out_dir, 10_000_000, block_lines=100, val_every=2
    )

    assert 0 < raport.lines_read < 5000
    assert raport.train_lines + raport.val_lines == raport.lines_read
    assert raport.target_reached is False

    wszystkie = _linie_napisow(_train(out_dir), PREFIKS_TRAIN) + _linie_napisow(
        _val(out_dir), PREFIKS_VAL
    )
    assert sorted(wszystkie) == [
        f"Linia numer {n:04d} tutaj." for n in range(raport.lines_read)
    ]


def test_build_deterministyczny(tmp_path):
    clean_dir = _zestaw_ab(tmp_path)
    gz = _gz_z_linii(tmp_path, ODRZUCONE_LINIE)
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    raport1 = _wywolanie_ab(gz, clean_dir, out1, 100000)
    raport2 = _wywolanie_ab(gz, clean_dir, out2, 100000)

    assert raport1 == raport2
    for nazwa in ("train.txt", "val.txt", "report.json"):
        assert (out1 / nazwa).read_bytes() == (out2 / nazwa).read_bytes()


def test_build_tworzy_out_dir_z_rodzicami(tmp_path):
    clean_dir = _zestaw_ab(tmp_path)
    gz = _gz_z_linii(tmp_path, PRZYKLAD_LINIE)
    out_dir = tmp_path / "x" / "y" / "pretrain"

    _wywolanie_ab(gz, clean_dir, out_dir, 1000)

    assert sorted(p.name for p in out_dir.iterdir()) == ["report.json", "train.txt", "val.txt"]


def test_build_nic_nie_wypisuje(tmp_path, capsys):
    _przyklad_z_opisu(tmp_path)

    przechwycone = capsys.readouterr()
    assert przechwycone.out == ""
    assert przechwycone.err == ""


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_bytes": 0},
        {"target_bytes": -1},
        {"block_lines": 0},
        {"block_lines": -3},
        {"val_every": 1},
        {"val_every": 0},
    ],
)
def test_build_bledne_argumenty(tmp_path, kwargs):
    clean_dir = _zestaw_ab(tmp_path)
    gz = _gz_z_linii(tmp_path, PRZYKLAD_LINIE)
    parametry: dict[str, object] = {"target_bytes": 1000}
    parametry.update(kwargs)
    target_bytes = parametry.pop("target_bytes")

    with pytest.raises(ValueError):
        _wywolanie_ab(gz, clean_dir, tmp_path / "out", target_bytes, **parametry)  # type: ignore[arg-type]


def test_build_poprawne_argumenty_graniczne(tmp_path):
    clean_dir = _zestaw_ab(tmp_path)
    gz = _gz_z_linii(tmp_path, PRZYKLAD_LINIE)

    raport = _wywolanie_ab(gz, clean_dir, tmp_path / "out", 1, block_lines=1, val_every=2)

    assert raport.target_reached is True
    assert raport.lines_read == 1
    assert raport.train_lines == 1
    assert raport.val_lines == 0


def test_build_jedna_ksiazka_value_error(tmp_path):
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    (clean_dir / "a.txt").write_text("Ala ma kota.", encoding="utf-8")
    gz = _gz_z_linii(tmp_path, PRZYKLAD_LINIE)

    with pytest.raises(ValueError):
        build_pretrain_corpus(gz, clean_dir, tmp_path / "out", 1000)


def test_build_pusty_clean_dir_value_error(tmp_path):
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    gz = _gz_z_linii(tmp_path, PRZYKLAD_LINIE)

    with pytest.raises(ValueError):
        build_pretrain_corpus(gz, clean_dir, tmp_path / "out", 1000)
