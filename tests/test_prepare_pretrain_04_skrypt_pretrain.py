"""Testy skryptu scripts/prepare_pretrain.py (korpus do wstępnego treningu)."""

import dataclasses
import gzip
import io
import json
import runpy
import sys
import urllib.request
from pathlib import Path

import pytest

import nanowriter.data as data
from nanowriter.data import OPUS_SUBTITLES_PL_URL, PretrainReport

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_pretrain.py"

LINIE1 = [
    "Dzien dobry panu.",
    "- Dzien dobry.",
    "Nie.",
    "UWAŻAJ NA SIEBIE!",
    "♪ La la la ♪",
    "Co za kurwa dzień.",
    "Why am I so happy?",
    "Jak sie masz?",
    "Jak sie masz?",
    "Dobrze, dziekuje.",
]
RAW1 = gzip.compress(("\n".join(LINIE1) + "\n").encode("utf-8"))
LINIE2 = ["Dzien dobry panu.", "Nie.", "Tak.", "Dzien dobry panu."]
RAW2 = gzip.compress(("\n".join(LINIE2) + "\n").encode("utf-8"))

ODRZUCONE_PELNY = "odrzucone: angielski=1, krotka=1, powtorka=1, przeklenstwa=1, wielkie=1, znaki=1"
OPCJE_POMOCY = [
    "--data-dir",
    "--url",
    "--check-size",
    "--max-download-mb",
    "--target-mb",
    "--block-lines",
    "--val-every",
    "--book-val-fraction",
    "--seed",
    "--allow-large",
]


def _siec(*args, **kwargs):
    raise AssertionError("siec")


@pytest.fixture(autouse=True)
def _osłona_przed_siecią(monkeypatch):
    monkeypatch.setattr(data, "_open_url", _siec)
    monkeypatch.setattr(data, "_head_url", _siec)
    monkeypatch.setattr(urllib.request, "urlopen", _siec)


@pytest.fixture(scope="module")
def g():
    return runpy.run_path(str(SCRIPT))


@pytest.fixture(scope="module")
def main(g):
    return g["main"]


@pytest.fixture(scope="module")
def parse_args(g):
    return g["parse_args"]


@pytest.fixture
def d(tmp_path):
    katalog = tmp_path / "d"
    clean = katalog / "clean"
    clean.mkdir(parents=True)
    (clean / "a.txt").write_bytes(b"Ala ma kota.")
    (clean / "b.txt").write_bytes(b"Ola ma psa.")
    return katalog


@pytest.fixture
def gz(d):
    return d / "opus" / "pl.txt.gz"


@pytest.fixture
def fake_open(monkeypatch):
    """Zwraca funkcję ustawiającą atrapę _open_url; funkcja zwraca listę wywołań."""

    def ustaw(raw):
        calls = []

        def atrapa(url):
            calls.append(url)
            return io.BytesIO(raw)

        monkeypatch.setattr(data, "_open_url", atrapa)
        return calls

    return ustaw


@pytest.fixture
def fake_head(monkeypatch):
    """Zwraca funkcję ustawiającą atrapę _head_url; funkcja zwraca listę wywołań."""

    def ustaw(rozmiar):
        head_calls = []

        def atrapa(url):
            head_calls.append(url)
            return rozmiar

        monkeypatch.setattr(data, "_head_url", atrapa)
        return head_calls

    return ustaw


def _linie(capsys):
    return capsys.readouterr().out.splitlines()


def _linie_pelnego_przebiegu(gz, d):
    return [
        f"pobrano: 0.0 MB -> {gz}",
        "przeczytano linii napisow: 10",
        "napisy: train 2 linii, val 2 linii, razem 0.0 MB",
        "ksiazki: train 1, val 1",
        f"pliki: train 0.0 MB, val 0.0 MB -> {d / 'pretrain'}",
        ODRZUCONE_PELNY,
        f"uwaga: nie osiagnieto celu --target-mb; zwieksz --max-download-mb i usun plik {gz}, "
        "zeby pobrac wiecej",
    ]


PELNY_ARGV = ["--block-lines", "2", "--val-every", "2"]


# ---------------------------------------------------------------- wczytanie


def test_wczytanie_skryptu_nic_nie_wypisuje(capsys):
    wynik = runpy.run_path(str(SCRIPT))
    assert capsys.readouterr().out == ""
    assert "parse_args" in wynik
    assert "main" in wynik


# ---------------------------------------------------------------- parse_args


def test_parse_args_domyslne(parse_args):
    a = parse_args([])
    assert a.data_dir == Path("data")
    assert a.url == OPUS_SUBTITLES_PL_URL
    assert a.check_size is False
    assert a.max_download_mb == 300.0
    assert a.target_mb == 600.0
    assert a.block_lines == 20000
    assert a.val_every == 50
    assert a.book_val_fraction == 0.1
    assert a.seed == 0
    assert a.allow_large is False
    assert isinstance(a.max_download_mb, float)
    assert isinstance(a.target_mb, float)
    assert isinstance(a.block_lines, int)
    assert isinstance(a.data_dir, Path)


def test_parse_args_check_size(parse_args):
    a = parse_args(["--check-size", "--target-mb", "10"])
    assert a.check_size is True
    assert a.target_mb == 10.0
    assert a.max_download_mb == 300.0


def test_parse_args_wszystkie_opcje(parse_args):
    a = parse_args(
        [
            "--data-dir", "x",
            "--url", "http://example.invalid/a.gz",
            "--max-download-mb", "5",
            "--target-mb", "7",
            "--block-lines", "7",
            "--val-every", "3",
            "--book-val-fraction", "0.2",
            "--seed", "4",
            "--allow-large",
        ]
    )  # fmt: skip
    assert a.data_dir == Path("x")
    assert a.url == "http://example.invalid/a.gz"
    assert a.max_download_mb == 5.0
    assert a.target_mb == 7.0
    assert a.block_lines == 7
    assert a.val_every == 3
    assert a.book_val_fraction == 0.2
    assert a.seed == 4
    assert a.allow_large is True
    assert a.check_size is False


def test_parse_args_limit_wymaga_zgody(parse_args, capsys):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--max-download-mb", "2000"])
    assert exc.value.code == 2
    assert "--allow-large" in capsys.readouterr().err

    a = parse_args(["--max-download-mb", "2000", "--allow-large"])
    assert a.max_download_mb == 2000.0
    assert a.allow_large is True

    assert parse_args(["--max-download-mb", "1000"]).max_download_mb == 1000.0

    for wartosc in ("1000.5", "1001"):
        with pytest.raises(SystemExit) as exc:
            parse_args(["--max-download-mb", wartosc])
        assert exc.value.code == 2


def test_parse_args_argv_none(parse_args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prepare_pretrain.py", "--target-mb", "7", "--seed", "3"])
    a = parse_args()
    assert a.target_mb == 7.0
    assert a.seed == 3


def test_parse_args_help(parse_args, capsys):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--help"])
    assert exc.value.code == 0
    tekst = capsys.readouterr().out
    for opcja in OPCJE_POMOCY:
        assert opcja in tekst


# ---------------------------------------------------------------- main: --check-size


def test_main_check_size(main, fake_head, tmp_path, capsys):
    head_calls = fake_head(5_000_000)
    d = tmp_path / "nie_ma"
    assert main(["--check-size", "--data-dir", str(d)]) is None
    assert _linie(capsys) == ["rozmiar pliku: 5 MB", "limit pobierania: 300 MB"]
    assert head_calls == [OPUS_SUBTITLES_PL_URL]
    assert not d.exists()

    pusty = tmp_path / "pusty"
    pusty.mkdir()
    assert main(["--check-size", "--data-dir", str(pusty)]) is None
    assert _linie(capsys) == ["rozmiar pliku: 5 MB", "limit pobierania: 300 MB"]
    assert list(pusty.iterdir()) == []


def test_main_check_size_opcje(main, fake_head, tmp_path, capsys):
    head_calls = fake_head(1_930_000_000)
    main(
        [
            "--check-size",
            "--url", "http://example.invalid/x.gz",
            "--max-download-mb", "50",
            "--data-dir", str(tmp_path / "d"),
        ]
    )  # fmt: skip
    assert _linie(capsys) == ["rozmiar pliku: 1930 MB", "limit pobierania: 50 MB"]
    assert head_calls == ["http://example.invalid/x.gz"]


def test_main_check_size_argv_none(main, fake_head, tmp_path, monkeypatch, capsys):
    fake_head(5_000_000)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_pretrain.py", "--check-size", "--data-dir", str(tmp_path / "d")],
    )
    assert main() is None
    assert _linie(capsys)[0] == "rozmiar pliku: 5 MB"


def test_main_check_size_rozmiar_zero(main, fake_head, tmp_path):
    fake_head(0)
    with pytest.raises(ValueError):
        main(["--check-size", "--data-dir", str(tmp_path / "d")])


# ---------------------------------------------------------------- main: pełny przebieg


def test_main_pelny_przebieg(main, fake_open, d, gz, capsys):
    calls = fake_open(RAW1)
    r = main(["--data-dir", str(d), *PELNY_ARGV])

    assert _linie(capsys) == _linie_pelnego_przebiegu(gz, d)
    assert calls == [OPUS_SUBTITLES_PL_URL]
    assert gz.read_bytes() == RAW1

    pretrain = d / "pretrain"
    assert (pretrain / "train.txt").read_bytes().decode("utf-8") == (
        "Ola ma psa.\n\nDzien dobry panu.\nDzien dobry.\n"
    )
    assert (pretrain / "val.txt").read_bytes().decode("utf-8") == (
        "Ala ma kota.\n\nJak sie masz?\nDobrze, dziekuje.\n"
    )

    oczekiwany = {
        "train_books": ["b"],
        "val_books": ["a"],
        "lines_read": 10,
        "train_lines": 2,
        "val_lines": 2,
        "subtitle_bytes": 63,
        "train_bytes": 44,
        "val_bytes": 46,
        "rejected": {
            "angielski": 1,
            "krotka": 1,
            "powtorka": 1,
            "przeklenstwa": 1,
            "wielkie": 1,
            "znaki": 1,
        },
        "target_reached": False,
    }
    assert isinstance(r, PretrainReport)
    assert dataclasses.asdict(r) == oczekiwany

    raport = json.loads((pretrain / "report.json").read_text(encoding="utf-8"))
    assert raport == {
        **oczekiwany,
        "params": {
            "target_bytes": 600000000,
            "block_lines": 2,
            "val_every": 2,
            "book_val_fraction": 0.1,
            "seed": 0,
        },
    }


def test_main_nic_poza_trzema_katalogami(main, fake_open, d):
    fake_open(RAW1)
    main(["--data-dir", str(d), *PELNY_ARGV])
    assert sorted(p.name for p in d.iterdir()) == ["clean", "opus", "pretrain"]
    assert sorted(p.name for p in (d / "pretrain").iterdir()) == [
        "report.json",
        "train.txt",
        "val.txt",
    ]
    assert sorted(p.name for p in (d / "opus").iterdir()) == ["pl.txt.gz"]
    assert (d / "clean" / "a.txt").read_bytes() == b"Ala ma kota."
    assert (d / "clean" / "b.txt").read_bytes() == b"Ola ma psa."


def test_main_ponowny_przebieg_uzywa_pliku(main, fake_open, d, gz, capsys):
    calls = fake_open(RAW1)
    argv = ["--data-dir", str(d), *PELNY_ARGV]
    main(argv)
    pierwsze_linie = _linie(capsys)
    pretrain = d / "pretrain"
    nazwy = ("train.txt", "val.txt", "report.json")
    pierwsze_bajty = {n: (pretrain / n).read_bytes() for n in nazwy}

    main(argv)
    drugie_linie = _linie(capsys)

    assert drugie_linie[0] == f"plik istnieje, pomijam pobieranie: {gz}"
    assert len(drugie_linie) == 7
    assert drugie_linie[1:] == pierwsze_linie[1:]
    assert len(calls) == 1
    for n in nazwy:
        assert (pretrain / n).read_bytes() == pierwsze_bajty[n]


def test_main_odrzucone_sortowanie(main, fake_open, d, gz, capsys):
    fake_open(RAW2)
    r = main(["--data-dir", str(d)])
    linie = _linie(capsys)
    assert linie[:6] == [
        f"pobrano: 0.0 MB -> {gz}",
        "przeczytano linii napisow: 4",
        "napisy: train 1 linii, val 0 linii, razem 0.0 MB",
        "ksiazki: train 1, val 1",
        f"pliki: train 0.0 MB, val 0.0 MB -> {d / 'pretrain'}",
        "odrzucone: krotka=2, powtorka=1",
    ]
    assert linie[6] == _linie_pelnego_przebiegu(gz, d)[6]
    assert len(linie) == 7
    assert r.lines_read == 4
    assert r.train_lines == 1
    assert r.val_lines == 0
    assert r.subtitle_bytes == 18
    assert r.rejected == {"krotka": 2, "powtorka": 1}
    assert r.target_reached is False


def test_main_cel_osiagniety_bez_uwagi(main, fake_open, d, capsys):
    fake_open(RAW1)
    r = main(["--data-dir", str(d), "--target-mb", "0.00002"])
    linie = _linie(capsys)
    assert r.target_reached is True
    assert r.lines_read == 2
    assert r.train_lines == 2
    assert r.val_lines == 0
    assert r.subtitle_bytes == 31
    assert r.rejected == {}
    assert not any(ln.startswith("uwaga:") for ln in linie)
    assert "odrzucone: brak" in linie
    assert "przeczytano linii napisow: 2" in linie
    assert "napisy: train 2 linii, val 0 linii, razem 0.0 MB" in linie
    raport = json.loads((d / "pretrain" / "report.json").read_text(encoding="utf-8"))
    assert raport["target_reached"] is True
    assert raport["params"]["target_bytes"] == 20


def test_main_twardy_limit_pobierania(main, fake_open, d, gz):
    assert len(RAW1) > 50
    fake_open(RAW1)
    r = main(["--data-dir", str(d), "--max-download-mb", "0.00005"])
    assert isinstance(r, PretrainReport)
    assert gz.stat().st_size == 50
    assert not (d / "opus" / "pl.txt.gz.part").exists()
    assert r.target_reached is False


def test_main_bledy_nie_sa_lapane(main, fake_open, tmp_path):
    fake_open(RAW1)
    d = tmp_path / "d"
    (d / "clean").mkdir(parents=True)
    (d / "clean" / "a.txt").write_bytes(b"Ala ma kota.")
    with pytest.raises(ValueError):
        main(["--data-dir", str(d)])
