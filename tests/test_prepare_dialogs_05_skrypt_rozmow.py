"""Testy skryptu scripts/prepare_dialogs.py (rozmowy) oraz bramka zadania 07.

Część „napisy bez sieci” sprawdza scripts/prepare_pretrain.py z podmienionym _open_url.
Testy nie korzystają z sieci, GPU ani torcha.
"""

import dataclasses
import gzip
import importlib.util
import io
import json
from pathlib import Path

import pytest

import nanowriter.data as data
from nanowriter.data import (
    OPUS_SUBTITLES_PL_URL,
    PROFANITY_STEMS,
    DialogCorpusResult,
    SubtitleFilter,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
REAL = Path(__file__).resolve().parents[1] / "data" / "dialogi" / "surowe"


def _load(name: str):
    """Ładuje skrypt z scripts/ jako moduł (scripts/ nie jest pakietem)."""
    sciezka = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, sciezka)
    assert spec is not None and spec.loader is not None
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


dialogs = _load("prepare_dialogs")
pretrain = _load("prepare_pretrain")

# ---------------------------------------------------------------------------
# Rozmowy testowe
# ---------------------------------------------------------------------------

D1 = ["A: Dzień dobry, jak się masz?", "B: Dziękuję, dobrze. A ty?",
      "A: Też dobrze, idę na spacer.", "B: Miłego spaceru."]
D2 = ["A: Słyszałeś prognozę pogody na jutro?", "B: Podobno ma padać od rana.",
      "A: To chyba zostanę w domu.", "B: Dobry pomysł, ja też.",
      "A: Zrobię sobie herbatę.", "B: Smacznej herbaty."]
D3 = ["A: Gdzie jest najbliższa piekarnia?", "B: Za rogiem, obok apteki.",
      "A: Dziękuję bardzo.", "B: Proszę bardzo."]
D4 = ["A: Którą godzinę masz?", "B: Dochodzi piąta.",
      "A: Muszę już iść.", "B: Do zobaczenia jutro."]
D5 = ["A: Lubisz gotować?", "B: Bardzo, zwłaszcza zupy.",
      "A: Jaką zupę lubisz najbardziej?", "B: Pomidorową z makaronem."]
D6 = ["A: Czy masz psa?", "B: Mam, nazywa się Burek.",
      "A: Ile ma lat?", "B: Pięć, jest bardzo żywy."]
DUP1 = ["A: DZIEŃ DOBRY, JAK SIĘ MASZ?", "B: DZIĘKUJĘ, DOBRZE. A TY?",
        "A: TEŻ DOBRZE, IDĘ NA SPACER.", "B: MIŁEGO SPACERU."]
DUP2 = ["A:   słyszałeś prognozę\tpogody na jutro?", "B:  podobno ma padać od rana.",
        "A: to chyba zostanę   w domu.", "B: dobry pomysł, ja też.",
        "A: zrobię sobie herbatę.", "B: smacznej herbaty."]
ZLA = ["A: Jak się dziś czujesz?", "B: Całkiem dobrze.",
       "C: Nie wiem, co powiedzieć.", "B: Dziękuję."]
START = ["B: Cześć, co słychać?", "A: Wszystko dobrze.", "B: To świetnie.", "A: Dzięki."]
KONIEC = ["A: Idziesz na zakupy?", "B: Tak, zaraz wychodzę.", "A: Kup mi chleb.",
          "B: Dobrze, kupię.", "A: Dziękuję."]
KOLEJNOSC = ["A: Co dziś gotujesz?", "B: Jeszcze nie wiem.", "B: Może zupę.",
             "A: Brzmi dobrze.", "B: Zobaczymy."]
PUSTA = ["A: Jak minął dzień?", "B: Spokojnie.", "", "A: To dobrze.", "B: Tak."]
ZNAKI = ["A: Co robisz?", "B: Czytam *ważną* książkę.", "A: O czym jest?", "B: O podróży."]
KROTKA = ["A: Hej.", "B: Cześć."]
ASYSTENT = ["A: Potrzebuję planu dnia.", "B: Oczywiście! Oto plan na dziś.",
            "A: Dziękuję.", "B: Proszę."]
N = [
    [f"A: Dzień dobry, numer {n}.", f"B: Dzień dobry, panie {n}.",
     f"A: Co słychać u pana {n}?", f"B: Dobrze, dziękuję {n}."]
    for n in range(4)
]
N0, N1, N2, N3 = N

MIX_REJECTED = {
    "001.txt#3": "zla-linia",
    "001.txt#4": "start",
    "001.txt#5": "koniec",
    "001.txt#6": "kolejnosc",
    "001.txt#8": "pusta-linia",
    "001.txt#9": "znaki",
    "001.txt#10": "dlugosc",
    "002.txt#6": "asystent",
}
MIX_DUPLICATES = {"001.txt#7": "001.txt#1", "002.txt#5": "001.txt#2"}


def _zapisz(plik: Path, rozmowy: list[list[str]]) -> None:
    """Zapisuje rozmowy: linie złączone \\n, po każdej rozmowie \\n---\\n."""
    tekst = "".join("\n".join(r) + "\n---\n" for r in rozmowy)
    plik.write_bytes(tekst.encode("utf-8"))


def _mix(tmp_path: Path) -> Path:
    """Tworzy katalog MIX z 16 rozmowami i plikiem nie-txt."""
    src = tmp_path / "surowe"
    src.mkdir()
    _zapisz(src / "001.txt", [D1, D2, ZLA, START, KONIEC, KOLEJNOSC, DUP1, PUSTA, ZNAKI, KROTKA])
    _zapisz(src / "002.txt", [D3, D4, D5, D6, DUP2, ASYSTENT])
    (src / "notes.md").write_text("A: x\n", encoding="utf-8")
    return src


def _run_mix(tmp_path: Path, out_name: str = "wynik"):
    """Wywołuje main na katalogu MIX; zwraca (src, out, wynik)."""
    src = _mix(tmp_path) if not (tmp_path / "surowe").exists() else tmp_path / "surowe"
    out = tmp_path / out_name
    wynik = dialogs.main(["--src", str(src), "--out", str(out), "--val-fraction", "0.25"])
    return src, out, wynik


def _czytaj(p: Path) -> str:
    return p.read_bytes().decode("utf-8")


def _bloki(tekst: str) -> list[tuple[str, ...]]:
    if tekst == "":
        return []
    return [tuple(blok.split("\n")) for blok in tekst[:-1].split("\n\n")]


def _sprawdz_format(tekst: str, min_lin: int = 1, max_lin: int | None = None) -> list:
    """Sprawdza format strumienia rozmów i zwraca listę bloków."""
    assert tekst != ""
    assert "\r" not in tekst
    assert tekst.endswith("\n") and not tekst.endswith("\n\n")
    assert "\n\n\n" not in tekst
    for linia in tekst.split("\n"):
        assert linia == "" or linia.startswith(("A: ", "B: "))
    bloki = _bloki(tekst)
    for blok in bloki:
        assert all(linia != "" for linia in blok)
        assert blok[0].startswith("A: ")
        assert blok[-1].startswith("B: ")
        for k, linia in enumerate(blok):
            assert linia.startswith("A: " if k % 2 == 0 else "B: ")
        assert len(blok) >= min_lin
        if max_lin is not None:
            assert len(blok) <= max_lin
    return bloki


# ---------------------------------------------------------------------------
# parse_args rozmów
# ---------------------------------------------------------------------------


def test_parse_args_dialogs_domyslne():
    args = dialogs.parse_args([])

    assert args.src == Path("data/dialogi/surowe")
    assert args.out == Path("data/rozmowy")
    assert args.val_fraction == 0.05
    assert isinstance(args.val_fraction, float)
    assert args.seed == 0
    assert isinstance(args.seed, int)
    assert args.min_lines == 4
    assert args.max_lines == 16
    assert isinstance(args.min_lines, int)
    assert isinstance(args.max_lines, int)


def test_parse_args_dialogs_wlasne_wartosci():
    args = dialogs.parse_args([
        "--src", "x", "--out", "y", "--val-fraction", "0.1", "--seed", "3",
        "--min-lines", "2", "--max-lines", "20",
    ])

    assert args.src == Path("x")
    assert args.out == Path("y")
    assert args.val_fraction == 0.1
    assert args.seed == 3
    assert args.min_lines == 2
    assert args.max_lines == 20
    assert isinstance(args.src, Path)
    assert isinstance(args.out, Path)
    assert isinstance(args.val_fraction, float)
    assert isinstance(args.seed, int)
    assert isinstance(args.min_lines, int)
    assert isinstance(args.max_lines, int)

    args = dialogs.parse_args(["--val-fraction", "0.1", "--max-lines", "20"])
    assert args.val_fraction == 0.1
    assert args.max_lines == 20
    assert args.src == Path("data/dialogi/surowe")
    assert args.out == Path("data/rozmowy")
    assert args.seed == 0
    assert args.min_lines == 4


def test_parse_args_dialogs_zly_typ():
    with pytest.raises(SystemExit) as exc1:
        dialogs.parse_args(["--seed", "abc"])
    assert exc1.value.code == 2

    with pytest.raises(SystemExit) as exc2:
        dialogs.parse_args(["--max-lines", "1.5"])
    assert exc2.value.code == 2


# ---------------------------------------------------------------------------
# main rozmów
# ---------------------------------------------------------------------------


def test_main_przyklad_z_opisu(tmp_path):
    src = tmp_path / "s"
    src.mkdir()
    _zapisz(src / "001.txt", [N0, N1, N2, N3])
    out = tmp_path / "out"

    wynik = dialogs.main(["--src", str(src), "--out", str(out), "--val-fraction", "0.25"])

    assert isinstance(wynik, DialogCorpusResult)
    assert wynik.train == 3
    assert wynik.val == 1
    assert _czytaj(out / "val.txt") == "\n".join(N2) + "\n"
    assert _czytaj(out / "train.txt") == (
        "\n".join(N0) + "\n\n" + "\n".join(N1) + "\n\n" + "\n".join(N3) + "\n"
    )
    assert wynik.train_path == out / "train.txt"
    assert wynik.val_path == out / "val.txt"


def test_main_mix_wynik_i_raport(tmp_path):
    _, out, wynik = _run_mix(tmp_path)

    assert isinstance(wynik, DialogCorpusResult)
    raport = wynik.report
    assert raport.files == 2
    assert raport.found == 16
    assert raport.accepted == 6
    assert raport.rejected == MIX_REJECTED
    assert list(raport.rejected.items()) == list(MIX_REJECTED.items())
    assert raport.duplicates == MIX_DUPLICATES
    assert list(raport.duplicates.items()) == list(MIX_DUPLICATES.items())
    assert raport.found == raport.accepted + len(raport.rejected) + len(raport.duplicates)
    assert wynik.train + wynik.val == 6
    assert wynik.train >= 1
    assert wynik.val >= 1
    assert wynik.train_path == out / "train.txt"
    assert wynik.val_path == out / "val.txt"
    assert sorted(p.name for p in out.iterdir()) == ["report.json", "train.txt", "val.txt"]


def test_main_mix_report_json(tmp_path):
    _, out, wynik = _run_mix(tmp_path)

    tekst = _czytaj(out / "report.json")
    dane = json.loads(tekst)

    assert list(dane) == [
        "seed", "val_fraction", "min_lines", "max_lines", "files", "found", "accepted",
        "train", "val", "rejected", "duplicates", "reasons",
    ]
    assert dane["seed"] == 0
    assert dane["val_fraction"] == 0.25
    assert dane["min_lines"] == 4
    assert dane["max_lines"] == 16
    assert dane["files"] == 2
    assert dane["found"] == 16
    assert dane["accepted"] == 6
    assert dane["train"] == wynik.train
    assert dane["val"] == wynik.val
    assert dane["rejected"] == MIX_REJECTED
    assert dane["duplicates"] == MIX_DUPLICATES
    assert dane["reasons"] == {
        "asystent": 1, "dlugosc": 1, "kolejnosc": 1, "koniec": 1,
        "pusta-linia": 1, "start": 1, "zla-linia": 1, "znaki": 1,
    }
    assert tekst.startswith('{\n  "seed": 0,')
    assert tekst.endswith("\n") and not tekst.endswith("\n\n")
    assert "\r" not in tekst


def test_main_mix_wypisuje_podsumowanie(tmp_path, capsys):
    _, _, wynik = _run_mix(tmp_path)

    linie = capsys.readouterr().out.splitlines()

    assert linie[0] == "pliki: 2"
    assert linie[1] == "znalezione rozmowy: 16"
    assert linie[2] == "przyjete: 6"
    assert linie[3] == "odrzucone: 8"
    assert linie[4:12] == [
        "  001.txt#3: zla-linia",
        "  001.txt#4: start",
        "  001.txt#5: koniec",
        "  001.txt#6: kolejnosc",
        "  001.txt#8: pusta-linia",
        "  001.txt#9: znaki",
        "  001.txt#10: dlugosc",
        "  002.txt#6: asystent",
    ]
    assert linie[12] == "duplikaty: 2"
    assert linie[13] == f"train: {wynik.train} rozmow -> {wynik.train_path}"
    assert linie[14] == f"val: {wynik.val} rozmow -> {wynik.val_path}"
    assert len(linie) == 15
    for linia in linie[0:13]:
        assert linia.isascii()


def test_bramka_rozmow_format_strumienia(tmp_path):
    _, out, wynik = _run_mix(tmp_path)

    for nazwa, oczekiwane in (("train.txt", wynik.train), ("val.txt", wynik.val)):
        tekst = _czytaj(out / nazwa)
        bloki = _sprawdz_format(tekst)
        assert len(bloki) == oczekiwane


def test_bramka_rozmow_rozlacznosc_i_kompletnosc(tmp_path):
    _, out, _ = _run_mix(tmp_path)

    train_tekst = _czytaj(out / "train.txt")
    val_tekst = _czytaj(out / "val.txt")
    train_lista = _bloki(train_tekst)
    val_lista = _bloki(val_tekst)
    train_bloki = set(train_lista)
    val_bloki = set(val_lista)

    assert train_bloki & val_bloki == set()
    assert len(train_lista) == len(train_bloki)
    assert len(val_lista) == len(val_bloki)
    assert train_bloki | val_bloki == {tuple(d) for d in (D1, D2, D3, D4, D5, D6)}

    for zla in (DUP1, DUP2, ZLA, START, KONIEC, KOLEJNOSC, PUSTA, ZNAKI, KROTKA, ASYSTENT):
        assert tuple(zla) not in train_bloki
        assert tuple(zla) not in val_bloki
    for tekst in (train_tekst, val_tekst):
        assert "*" not in tekst
        assert "Oczywiście" not in tekst


def test_main_mix_deterministyczny(tmp_path):
    _, out1, wynik1 = _run_mix(tmp_path, "w1")
    _, out2, wynik2 = _run_mix(tmp_path, "w2")

    for nazwa in ("train.txt", "val.txt", "report.json"):
        assert (out1 / nazwa).read_bytes() == (out2 / nazwa).read_bytes()
    assert wynik1.train == wynik2.train
    assert wynik1.val == wynik2.val


def test_main_nie_zmienia_plikow_wejsciowych(tmp_path):
    src = _mix(tmp_path)
    przed = {p.name: p.read_bytes() for p in src.iterdir()}

    dialogs.main(["--src", str(src), "--out", str(tmp_path / "wynik"), "--val-fraction", "0.25"])

    po = {p.name: p.read_bytes() for p in src.iterdir()}
    assert przed == po
    assert "notes.md" in po


def test_main_ignoruje_pliki_nie_txt(tmp_path):
    src = tmp_path / "s"
    src.mkdir()
    _zapisz(src / "001.txt", [D1, D3])
    (src / "notes.md").write_bytes(b"A: x\n---\n")

    wynik = dialogs.main(["--src", str(src), "--out", str(tmp_path / "out")])

    assert wynik.report.files == 1
    assert wynik.report.found == 2
    assert wynik.report.accepted == 2


def test_main_wypisuje_20_odrzuconych_i_reszte(tmp_path, capsys):
    src = tmp_path / "s"
    src.mkdir()
    _zapisz(src / "001.txt", [ZLA] * 25 + [D1, D3])
    out = tmp_path / "out"

    wynik = dialogs.main(["--src", str(src), "--out", str(out)])

    linie = capsys.readouterr().out.splitlines()
    assert "odrzucone: 25" in linie
    i = linie.index("odrzucone: 25")
    assert linie[i + 1: i + 21] == [f"  001.txt#{n}: zla-linia" for n in range(1, 21)]
    assert linie[i + 21] == "  ... i 5 kolejnych (patrz report.json)"
    assert linie[i + 22] == "duplikaty: 0"
    assert "  001.txt#21: zla-linia" not in linie
    assert wynik.report.found == 27
    assert wynik.report.accepted == 2
    assert len(wynik.report.rejected) == 25
    dane = json.loads(_czytaj(out / "report.json"))
    assert len(dane["rejected"]) == 25
    assert dane["reasons"] == {"zla-linia": 25}


def test_main_dokladnie_20_odrzuconych_bez_reszty(tmp_path, capsys):
    src = tmp_path / "s"
    src.mkdir()
    _zapisz(src / "001.txt", [ZLA] * 20 + [D1, D3])

    dialogs.main(["--src", str(src), "--out", str(tmp_path / "out")])

    linie = capsys.readouterr().out.splitlines()
    assert "odrzucone: 20" in linie
    i = linie.index("odrzucone: 20")
    assert linie[i + 1: i + 21] == [f"  001.txt#{n}: zla-linia" for n in range(1, 21)]
    assert linie[i + 21] == "duplikaty: 0"
    assert not any("kolejnych" in linia for linia in linie)


def test_main_min_lines_i_max_lines_sa_przekazywane(tmp_path):
    # (a) --max-lines
    src_a = tmp_path / "a"
    src_a.mkdir()
    _zapisz(src_a / "001.txt", [D1, D2, D3])
    wynik = dialogs.main(
        ["--src", str(src_a), "--out", str(tmp_path / "out_a"), "--max-lines", "4"]
    )
    assert wynik.report.rejected == {"001.txt#2": "dlugosc"}
    assert wynik.report.accepted == 2

    # (b) --min-lines
    src_b = tmp_path / "b"
    src_b.mkdir()
    _zapisz(src_b / "001.txt", [["A: Hej.", "B: Cześć."], ["A: Co tam?", "B: Nic."]])
    with pytest.raises(ValueError):
        dialogs.main(["--src", str(src_b), "--out", str(tmp_path / "out_b1")])
    wynik = dialogs.main(
        ["--src", str(src_b), "--out", str(tmp_path / "out_b2"), "--min-lines", "2"]
    )
    assert wynik.report.accepted == 2
    assert wynik.report.rejected == {}
    assert wynik.train == 1
    assert wynik.val == 1


def test_main_val_fraction_i_seed_sa_przekazywane(tmp_path):
    src = tmp_path / "s"
    src.mkdir()
    _zapisz(src / "001.txt", [N0, N1, N2, N3])
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    wynik = dialogs.main(
        ["--src", str(src), "--out", str(out1), "--val-fraction", "0.25", "--seed", "0"]
    )
    assert wynik.val == 1
    dane = json.loads(_czytaj(out1 / "report.json"))
    assert dane["val_fraction"] == 0.25
    assert dane["seed"] == 0

    dialogs.main(
        ["--src", str(src), "--out", str(out2), "--seed", "5", "--val-fraction", "0.5"]
    )
    dane = json.loads(_czytaj(out2 / "report.json"))
    assert dane["seed"] == 5
    assert dane["val_fraction"] == 0.5


def test_main_brak_katalogu_src(tmp_path):
    out = tmp_path / "out"

    with pytest.raises(FileNotFoundError):
        dialogs.main(["--src", str(tmp_path / "nie_ma"), "--out", str(out)])

    assert not out.exists()


def test_main_katalog_bez_rozmow(tmp_path):
    pusty = tmp_path / "pusty"
    pusty.mkdir()
    with pytest.raises(ValueError):
        dialogs.main(["--src", str(pusty), "--out", str(tmp_path / "o1")])

    jedna = tmp_path / "jedna"
    jedna.mkdir()
    _zapisz(jedna / "001.txt", [D1])
    with pytest.raises(ValueError):
        dialogs.main(["--src", str(jedna), "--out", str(tmp_path / "o2")])

    zle = tmp_path / "zle"
    zle.mkdir()
    _zapisz(zle / "001.txt", [ZLA, START])
    with pytest.raises(ValueError):
        dialogs.main(["--src", str(zle), "--out", str(tmp_path / "o3")])


def test_main_zwraca_dialogcorpusresult(tmp_path):
    _, _, wynik = _run_mix(tmp_path)

    assert isinstance(wynik, DialogCorpusResult)
    assert isinstance(wynik.train, int)
    assert isinstance(wynik.train_path, Path)
    assert wynik.train_path.is_file()
    assert wynik.val_path.is_file()


@pytest.mark.skipif(not REAL.is_dir(), reason="brak data/dialogi/surowe")
def test_bramka_prawdziwe_rozmowy(tmp_path):
    przed = {p.name: p.read_bytes() for p in REAL.iterdir() if p.is_file()}
    out = tmp_path / "out"

    wynik = dialogs.main(["--src", str(REAL), "--out", str(out)])

    raport = wynik.report
    assert raport.accepted > 0
    assert wynik.train > wynik.val > 0
    assert wynik.train + wynik.val == raport.accepted
    assert raport.found == raport.accepted + len(raport.rejected) + len(raport.duplicates)
    po = {p.name: p.read_bytes() for p in REAL.iterdir() if p.is_file()}
    assert po == przed
    assert wynik.train_path.parent == out
    assert wynik.val_path.parent == out

    train_bloki = _sprawdz_format(_czytaj(wynik.train_path), 4, 16)
    val_bloki = _sprawdz_format(_czytaj(wynik.val_path), 4, 16)
    assert set(train_bloki) & set(val_bloki) == set()


# ---------------------------------------------------------------------------
# Część „napisy bez sieci” (scripts/prepare_pretrain.py)
# ---------------------------------------------------------------------------


def _linie_gz() -> list[str]:
    linie: list[str] = []
    for i in range(20):
        linie += [
            f"Dzień dobry, panie numer {i}.",
            f"- Tak jest, panie numer {i}.",
            f"UWAŻAJ NA SIEBIE {i}!",
            f"Co za kurwa dzień numer {i}.",
            f"♪ La la la {i} ♪",
            "Nie.",
            f"Why are you here {i}?",
        ]
    linie += ["Ten sam tekst wraca.", "Ten sam tekst wraca."]
    return linie


RAW = gzip.compress(("\n".join(_linie_gz()) + "\n").encode("utf-8"))

KSIAZKI = {
    "a": "Ala ma kota.",
    "b": "Ola ma psa.",
    "c": "Ela ma rybę.",
    "d": "Ula ma ptaka.",
}

PRE_ARGS = ["--block-lines", "10", "--val-every", "2", "--book-val-fraction", "0.5"]

S: list[str] = []
for _i in range(20):
    S.append(f"Dzień dobry, panie numer {_i}.")
    S.append(f"Tak jest, panie numer {_i}.")
S.append("Ten sam tekst wraca.")


def _ksiazki(d: Path) -> None:
    clean = d / "clean"
    clean.mkdir(parents=True)
    for slug, tresc in KSIAZKI.items():
        (clean / f"{slug}.txt").write_text(tresc, encoding="utf-8")


def _podmien_siec(monkeypatch) -> tuple[list[str], list[str]]:
    otwarcia: list[str] = []
    head_calls: list[str] = []

    def fake_open(url):
        otwarcia.append(url)
        return io.BytesIO(RAW)

    def fake_head(url):
        head_calls.append(url)
        return 5_000_000

    monkeypatch.setattr(data, "_open_url", fake_open)
    monkeypatch.setattr(data, "_head_url", fake_head)
    return otwarcia, head_calls


def _przebieg_pretrain(tmp_path, monkeypatch, capsys):
    d = tmp_path / "d"
    _ksiazki(d)
    otwarcia, head_calls = _podmien_siec(monkeypatch)
    raport = pretrain.main(["--data-dir", str(d), *PRE_ARGS])
    wypisane = capsys.readouterr().out
    return d, raport, wypisane, otwarcia, head_calls


def _napisy(d: Path, nazwa: str, raport) -> list[str]:
    tekst = _czytaj(d / "pretrain" / nazwa)
    ksiazki = raport.train_books if nazwa == "train.txt" else raport.val_books
    prefiks = "\n\n".join(_czytaj(d / "clean" / f"{slug}.txt") for slug in ksiazki) + "\n\n"
    assert tekst.startswith(prefiks)
    reszta = tekst[len(prefiks):]
    linie = reszta.split("\n")
    assert linie[-1] == ""
    return linie[:-1]


def test_parse_args_pretrain_przyklady():
    args = pretrain.parse_args([])
    assert args.max_download_mb == 300.0
    assert args.target_mb == 600.0
    assert args.check_size is False
    assert args.allow_large is False
    assert args.data_dir == Path("data")
    assert args.url == OPUS_SUBTITLES_PL_URL

    args = pretrain.parse_args(["--max-download-mb", "2000", "--allow-large"])
    assert args.max_download_mb == 2000.0
    assert args.allow_large is True

    args = pretrain.parse_args(["--max-download-mb", "1000"])
    assert args.max_download_mb == 1000.0


def test_bramka_napisy_pliki_i_raport(tmp_path, monkeypatch, capsys):
    d, raport, _, otwarcia, head_calls = _przebieg_pretrain(tmp_path, monkeypatch, capsys)

    for nazwa in ("train.txt", "val.txt", "report.json"):
        assert (d / "pretrain" / nazwa).is_file()
    assert (d / "pretrain" / "train.txt").stat().st_size > 0
    assert (d / "pretrain" / "val.txt").stat().st_size > 0
    assert (d / "opus" / "pl.txt.gz").read_bytes() == RAW
    assert otwarcia == [OPUS_SUBTITLES_PL_URL]
    assert head_calls == []

    assert raport.lines_read == 142
    assert raport.train_lines == 21
    assert raport.val_lines == 20
    assert raport.subtitle_bytes == 1121
    assert raport.rejected == {
        "angielski": 20, "krotka": 20, "powtorka": 1,
        "przeklenstwa": 20, "wielkie": 20, "znaki": 20,
    }
    assert raport.target_reached is False
    assert sorted(p.name for p in (d / "pretrain").iterdir()) == [
        "report.json", "train.txt", "val.txt",
    ]


def test_bramka_napisy_ksiazki_rozlaczne(tmp_path, monkeypatch, capsys):
    d, raport, _, _, _ = _przebieg_pretrain(tmp_path, monkeypatch, capsys)

    assert raport.train_books
    assert raport.val_books
    assert set(raport.train_books) & set(raport.val_books) == set()
    assert sorted(raport.train_books + raport.val_books) == ["a", "b", "c", "d"]

    train = _czytaj(d / "pretrain" / "train.txt")
    val = _czytaj(d / "pretrain" / "val.txt")
    for slug in raport.train_books:
        assert KSIAZKI[slug] in train
        assert KSIAZKI[slug] not in val
    for slug in raport.val_books:
        assert KSIAZKI[slug] in val
        assert KSIAZKI[slug] not in train


def test_bramka_napisy_linie_rozlaczne_i_dokladne(tmp_path, monkeypatch, capsys):
    d, raport, _, _, _ = _przebieg_pretrain(tmp_path, monkeypatch, capsys)

    train_linie = _napisy(d, "train.txt", raport)
    val_linie = _napisy(d, "val.txt", raport)

    assert set(train_linie) & set(val_linie) == set()
    assert len(train_linie) == 21
    assert len(val_linie) == 20
    assert val_linie == [S[k] for k in range(10, 20)] + [S[k] for k in range(30, 40)]
    assert train_linie == (
        [S[k] for k in range(0, 10)] + [S[k] for k in range(20, 30)] + ["Ten sam tekst wraca."]
    )
    assert (train_linie + val_linie).count("Ten sam tekst wraca.") == 1


def test_bramka_napisy_linie_zgodne_z_regulami_filtra(tmp_path, monkeypatch, capsys):
    d, raport, _, _, _ = _przebieg_pretrain(tmp_path, monkeypatch, capsys)

    train_linie = _napisy(d, "train.txt", raport)
    val_linie = _napisy(d, "val.txt", raport)
    wszystkie = train_linie + val_linie

    assert len(wszystkie) == 41
    for linia in wszystkie:
        assert linia != ""
        assert not linia.startswith(("-", "–", "—"))
        assert not any(znak in linia for znak in ("♪", "/", "#"))
        assert not linia.isupper()
        assert len(linia.split()) >= 2
        male = linia.lower()
        assert not any(rdzen in male for rdzen in PROFANITY_STEMS)
        for zakazane in ("UWAŻAJ", "kurwa", "Why", "La la la"):
            assert zakazane not in linia
        assert SubtitleFilter().accept(linia) == linia

    for odrzucona in (
        "Nie.", "UWAŻAJ NA SIEBIE 0!", "Co za kurwa dzień numer 0.",
        "♪ La la la 0 ♪", "Why are you here 0?", "- Tak jest, panie numer 0.",
    ):
        assert odrzucona not in wszystkie


def test_bramka_napisy_pliki_bez_cr_i_ze_znakiem_nowej_linii_na_koncu(
    tmp_path, monkeypatch, capsys
):
    d, _, _, _, _ = _przebieg_pretrain(tmp_path, monkeypatch, capsys)

    for nazwa in ("train.txt", "val.txt", "report.json"):
        bajty = (d / "pretrain" / nazwa).read_bytes()
        assert b"\r" not in bajty
        assert not bajty.startswith(b"\xef\xbb\xbf")
    for nazwa in ("train.txt", "val.txt"):
        tekst = _czytaj(d / "pretrain" / nazwa)
        assert tekst.endswith("\n")
        assert not tekst.endswith("\n\n")


def test_bramka_napisy_report_json_zgodny_z_plikami(tmp_path, monkeypatch, capsys):
    d, raport, _, _, _ = _przebieg_pretrain(tmp_path, monkeypatch, capsys)
    train_linie = _napisy(d, "train.txt", raport)
    val_linie = _napisy(d, "val.txt", raport)

    tekst = _czytaj(d / "pretrain" / "report.json")
    dane = json.loads(tekst)

    assert list(dane) == [
        "train_books", "val_books", "lines_read", "train_lines", "val_lines",
        "subtitle_bytes", "train_bytes", "val_bytes", "rejected", "target_reached", "params",
    ]
    bez_params = {k: v for k, v in dane.items() if k != "params"}
    assert bez_params == dataclasses.asdict(raport)
    assert dane["params"] == {
        "target_bytes": 600000000, "block_lines": 10, "val_every": 2,
        "book_val_fraction": 0.5, "seed": 0,
    }
    assert dane["train_lines"] == len(train_linie)
    assert dane["val_lines"] == len(val_linie)
    assert dane["train_bytes"] == len((d / "pretrain" / "train.txt").read_bytes())
    assert dane["val_bytes"] == len((d / "pretrain" / "val.txt").read_bytes())
    zapisane = train_linie + val_linie
    assert dane["subtitle_bytes"] == sum(len(x.encode("utf-8")) + 1 for x in zapisane)
    assert dane["subtitle_bytes"] == 1121
    assert dane["lines_read"] == (
        dane["train_lines"] + dane["val_lines"] + sum(dane["rejected"].values())
    )
    assert dane["train_books"] == raport.train_books
    assert dane["val_books"] == raport.val_books
    assert tekst.endswith("\n") and not tekst.endswith("\n\n")


def test_bramka_napisy_wypisane_podsumowanie(tmp_path, monkeypatch, capsys):
    d, raport, wypisane, _, _ = _przebieg_pretrain(tmp_path, monkeypatch, capsys)

    linie = wypisane.splitlines()

    assert linie[0] == f"pobrano: {len(RAW) / 1e6:.1f} MB -> {d / 'opus' / 'pl.txt.gz'}"
    assert "przeczytano linii napisow: 142" in linie
    assert "napisy: train 21 linii, val 20 linii, razem 0.0 MB" in linie
    assert f"ksiazki: train {len(raport.train_books)}, val {len(raport.val_books)}" in linie
    assert (
        "odrzucone: angielski=20, krotka=20, przeklenstwa=20, wielkie=20, znaki=20, powtorka=1"
        in linie
    )
    assert sum(1 for x in linie if x.startswith("uwaga: nie osiagnieto celu --target-mb")) == 1
    assert not any(x.startswith("plik istnieje") for x in linie)


def test_bramka_napisy_drugi_przebieg_bez_pobierania(tmp_path, monkeypatch, capsys):
    d, raport, _, otwarcia, _ = _przebieg_pretrain(tmp_path, monkeypatch, capsys)
    nazwy = ("train.txt", "val.txt", "report.json")
    pliki_przed = {n: (d / "pretrain" / n).read_bytes() for n in nazwy}
    assert len(otwarcia) == 1

    raport2 = pretrain.main(["--data-dir", str(d), *PRE_ARGS])
    wypisane2 = capsys.readouterr().out

    assert len(otwarcia) == 1
    komunikat = f"plik istnieje, pomijam pobieranie: {d / 'opus' / 'pl.txt.gz'}"
    assert komunikat in wypisane2.splitlines()
    for n in nazwy:
        assert (d / "pretrain" / n).read_bytes() == pliki_przed[n]
    assert raport2 == raport


def test_bramka_check_size_nic_nie_pobiera(tmp_path, monkeypatch, capsys):
    head_calls: list[str] = []
    otwarcia: list[str] = []

    def fake_head(url):
        head_calls.append(url)
        return 5_000_000

    def fake_open(url):
        otwarcia.append(url)
        raise AssertionError("sieć")

    monkeypatch.setattr(data, "_head_url", fake_head)
    monkeypatch.setattr(data, "_open_url", fake_open)
    d = tmp_path / "d"

    wynik = pretrain.main(["--check-size", "--data-dir", str(d)])
    wypisane = capsys.readouterr().out

    assert wynik is None
    assert wypisane.splitlines() == ["rozmiar pliku: 5 MB", "limit pobierania: 300 MB"]
    assert head_calls == [OPUS_SUBTITLES_PL_URL]
    assert otwarcia == []
    assert not d.exists()
    assert list(tmp_path.iterdir()) == []

    wynik = pretrain.main(["--check-size", "--data-dir", str(d), "--max-download-mb", "50"])
    wypisane = capsys.readouterr().out
    assert wynik is None
    assert wypisane.splitlines() == ["rozmiar pliku: 5 MB", "limit pobierania: 50 MB"]
    assert otwarcia == []
    assert not d.exists()
    assert list(tmp_path.iterdir()) == []

    _ksiazki(d)
    pretrain.main(["--check-size", "--data-dir", str(d)])
    capsys.readouterr()
    assert otwarcia == []
    assert sorted(p.name for p in d.iterdir()) == ["clean"]


def test_bramka_duze_pobranie_bez_zgody_konczy_sie_systemexit(tmp_path, monkeypatch, capsys):
    otwarcia: list[str] = []
    head_calls: list[str] = []

    def fake_open(url):
        otwarcia.append(url)
        raise AssertionError("sieć")

    def fake_head(url):
        head_calls.append(url)
        return 5_000_000

    monkeypatch.setattr(data, "_open_url", fake_open)
    monkeypatch.setattr(data, "_head_url", fake_head)
    d = tmp_path / "d"

    with pytest.raises(SystemExit) as excinfo:
        pretrain.main(["--max-download-mb", "2000", "--data-dir", str(d)])

    assert excinfo.value.code == 2
    assert otwarcia == []
    assert head_calls == []
    assert not d.exists()
    assert "--allow-large" in capsys.readouterr().err

    with pytest.raises(SystemExit) as excinfo2:
        pretrain.main(["--max-download-mb", "1000.5", "--data-dir", str(d)])
    assert excinfo2.value.code == 2
    assert otwarcia == []
    assert not d.exists()
