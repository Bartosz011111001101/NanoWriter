import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from nanowriter.tokenizer import BPETokenizer
from przewodnik.bpe import zrob
from przewodnik.proba import maly_kontekst_i
from przewodnik.wspolne import ZDANIE, bajty_tokenu, etykieta_tokenu

BAJTY_ZDANIA = list(ZDANIE.encode("utf-8"))


@pytest.fixture(scope="module")
def ctx(tmp_path_factory: pytest.TempPathFactory) -> Any:
    return maly_kontekst_i(tmp_path_factory.mktemp("bpe_ctx"))


@pytest.fixture(scope="module")
def merges(ctx: Any) -> list[tuple[int, int]]:
    return list(ctx.tokenizer.merges)


@pytest.fixture(scope="module")
def pliki(ctx: Any) -> list[Path]:
    return zrob(ctx)


@pytest.fixture(scope="module")
def dane(pliki: list[Path]) -> dict[str, Any]:
    return json.loads(pliki[2].read_text(encoding="utf-8"))


def test_pliki_nazwy_kolejnosc_i_format(ctx: Any, pliki: list[Path]) -> None:
    assert [p.name for p in pliki] == ["bpe.gif", "bpe.png", "bpe.json"]
    for p in pliki:
        assert p.parent == ctx.wyniki_dir
        assert p.exists()
        assert p.stat().st_size > 0
    assert pliki[0].read_bytes().startswith(b"GIF8")
    assert pliki[1].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    surowy = pliki[2].read_bytes()
    assert surowy.endswith(b"\n")
    assert b"\r\n" not in surowy


def test_json_klucze(dane: dict[str, Any]) -> None:
    assert dane["material"] == "bpe"
    assert dane["zdanie"] == ZDANIE
    assert dane["bajty"] == BAJTY_ZDANIA
    for klucz in ("kroki", "tokeny_koncowe", "liczba_scalen"):
        assert klucz in dane
    for krok in dane["kroki"]:
        for klucz in ("nr", "ranga", "para", "nowy_token", "tokeny", "etykiety"):
            assert klucz in krok


def test_krok_poczatkowy(ctx: Any, dane: dict[str, Any]) -> None:
    krok = dane["kroki"][0]
    assert krok["nr"] == 0
    assert krok["ranga"] is None
    assert krok["para"] is None
    assert krok["nowy_token"] is None
    assert krok["tokeny"] == BAJTY_ZDANIA
    assert krok["etykiety"] == [etykieta_tokenu(ctx.tokenizer, b) for b in BAJTY_ZDANIA]


def test_kroki_kolejne_zgodne_z_tokenizerem(
    ctx: Any, merges: list[tuple[int, int]], dane: dict[str, Any]
) -> None:
    kroki = dane["kroki"]
    poprzednia_ranga = -1
    for k in range(1, len(kroki)):
        krok = kroki[k]
        assert krok["nr"] == k
        ranga = krok["ranga"]
        assert isinstance(ranga, int)
        assert ranga > poprzednia_ranga
        poprzednia_ranga = ranga
        assert krok["nowy_token"] == 256 + ranga
        assert krok["para"] == list(merges[ranga])
        assert len(krok["tokeny"]) < len(kroki[k - 1]["tokeny"])
        assert krok["tokeny"] == BPETokenizer(merges[: ranga + 1]).encode(ZDANIE)
        assert krok["etykiety"] == [etykieta_tokenu(ctx.tokenizer, i) for i in krok["tokeny"]]
        assert krok["nowy_token"] in krok["tokeny"]


def test_bajty_zachowane_w_kazdym_kroku(ctx: Any, dane: dict[str, Any]) -> None:
    for krok in dane["kroki"]:
        zlozone = b"".join(bajty_tokenu(ctx.tokenizer, i) for i in krok["tokeny"])
        assert zlozone == ZDANIE.encode("utf-8")


def test_stan_koncowy(ctx: Any, merges: list[tuple[int, int]], dane: dict[str, Any]) -> None:
    kroki = dane["kroki"]
    assert dane["tokeny_koncowe"] == ctx.tokenizer.encode(ZDANIE)
    assert dane["tokeny_koncowe"] == kroki[-1]["tokeny"]
    assert dane["liczba_scalen"] == len(kroki) - 1
    assert 1 <= dane["liczba_scalen"] <= len(merges)


def test_etykiety_bez_znaku_zastepczego(dane: dict[str, Any]) -> None:
    for krok in dane["kroki"]:
        for etykieta in krok["etykiety"]:
            assert "�" not in etykieta


def test_gif_liczba_klatek(pliki: list[Path], dane: dict[str, Any]) -> None:
    with Image.open(pliki[0]) as obraz:
        assert obraz.n_frames == len(dane["kroki"])


def test_bez_scalen(ctx: Any, tmp_path: Path) -> None:
    ctx2 = dataclasses.replace(ctx, tokenizer=BPETokenizer([]), wyniki_dir=tmp_path)
    wynik = zrob(ctx2)
    dane = json.loads(wynik[2].read_text(encoding="utf-8"))
    assert len(dane["kroki"]) == 1
    assert dane["liczba_scalen"] == 0
    assert dane["tokeny_koncowe"] == BAJTY_ZDANIA
    assert wynik[0].exists()
    with Image.open(wynik[0]) as obraz:
        assert obraz.n_frames == 1


def test_scalenie_bez_efektu_nie_daje_klatki(ctx: Any, tmp_path: Path) -> None:
    scalenia = [(97, 98), (115, 116)]
    ctx2 = dataclasses.replace(ctx, tokenizer=BPETokenizer(scalenia), wyniki_dir=tmp_path)
    wynik = zrob(ctx2)
    dane = json.loads(wynik[2].read_text(encoding="utf-8"))
    kroki = dane["kroki"]
    assert len(kroki) == 2
    assert kroki[1]["nr"] == 1
    assert kroki[1]["ranga"] == 1
    assert kroki[1]["para"] == [115, 116]
    assert kroki[1]["nowy_token"] == 257
    assert len(kroki[1]["tokeny"]) == 39
    assert dane["liczba_scalen"] == 1
    assert dane["tokeny_koncowe"] == BPETokenizer(scalenia).encode(ZDANIE)
    assert "st" in kroki[1]["etykiety"]


def test_niezgodny_stan_koncowy_runtime_error(
    ctx: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx2 = dataclasses.replace(ctx, wyniki_dir=tmp_path)
    monkeypatch.setattr(ctx.tokenizer, "encode", lambda tekst: [1, 2, 3])
    with pytest.raises(RuntimeError):
        zrob(ctx2)


def test_tokenizer_i_model_bez_zmian(tmp_path: Path) -> None:
    ctx = maly_kontekst_i(tmp_path)
    przed = list(ctx.tokenizer.merges)
    zrob(ctx)
    assert list(ctx.tokenizer.merges) == przed
