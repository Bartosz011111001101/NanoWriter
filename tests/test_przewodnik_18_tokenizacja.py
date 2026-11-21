"""Testy materiału przewodnik.tokenizacja (CPU, malutki kontekst części II)."""

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from nanowriter.tokenizer import BPETokenizer
from przewodnik.konteksty import KontekstII
from przewodnik.proba import maly_kontekst_ii
from przewodnik.tokenizacja import ZDANIA, zrob
from przewodnik.wspolne import ZDANIE, bajty_tokenu, etykieta_tokenu

ZDANIE_3 = "Zażółć gęślą jaźń."


class Zly:
    """Atrapa tokenizera z błędnym decode."""

    def __init__(self, t: BPETokenizer) -> None:
        self._t = t
        self.vocab_size = t.vocab_size
        self.merges = t.merges

    def encode(self, tekst: str) -> list[int]:
        return self._t.encode(tekst)

    def decode(self, ids: list[int]) -> str:
        return "zle"


def _kontekst_z_roznymi_tokenizerami(katalog: Path) -> KontekstII:
    ctx0 = maly_kontekst_ii(katalog)
    return dataclasses.replace(ctx0, tokenizer_10m=BPETokenizer([]))


@pytest.fixture(scope="module")
def ctx(tmp_path_factory: pytest.TempPathFactory) -> KontekstII:
    return _kontekst_z_roznymi_tokenizerami(tmp_path_factory.mktemp("tokenizacja"))


@pytest.fixture(scope="module")
def pliki(ctx: KontekstII) -> list[Path]:
    return zrob(ctx)


@pytest.fixture(scope="module")
def dane(pliki: list[Path]) -> dict[str, Any]:
    wynik: dict[str, Any] = json.loads(pliki[1].read_text(encoding="utf-8"))
    return wynik


def test_stala_zdania() -> None:
    assert ZDANIA == (ZDANIE, "Dzień dobry, jak minął ci weekend?", ZDANIE_3)


def test_pliki_nazwy_kolejnosc_i_format(ctx: KontekstII, pliki: list[Path]) -> None:
    assert [p.name for p in pliki] == ["tokenizacja.png", "tokenizacja.json"]
    for p in pliki:
        assert p.parent == ctx.wyniki_dir
        assert p.exists()
        assert p.stat().st_size > 0
    assert pliki[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    surowe = pliki[1].read_bytes()
    assert surowe.endswith(b"\n")
    assert b"\r\n" not in surowe


def test_json_struktura(dane: dict[str, Any]) -> None:
    assert dane["material"] == "tokenizacja"
    assert len(dane["zdania"]) == 3
    assert [z["zdanie"] for z in dane["zdania"]] == list(ZDANIA)
    for wpis in dane["zdania"]:
        assert "zdanie" in wpis
        assert "slownik_1m" in wpis
        assert "slownik_10m" in wpis
        for klucz in ("slownik_1m", "slownik_10m"):
            assert set(wpis[klucz]) == {"vocab_size", "tokeny", "id", "liczba", "znakow_na_token"}


def test_slowniki_nie_pomylone(ctx: KontekstII, dane: dict[str, Any]) -> None:
    for wpis in dane["zdania"]:
        zdanie = wpis["zdanie"]
        assert wpis["slownik_1m"]["vocab_size"] == 320
        assert wpis["slownik_1m"]["id"] == ctx.tokenizer_1m.encode(zdanie)
        assert wpis["slownik_10m"]["vocab_size"] == 256
        assert wpis["slownik_10m"]["id"] == ctx.tokenizer_10m.encode(zdanie)
        assert wpis["slownik_10m"]["id"] == list(zdanie.encode("utf-8"))
    pierwszy = dane["zdania"][0]
    assert pierwszy["slownik_1m"]["liczba"] < pierwszy["slownik_10m"]["liczba"]


def test_liczby_i_etykiety(ctx: KontekstII, dane: dict[str, Any]) -> None:
    tokenizery = {"slownik_1m": ctx.tokenizer_1m, "slownik_10m": ctx.tokenizer_10m}
    for wpis in dane["zdania"]:
        zdanie = wpis["zdanie"]
        for klucz, tokenizer in tokenizery.items():
            slownik = wpis[klucz]
            ids = slownik["id"]
            assert slownik["liczba"] == len(ids) == len(slownik["tokeny"])
            assert slownik["tokeny"] == [etykieta_tokenu(tokenizer, i) for i in ids]
            assert all("\ufffd" not in etykieta for etykieta in slownik["tokeny"])
            assert slownik["znakow_na_token"] == pytest.approx(
                len(zdanie) / slownik["liczba"], rel=1e-12
            )
            assert b"".join(bajty_tokenu(tokenizer, i) for i in ids) == zdanie.encode("utf-8")
            assert tokenizer.decode(ids) == zdanie


def test_tokenizer_bez_scalen_liczy_bajty(dane: dict[str, Any]) -> None:
    slownik = dane["zdania"][2]["slownik_10m"]
    assert dane["zdania"][2]["zdanie"] == ZDANIE_3
    assert slownik["liczba"] == 27
    assert slownik["id"] == list(ZDANIE_3.encode("utf-8"))
    assert slownik["znakow_na_token"] == pytest.approx(18 / 27, rel=1e-12)
    assert any(etykieta.startswith("\\x") for etykieta in slownik["tokeny"])


def test_zly_decode_runtime_error(tmp_path: Path) -> None:
    ctx = _kontekst_z_roznymi_tokenizerami(tmp_path)
    ctx_1m = dataclasses.replace(ctx, tokenizer_1m=Zly(ctx.tokenizer_1m))
    with pytest.raises(RuntimeError):
        zrob(ctx_1m)
    ctx_10m = dataclasses.replace(ctx, tokenizer_10m=Zly(ctx.tokenizer_10m))
    with pytest.raises(RuntimeError):
        zrob(ctx_10m)


def test_tokenizery_bez_zmian(tmp_path: Path) -> None:
    ctx = _kontekst_z_roznymi_tokenizerami(tmp_path)
    przed_1m = list(ctx.tokenizer_1m.merges)
    przed_10m = list(ctx.tokenizer_10m.merges)
    zrob(ctx)
    assert list(ctx.tokenizer_1m.merges) == przed_1m
    assert list(ctx.tokenizer_10m.merges) == przed_10m
