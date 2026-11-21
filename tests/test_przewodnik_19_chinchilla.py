"""Testy modułu przewodnik.chinchilla."""

import dataclasses
import json

import pytest

from przewodnik.chinchilla import TOKENOW_NA_PARAMETR, zrob
from przewodnik.proba import maly_kontekst_ii

PNG_NAGLOWEK = b"\x89PNG\r\n\x1a\n"
NAZWY_PRZEBIEGOW = ["przebieg_1m", "przebieg_wstepny", "przebieg_douczania"]
KLUCZE_MODELU = {
    "nazwa",
    "parametry",
    "tokeny_treningu",
    "tokenow_na_parametr",
    "optimum_tokenow",
    "tokenow_korpusu",
    "epoki",
}


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    return maly_kontekst_ii(tmp_path_factory.mktemp("chinchilla"))


@pytest.fixture(scope="module")
def pliki(ctx):
    return zrob(ctx)


@pytest.fixture(scope="module")
def dane(pliki):
    return json.loads(pliki[1].read_text(encoding="utf-8"))


def test_stala():
    assert TOKENOW_NA_PARAMETR == 20


def test_pliki_nazwy_kolejnosc_i_format(ctx, pliki):
    assert [p.name for p in pliki] == ["chinchilla.png", "chinchilla.json"]
    for plik in pliki:
        assert plik.parent == ctx.wyniki_dir
        assert plik.exists()
        assert plik.stat().st_size > 0

    assert pliki[0].read_bytes().startswith(PNG_NAGLOWEK)
    tresc_json = pliki[1].read_bytes()
    assert tresc_json.endswith(b"\n")
    assert b"\r\n" not in tresc_json


def test_json_modele_i_tokeny_treningu(ctx, dane):
    assert dane["material"] == "chinchilla"
    assert dane["tokenow_na_parametr_chinchilla"] == 20

    modele = dane["modele"]
    assert len(modele) == 3
    assert [m["nazwa"] for m in modele] == ["1M", "10M wstępny", "10M douczony"]
    for m in modele:
        assert set(m.keys()) == KLUCZE_MODELU

    assert [m["tokeny_treningu"] for m in modele] == [40960, 40960, 10240]
    for m in modele:
        assert type(m["tokeny_treningu"]) is int

    for m, nazwa in zip(modele, NAZWY_PRZEBIEGOW, strict=True):
        przebieg = getattr(ctx, nazwa)
        assert m["parametry"] == przebieg.parametry
        assert m["tokenow_korpusu"] == przebieg.tokenow_korpusu


def test_wzory(dane):
    for m in dane["modele"]:
        assert m["tokenow_na_parametr"] == pytest.approx(
            m["tokeny_treningu"] / m["parametry"], rel=1e-12
        )
        assert m["optimum_tokenow"] == 20 * m["parametry"]
        assert type(m["optimum_tokenow"]) is int
        assert m["epoki"] == pytest.approx(
            m["tokeny_treningu"] / m["tokenow_korpusu"], rel=1e-12
        )


def test_przyklad_z_opisu(tmp_path):
    ctx = maly_kontekst_ii(tmp_path)
    ctx = dataclasses.replace(
        ctx,
        przebieg_wstepny=dataclasses.replace(
            ctx.przebieg_wstepny,
            kroki=12000,
            batch_size=64,
            block_size=256,
            parametry=10101760,
        ),
    )

    pliki = zrob(ctx)
    modele = json.loads(pliki[1].read_text(encoding="utf-8"))["modele"]

    assert modele[1]["tokeny_treningu"] == 196608000
    assert modele[1]["tokenow_na_parametr"] == pytest.approx(19.46, abs=0.01)
    assert modele[1]["optimum_tokenow"] == 202035200


@pytest.mark.parametrize("nazwa", NAZWY_PRZEBIEGOW)
def test_zerowe_parametry_value_error(tmp_path, nazwa):
    ctx = maly_kontekst_ii(tmp_path)
    ctx = dataclasses.replace(
        ctx, **{nazwa: dataclasses.replace(getattr(ctx, nazwa), parametry=0)}
    )

    with pytest.raises(ValueError):
        zrob(ctx)


@pytest.mark.parametrize("nazwa", NAZWY_PRZEBIEGOW)
def test_zerowe_tokeny_korpusu_value_error(tmp_path, nazwa):
    ctx = maly_kontekst_ii(tmp_path)
    ctx = dataclasses.replace(
        ctx, **{nazwa: dataclasses.replace(getattr(ctx, nazwa), tokenow_korpusu=0)}
    )

    with pytest.raises(ValueError):
        zrob(ctx)
