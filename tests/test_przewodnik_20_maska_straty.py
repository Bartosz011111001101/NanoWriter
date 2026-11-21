"""Testy modułu przewodnik.maska_straty (CPU, malutki kontekst w tmp_path)."""

import dataclasses
import json

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from nanowriter.dataset import encode_dialogs
from nanowriter.model import GPT, GPTConfig
from przewodnik.maska_straty import zrob
from przewodnik.proba import maly_kontekst_ii
from przewodnik.wspolne import ROZMOWA_TEKST, etykieta_tokenu


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    return maly_kontekst_ii(tmp_path_factory.mktemp("maska_straty"))


@pytest.fixture(scope="module")
def pliki(ctx):
    return zrob(ctx)


@pytest.fixture(scope="module")
def dane(pliki):
    return json.loads(pliki[2].read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def kodowanie(ctx):
    return encode_dialogs(ctx.tokenizer_10m, ROZMOWA_TEKST)


@pytest.fixture(scope="module")
def strata_ref(ctx, kodowanie):
    ids, _ = kodowanie
    x = torch.tensor([ids[:-1]])
    y = torch.tensor([ids[1:]])
    with torch.no_grad():
        logity = ctx.model_douczony(x)
    return F.cross_entropy(logity[0], y[0], reduction="none")


@pytest.fixture(scope="module")
def id_nl(ctx):
    return ctx.tokenizer_10m.encode("\n")[0]


def test_pliki_nazwy_kolejnosc_i_format(ctx, pliki):
    assert [p.name for p in pliki] == ["maska_straty.gif", "maska_straty.png", "maska_straty.json"]
    for p in pliki:
        assert p.parent == ctx.wyniki_dir
        assert p.exists()
        assert p.stat().st_size > 0
    assert pliki[0].read_bytes().startswith(b"GIF8")
    assert pliki[1].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    bajty_json = pliki[2].read_bytes()
    assert bajty_json.endswith(b"\n")
    assert b"\r\n" not in bajty_json


def test_json_klucze_i_liczniki(dane, kodowanie):
    ids, maska = kodowanie
    n = len(ids)
    assert dane["material"] == "maska_straty"
    assert dane["rozmowa"] == ROZMOWA_TEKST
    klucze = ("tokeny", "liczone", "wszystkie", "strata_srednia_wszystkie", "strata_srednia_maska")
    for klucz in klucze:
        assert klucz in dane
    tokeny = dane["tokeny"]
    assert dane["wszystkie"] == len(tokeny) == n
    assert dane["liczone"] == sum(t["maska"] for t in tokeny) == sum(maska)
    for t in tokeny:
        assert set(t) == {"i", "token", "maska", "strata", "linia"}
    assert [t["i"] for t in tokeny] == list(range(n))


def test_tokeny_i_maska_zgodne_z_encode_dialogs(ctx, dane, kodowanie):
    ids, maska = kodowanie
    tokeny = dane["tokeny"]
    assert [t["token"] for t in tokeny] == [etykieta_tokenu(ctx.tokenizer_10m, i) for i in ids]
    assert [t["maska"] for t in tokeny] == maska
    assert set(t["maska"] for t in tokeny) <= {0, 1}


def test_struktura_maski_wypowiedzi(dane, kodowanie, id_nl):
    ids, _ = kodowanie
    tokeny = dane["tokeny"]
    linie: list[list[int]] = [[]]
    for i, token_id in enumerate(ids):
        linie[-1].append(i)
        if token_id == id_nl:
            linie.append([])
    assert linie[-1] == []
    linie = linie[:-1]
    assert len(linie) == 4
    for nr, idx in enumerate(linie):
        maski = [tokeny[i]["maska"] for i in idx]
        if nr % 2 == 0:
            # linia A: nic nie jest liczone
            assert maski == [0] * len(idx)
        else:
            # linia B: etykieta "B" i ":" nie są liczone, reszta z końcowym "\n" tak
            assert maski[:2] == [0, 0]
            assert maski[2:] == [1] * (len(idx) - 2)
        assert ids[idx[-1]] == id_nl
        assert tokeny[idx[-1]]["maska"] == (1 if nr % 2 == 1 else 0)


def test_linie(dane, kodowanie, id_nl):
    ids, _ = kodowanie
    tokeny = dane["tokeny"]
    for i, t in enumerate(tokeny):
        assert t["linia"] == sum(1 for x in ids[:i] if x == id_nl)
    assert tokeny[0]["linia"] == 0
    assert tokeny[-1]["linia"] == 3
    assert {t["linia"] for t in tokeny} <= {0, 1, 2, 3}


def test_straty_tokenow_z_modelu(dane, strata_ref):
    tokeny = dane["tokeny"]
    assert tokeny[0]["strata"] is None
    for i in range(1, len(tokeny)):
        s = tokeny[i]["strata"]
        assert s == pytest.approx(float(strata_ref[i - 1]), abs=1e-4)
        assert s >= 0
        assert s < float("inf")
        assert s == s


def test_srednie_straty(dane, kodowanie, strata_ref):
    _, maska = kodowanie
    tokeny = dane["tokeny"]
    wszystkie = [t["strata"] for t in tokeny[1:]]
    z_maska = [t["strata"] for t in tokeny if t["maska"] == 1]
    srednia_wszystkie = sum(wszystkie) / len(wszystkie)
    assert dane["strata_srednia_wszystkie"] == pytest.approx(srednia_wszystkie, abs=1e-4)
    assert dane["strata_srednia_wszystkie"] == pytest.approx(float(strata_ref.mean()), abs=1e-4)
    assert dane["strata_srednia_maska"] == pytest.approx(sum(z_maska) / len(z_maska), abs=1e-4)
    ref_maska = float(strata_ref[torch.tensor(maska[1:], dtype=torch.bool)].mean())
    assert dane["strata_srednia_maska"] == pytest.approx(ref_maska, abs=1e-4)


def test_gif_liczba_klatek(pliki):
    with Image.open(pliki[0]) as gif:
        assert gif.n_frames == 4


def test_za_krotki_block_size_value_error(ctx):
    torch.manual_seed(0)
    maly = GPT(
        GPTConfig(
            vocab_size=ctx.tokenizer_10m.vocab_size,
            block_size=16,
            n_layer=2,
            n_head=2,
            d_model=32,
        )
    ).eval()
    ctx_krotki = dataclasses.replace(ctx, model_douczony=maly)
    with pytest.raises(ValueError):
        zrob(ctx_krotki)


def test_model_bez_zmian(tmp_path):
    ctx = maly_kontekst_ii(tmp_path)
    przed = {k: v.clone() for k, v in ctx.model_douczony.state_dict().items()}
    training_przed = ctx.model_douczony.training
    merges_przed = list(ctx.tokenizer_10m.merges)
    zrob(ctx)
    po = ctx.model_douczony.state_dict()
    assert set(po) == set(przed)
    for k, v in przed.items():
        assert torch.equal(po[k], v)
    assert training_przed is False
    assert ctx.model_douczony.training is False
    assert list(ctx.tokenizer_10m.merges) == merges_przed
