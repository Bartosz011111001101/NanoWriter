"""Testy materiału przewodnik.nastepny_token (top-10 następnego tokenu dla zdania)."""

import json

import pytest
import torch
from PIL import Image

from przewodnik.nastepny_token import zrob
from przewodnik.proba import maly_kontekst_i
from przewodnik.wspolne import ZDANIE, etykieta_tokenu, zakoduj_zdanie

NAZWY = ["nastepny_token.gif", "nastepny_token.png", "nastepny_token.json"]
PNG_NAGLOWEK = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def wynik(tmp_path_factory):
    """Kontekst, jego pliki z zrob i liczby odniesienia liczone niezależnie z modelu."""
    ctx = maly_kontekst_i(tmp_path_factory.mktemp("nt"))
    pliki = zrob(ctx)
    ids = zakoduj_zdanie(ctx.tokenizer, ZDANIE, ctx.model.config.block_size)
    with torch.no_grad():
        logity = ctx.model(ids)[0]
    return {
        "ctx": ctx,
        "pliki": pliki,
        "ids": ids,
        "prob": torch.softmax(logity, dim=-1),
        "tokeny_ref": [etykieta_tokenu(ctx.tokenizer, int(i)) for i in ids[0]],
        "dane": json.loads(pliki[2].read_text(encoding="utf-8")),
    }


def test_pliki_nazwy_kolejnosc_i_format(wynik):
    pliki = wynik["pliki"]
    assert len(pliki) == 3
    assert [p.name for p in pliki] == NAZWY
    for p in pliki:
        assert p.parent == wynik["ctx"].wyniki_dir
        assert p.exists()
        assert p.stat().st_size > 0
    assert pliki[0].read_bytes().startswith(b"GIF8")
    assert pliki[1].read_bytes().startswith(PNG_NAGLOWEK)
    bajty_json = pliki[2].read_bytes()
    bajty_json.decode("utf-8")
    assert bajty_json.endswith(b"\n")
    assert b"\r\n" not in bajty_json


def test_json_klucze_i_tokeny(wynik):
    dane = wynik["dane"]
    for klucz in ("material", "zdanie", "tokeny", "pozycje"):
        assert klucz in dane
    assert dane["material"] == "nastepny_token"
    assert dane["zdanie"] == ZDANIE
    assert dane["tokeny"] == wynik["tokeny_ref"]
    assert len(dane["pozycje"]) == wynik["ids"].shape[1]


def test_top10_struktura(wynik):
    for poz in wynik["dane"]["pozycje"]:
        top10 = poz["top10"]
        assert len(top10) == 10
        for el in top10:
            assert set(el.keys()) == {"id", "token", "p"}
        p = [el["p"] for el in top10]
        assert all(a >= b for a, b in zip(p, p[1:], strict=False))
        assert all(0 < x <= 1 for x in p)
        assert 0 < sum(p) <= 1 + 1e-6


def test_pozycje_pola(wynik):
    T = wynik["ids"].shape[1]
    pozycje = wynik["dane"]["pozycje"]
    tokeny_ref = wynik["tokeny_ref"]
    for t in range(T):
        poz = pozycje[t]
        assert poz["pozycja"] == t
        assert poz["kontekst"] == tokeny_ref[: t + 1]
        assert len(poz["kontekst"]) == t + 1
        if t < T - 1:
            assert poz["prawdziwy_nastepny"] == tokeny_ref[t + 1]
            assert 0 < poz["p_prawdziwego"] <= 1
            assert isinstance(poz["ranga_prawdziwego"], int)
            assert poz["ranga_prawdziwego"] >= 1
        else:
            assert poz["prawdziwy_nastepny"] is None
            assert poz["p_prawdziwego"] is None
            assert poz["ranga_prawdziwego"] is None


def test_liczby_zgodne_z_modelem(wynik):
    ctx = wynik["ctx"]
    ids = wynik["ids"]
    prob = wynik["prob"]
    T = ids.shape[1]
    pozycje = wynik["dane"]["pozycje"]
    for t in range(T):
        top10 = pozycje[t]["top10"]
        assert top10[0]["id"] == int(prob[t].argmax())
        oczekiwane = torch.topk(prob[t], 10).values
        zapisane = torch.tensor([el["p"] for el in top10])
        assert torch.allclose(zapisane, oczekiwane, atol=1e-5)
        for el in top10:
            assert el["token"] == etykieta_tokenu(ctx.tokenizer, el["id"])
        if t < T - 1:
            id_prawdz = int(ids[0, t + 1])
            assert pozycje[t]["p_prawdziwego"] == pytest.approx(
                float(prob[t, id_prawdz]), abs=1e-6
            )
            assert pozycje[t]["ranga_prawdziwego"] == 1 + int(
                (prob[t] > prob[t, id_prawdz]).sum()
            )


def test_p_prawdziwego_zgodne_z_top10(wynik):
    ids = wynik["ids"]
    T = ids.shape[1]
    pozycje = wynik["dane"]["pozycje"]
    for t in range(T - 1):
        id_prawdz = int(ids[0, t + 1])
        top10 = pozycje[t]["top10"]
        id_w_top10 = [el["id"] for el in top10]
        if id_prawdz in id_w_top10:
            el = top10[id_w_top10.index(id_prawdz)]
            assert el["p"] == pytest.approx(pozycje[t]["p_prawdziwego"], abs=1e-6)
        if pozycje[t]["ranga_prawdziwego"] <= 10:
            assert id_prawdz in id_w_top10


def test_gif_liczba_klatek(wynik):
    T = wynik["ids"].shape[1]
    with Image.open(wynik["pliki"][0]) as gif:
        assert gif.n_frames == T


def test_model_bez_zmian(tmp_path):
    ctx = maly_kontekst_i(tmp_path)
    migawka = {k: v.clone() for k, v in ctx.model.state_dict().items()}
    zrob(ctx)
    po = ctx.model.state_dict()
    assert set(po.keys()) == set(migawka.keys())
    for k, v in migawka.items():
        assert torch.equal(po[k], v), k
    assert ctx.model.training is False
    for blok in ctx.model.blocks:
        for modul in (blok, blok.attn, blok.mlp):
            assert len(modul._forward_hooks) == 0
            assert len(modul._forward_pre_hooks) == 0
        assert blok.attn.trace is None


def test_powtarzalnosc(tmp_path):
    a = zrob(maly_kontekst_i(tmp_path / "a"))[2].read_bytes()
    b = zrob(maly_kontekst_i(tmp_path / "b"))[2].read_bytes()
    assert a == b
    c = zrob(maly_kontekst_i(tmp_path / "c", seed=1))[2].read_bytes()
    assert c != a
