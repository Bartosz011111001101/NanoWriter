import json
import math
from types import SimpleNamespace

import pytest
import torch

from przewodnik.embeddingi import zrob
from przewodnik.proba import maly_kontekst_i
from przewodnik.wspolne import ZDANIE, etykieta_tokenu


def _pca_odniesienia(ctx):
    """Niezależne PCA z wag przed wywołaniem zrob: (wariancja dwóch składowych, rzut [V, 2])."""
    wagi = ctx.model.tok_emb.weight.detach().float().cpu()
    wc = wagi - wagi.mean(dim=0)
    _, s, vh = torch.linalg.svd(wc, full_matrices=False)
    wariancja = (s[:2] ** 2 / (s**2).sum()).tolist()
    return wariancja, wc @ vh[:2].T


@pytest.fixture(scope="module")
def wynik(tmp_path_factory):
    ctx = maly_kontekst_i(tmp_path_factory.mktemp("embeddingi"))
    wariancja_ref, xy_ref = _pca_odniesienia(ctx)
    sciezki = zrob(ctx)
    json_sciezka = ctx.wyniki_dir / "embeddingi.json"
    dane = json.loads(json_sciezka.read_text(encoding="utf-8"))
    return SimpleNamespace(
        ctx=ctx, sciezki=sciezki, dane=dane, wariancja_ref=wariancja_ref, xy_ref=xy_ref
    )


def _licznosc(ctx):
    return torch.bincount(ctx.dane_train, minlength=ctx.model.config.vocab_size)


def _ids_zdania(ctx):
    return set(ctx.tokenizer.encode(ZDANIE))


def test_pliki_nazwy_kolejnosc_i_format(wynik):
    assert [p.name for p in wynik.sciezki] == ["embeddingi.png", "embeddingi.json"]
    for sciezka in wynik.sciezki:
        assert sciezka.parent == wynik.ctx.wyniki_dir
        assert sciezka.exists()
        assert sciezka.stat().st_size > 0

    png = wynik.sciezki[0].read_bytes()
    assert png.startswith(b"\x89PNG\r\n\x1a\n")

    surowy_json = wynik.sciezki[1].read_bytes()
    assert surowy_json.endswith(b"\n")
    assert b"\r\n" not in surowy_json


def test_json_klucze(wynik):
    dane = wynik.dane
    assert dane["material"] == "embeddingi"
    assert dane["zdanie"] == ZDANIE
    assert dane["vocab_size"] == wynik.ctx.model.config.vocab_size == 320
    assert dane["d_model"] == 32
    for klucz in ("wariancja_wyjasniona", "etykietowane", "tokeny"):
        assert klucz in dane
    for token in dane["tokeny"]:
        assert set(token.keys()) == {"id", "token", "czestosc", "x", "y", "w_zdaniu"}


def test_wariancja_wyjasniona(wynik):
    w = wynik.dane["wariancja_wyjasniona"]
    assert len(w) == 2
    assert w[0] >= w[1] > 0
    assert w[0] <= 1
    assert w[1] <= 1
    assert sum(w) <= 1 + 1e-6
    assert torch.allclose(torch.tensor(w), torch.tensor(wynik.wariancja_ref), atol=1e-4)


def test_wspolrzedne_pca_do_znaku(wynik):
    liczba = wynik.ctx.model.config.vocab_size
    ids = [t["id"] for t in wynik.dane["tokeny"]]
    for token in wynik.dane["tokeny"]:
        assert math.isfinite(token["x"])
        assert math.isfinite(token["y"])
    for k, klucz in enumerate("xy"):
        wektor = torch.zeros(liczba)
        for token in wynik.dane["tokeny"]:
            wektor[token["id"]] = token[klucz]
        wartosci = wektor[ids]
        odniesienie = wynik.xy_ref[ids, k]
        # Jeden wspólny znak dla całego wektora.
        assert torch.allclose(wartosci, odniesienie, atol=1e-3) or torch.allclose(
            wartosci, -odniesienie, atol=1e-3
        ), klucz


def test_skład_listy_tokenow(wynik):
    tokeny = wynik.dane["tokeny"]
    licznosc = _licznosc(wynik.ctx)
    ids = [t["id"] for t in tokeny]

    assert len(set(ids)) == len(ids)
    assert len(tokeny) >= 200
    for token in tokeny:
        assert token["czestosc"] == int(licznosc[token["id"]])
        assert token["token"] == etykieta_tokenu(wynik.ctx.tokenizer, token["id"])

    pierwsze = [t["czestosc"] for t in tokeny[:200]]
    assert all(a >= b for a, b in zip(pierwsze, pierwsze[1:], strict=False))
    assert pierwsze == sorted(licznosc.tolist(), reverse=True)[:200]
    assert tokeny[0]["czestosc"] == int(licznosc.max())


def test_w_zdaniu(wynik):
    tokeny = wynik.dane["tokeny"]
    ids_zdania = _ids_zdania(wynik.ctx)
    po_id = {t["id"]: t for t in tokeny}

    for token_id in ids_zdania:
        assert token_id in po_id
        assert po_id[token_id]["w_zdaniu"] is True
    for token in tokeny:
        if token["id"] not in ids_zdania:
            assert token["w_zdaniu"] is False
    for token in tokeny[200:]:
        assert token["w_zdaniu"] is True


def test_etykietowane(wynik):
    tokeny = wynik.dane["tokeny"]
    etykietowane = wynik.dane["etykietowane"]
    ids_zdania = _ids_zdania(wynik.ctx)
    pierwsze_40 = {t["id"] for t in tokeny[:40]}

    assert len(set(etykietowane)) == len(etykietowane)
    assert pierwsze_40 <= set(etykietowane)
    assert ids_zdania <= set(etykietowane)
    assert set(etykietowane) == pierwsze_40 | ids_zdania
    assert set(etykietowane) <= {t["id"] for t in tokeny}


def test_rzad_jeden(tmp_path):
    ctx = maly_kontekst_i(tmp_path)
    liczba = ctx.model.config.vocab_size
    torch.manual_seed(0)
    with torch.no_grad():
        ctx.model.tok_emb.weight.copy_(torch.outer(torch.randn(liczba), torch.randn(32)))

    zrob(ctx)
    dane = json.loads((ctx.wyniki_dir / "embeddingi.json").read_text(encoding="utf-8"))

    assert dane["wariancja_wyjasniona"][0] == pytest.approx(1.0, abs=1e-4)
    assert dane["wariancja_wyjasniona"][1] == pytest.approx(0.0, abs=1e-4)


def test_wagi_bez_zmian(tmp_path):
    ctx = maly_kontekst_i(tmp_path)
    migawka = {nazwa: t.clone() for nazwa, t in ctx.model.state_dict().items()}
    training_przed = ctx.model.training

    zrob(ctx)

    po = ctx.model.state_dict()
    assert set(po.keys()) == set(migawka.keys())
    for nazwa, tensor in migawka.items():
        assert torch.equal(po[nazwa], tensor), nazwa
    assert training_przed is False
    assert ctx.model.training is False
