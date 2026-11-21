"""Testy modułu przewodnik.probkowanie."""

import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from przewodnik.proba import maly_kontekst_i
from przewodnik.probkowanie import NAJWIECEJ_SLUPKOW, TEMPERATURY_GIF, zrob
from przewodnik.wspolne import etykieta_tokenu

PREFIKS = "Wieczorem stary kot zasnął przy"
V = 320


def ostry_kontekst(katalog):
    """Kontekst z przeskalowanymi embeddingami (wagi wyjścia współdzielone): ostry rozkład."""
    ctx = maly_kontekst_i(katalog)
    with torch.no_grad():
        ctx.model.tok_emb.weight.mul_(20)
    return ctx


@pytest.fixture(scope="module")
def ostry(tmp_path_factory):
    """Wynik zrob dla kontekstu ostrego i rozkład odniesienia z tego samego modelu."""
    ctx = ostry_kontekst(tmp_path_factory.mktemp("probkowanie"))
    ids = torch.tensor([ctx.tokenizer.encode(PREFIKS)])
    with torch.no_grad():
        logity = ctx.model(ids)[0, -1]
    p_ref = torch.softmax(logity, dim=-1)
    p_ref_sort, id_sort = torch.sort(p_ref, descending=True)
    pliki = zrob(ctx)
    dane = json.loads(pliki[3].read_text(encoding="utf-8"))
    return SimpleNamespace(
        ctx=ctx,
        pliki=pliki,
        dane=dane,
        ustawienia={u["nazwa"]: u for u in dane["ustawienia"]},
        p_ref=p_ref,
        p_ref_sort=p_ref_sort,
        id_sort=id_sort,
    )


def test_stale_modulu():
    assert NAJWIECEJ_SLUPKOW == 12
    assert TEMPERATURY_GIF == (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)


def test_pliki_nazwy_kolejnosc_i_format(ostry):
    pliki = ostry.pliki

    assert [p.name for p in pliki] == [
        "probkowanie.png",
        "probkowanie_temperatura.gif",
        "probkowanie_temperatura.png",
        "probkowanie.json",
    ]
    for p in pliki:
        assert p.parent == ostry.ctx.wyniki_dir
        assert p.exists()
        assert p.stat().st_size > 0
    png = b"\x89PNG\r\n\x1a\n"
    assert pliki[0].read_bytes().startswith(png)
    assert pliki[1].read_bytes().startswith(b"GIF8")
    assert pliki[2].read_bytes().startswith(png)
    surowy = pliki[3].read_bytes()
    assert surowy.endswith(b"\n")
    assert b"\r\n" not in surowy


def test_json_prefiks_i_ustawienia(ostry):
    dane = ostry.dane
    tokenizer = ostry.ctx.tokenizer

    assert dane["material"] == "probkowanie"
    assert dane["prefiks"] == PREFIKS
    assert dane["tokeny_prefiksu"] == [
        etykieta_tokenu(tokenizer, i) for i in tokenizer.encode(PREFIKS)
    ]
    ustawienia = dane["ustawienia"]
    assert len(ustawienia) == 5
    for u in ustawienia:
        klucze = ("nazwa", "temperatura", "top_k", "top_p", "entropia", "liczba_niezerowych", "top")
        for klucz in klucze:
            assert klucz in u
    assert [u["nazwa"] for u in ustawienia] == [
        "temperatura 0.5",
        "temperatura 1.0",
        "temperatura 1.5",
        "top-k = 5",
        "top-p = 0.9",
    ]
    assert [u["temperatura"] for u in ustawienia] == [0.5, 1.0, 1.5, 1.0, 1.0]
    assert [u["top_k"] for u in ustawienia] == [None, None, None, 5, None]
    assert [u["top_p"] for u in ustawienia] == [None, None, None, None, 0.9]


def test_top_wspolna_kolejnosc_i_rozklad_t1(ostry):
    ustawienia = ostry.dane["ustawienia"]
    oczekiwane = [etykieta_tokenu(ostry.ctx.tokenizer, int(i)) for i in ostry.id_sort[:12]]

    for u in ustawienia:
        assert len(u["top"]) == NAJWIECEJ_SLUPKOW
        for x in u["top"]:
            assert "token" in x
            assert "p" in x
        assert [x["token"] for x in u["top"]] == oczekiwane

    p = [x["p"] for x in ostry.ustawienia["temperatura 1.0"]["top"]]
    assert p == pytest.approx(ostry.p_ref_sort[:12].tolist(), abs=1e-6)
    assert all(a >= b for a, b in zip(p, p[1:], strict=False))


def test_entropia_rosnie_z_temperatura(ostry):
    u05 = ostry.ustawienia["temperatura 0.5"]
    u10 = ostry.ustawienia["temperatura 1.0"]
    u15 = ostry.ustawienia["temperatura 1.5"]
    p_ref = ostry.p_ref

    assert u05["entropia"] < u10["entropia"] < u15["entropia"]
    oczekiwana = float(-(p_ref * p_ref.clamp(min=1e-12).log()).sum())
    assert u10["entropia"] == pytest.approx(oczekiwana, abs=1e-4)
    assert u05["top"][0]["p"] > u15["top"][0]["p"]
    for u in (u05, u10, u15):
        assert u["liczba_niezerowych"] == 320


def test_top_k_5(ostry):
    u = ostry.ustawienia["top-k = 5"]
    p = [x["p"] for x in u["top"]]

    assert u["liczba_niezerowych"] == 5
    assert sum(p[:5]) == pytest.approx(1.0, abs=1e-5)
    assert all(x == 0.0 for x in p[5:12])
    assert u["entropia"] < ostry.ustawienia["temperatura 1.0"]["entropia"]


def test_top_p_09(ostry):
    u = ostry.ustawienia["top-p = 0.9"]
    n = u["liczba_niezerowych"]

    assert 1 <= n < 320
    assert float(ostry.p_ref_sort[:n].sum()) >= 0.9 - 1e-6
    assert sum(x["p"] for x in u["top"]) <= 1 + 1e-6
    assert u["entropia"] < ostry.ustawienia["temperatura 1.0"]["entropia"]


def test_przemiatanie(ostry):
    przemiatanie = ostry.dane["przemiatanie"]

    assert len(przemiatanie) == 10
    for w in przemiatanie:
        assert "temperatura" in w
        assert "entropia" in w
    assert [w["temperatura"] for w in przemiatanie] == list(TEMPERATURY_GIF)
    entropie = [w["entropia"] for w in przemiatanie]
    assert all(a < b for a, b in zip(entropie, entropie[1:], strict=False))
    wpis = next(w for w in przemiatanie if w["temperatura"] == 1.0)
    assert wpis["entropia"] == pytest.approx(
        ostry.ustawienia["temperatura 1.0"]["entropia"], abs=1e-6
    )


def test_gif_liczba_klatek(ostry):
    with Image.open(ostry.pliki[1]) as obraz:
        assert obraz.n_frames == 10


def test_model_bez_zmian(tmp_path):
    ctx = ostry_kontekst(tmp_path)
    migawka = {k: v.clone() for k, v in ctx.model.state_dict().items()}

    zrob(ctx)

    po = ctx.model.state_dict()
    assert po.keys() == migawka.keys()
    for k, v in migawka.items():
        assert torch.equal(po[k], v)
    assert ctx.model.training is False
