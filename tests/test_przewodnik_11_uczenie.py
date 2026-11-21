import dataclasses
import json
import math

import pytest
import torch
from PIL import Image

from przewodnik.proba import maly_kontekst_i
from przewodnik.uczenie import LICZBA_TOKENOW, LR, PROMPT, zrob


@pytest.fixture(scope="module")
def wynik12(tmp_path_factory):
    """Jeden przebieg zrob(ctx) dla 12 kroków: (ctx, ścieżki, wczytany JSON)."""
    ctx = maly_kontekst_i(tmp_path_factory.mktemp("uczenie12"))
    sciezki = zrob(ctx)
    dane = json.loads(sciezki[2].read_text(encoding="utf-8"))
    return ctx, sciezki, dane


def _json(sciezki):
    return json.loads(sciezki[2].read_text(encoding="utf-8"))


def test_stale_modulu():
    assert PROMPT == "Wieczorem stary kot"
    assert LICZBA_TOKENOW == 60
    assert LR == 0.002


def test_pliki_nazwy_kolejnosc_i_format(wynik12):
    ctx, sciezki, _ = wynik12

    assert [p.name for p in sciezki] == ["uczenie.gif", "uczenie.png", "uczenie.json"]
    for p in sciezki:
        assert p.parent == ctx.wyniki_dir
        assert p.exists()
        assert p.stat().st_size > 0

    gif, png, js = (p.read_bytes() for p in sciezki)
    assert gif.startswith(b"GIF8")
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert js.endswith(b"\n")
    assert b"\r\n" not in js


def test_json_klucze_i_skalary(wynik12):
    _, _, dane = wynik12

    assert dane["material"] == "uczenie"
    assert dane["kroki"] == 12
    assert dane["batch_size"] == 4
    assert dane["lr"] == 0.002
    assert dane["prompt"] == PROMPT
    assert "probki" in dane
    assert "krzywa" in dane
    for klucz in ("kroki", "train_loss", "val_kroki", "val_loss"):
        assert klucz in dane["krzywa"]
    assert dane["czas_s"] > 0


def test_probki_12_krokow(wynik12):
    _, _, dane = wynik12
    probki = dane["probki"]
    train_loss = dane["krzywa"]["train_loss"]

    assert probki[0]["krok"] == 0
    assert probki[0]["strata"] is None
    assert [p["krok"] for p in probki] == list(range(13))
    for p in probki:
        assert isinstance(p["tekst"], str)
        assert p["tekst"].startswith(PROMPT)
    for k in range(1, 13):
        assert probki[k]["strata"] == pytest.approx(train_loss[k - 1], abs=1e-9)


def test_probki_40_krokow(tmp_path):
    ctx = dataclasses.replace(maly_kontekst_i(tmp_path), kroki_uczenia=40)

    dane = _json(zrob(ctx))
    probki = dane["probki"]
    train_loss = dane["krzywa"]["train_loss"]

    assert [p["krok"] for p in probki] == [0] + list(range(1, 40, 2)) + [40]
    assert len(probki) == 22
    assert len(train_loss) == 40
    assert probki[-1]["strata"] == pytest.approx(train_loss[39], abs=1e-9)


def test_probki_3_kroki_i_gif(tmp_path):
    ctx = dataclasses.replace(maly_kontekst_i(tmp_path), kroki_uczenia=3)

    sciezki = zrob(ctx)
    dane = _json(sciezki)

    assert [p["krok"] for p in dane["probki"]] == [0, 1, 2, 3]
    with Image.open(sciezki[0]) as gif:
        assert gif.n_frames == 4


def test_gif_liczba_klatek_rowna_probkom(wynik12):
    _, sciezki, dane = wynik12

    with Image.open(sciezki[0]) as gif:
        assert gif.n_frames == len(dane["probki"]) == 13


def test_krzywa(wynik12):
    _, _, dane = wynik12
    krzywa = dane["krzywa"]

    assert krzywa["kroki"] == list(range(12))
    assert len(krzywa["train_loss"]) == 12
    assert all(math.isfinite(x) for x in krzywa["train_loss"])
    assert len(krzywa["val_kroki"]) == len(krzywa["val_loss"]) >= 1
    assert krzywa["val_kroki"][-1] == 11


def test_strata_startowa_bliska_ln_v(wynik12):
    _, _, dane = wynik12

    start = dane["krzywa"]["train_loss"][0]
    assert 0.9 * math.log(320) <= start <= 1.1 * math.log(320)


def test_powtarzalnosc_i_inny_seed(tmp_path):
    dane_a = _json(zrob(maly_kontekst_i(tmp_path / "a")))
    dane_b = _json(zrob(maly_kontekst_i(tmp_path / "b")))
    dane_s1 = _json(zrob(maly_kontekst_i(tmp_path / "s1", seed=1)))

    assert dane_a["krzywa"]["train_loss"] == dane_b["krzywa"]["train_loss"]
    assert [p["tekst"] for p in dane_a["probki"]] == [p["tekst"] for p in dane_b["probki"]]
    assert dane_s1["krzywa"]["train_loss"] != dane_a["krzywa"]["train_loss"]


def test_model_kontekstu_bez_zmian(tmp_path):
    ctx = maly_kontekst_i(tmp_path)
    przed = {k: v.clone() for k, v in ctx.model.state_dict().items()}

    zrob(ctx)

    po = ctx.model.state_dict()
    assert set(po) == set(przed)
    for k, v in przed.items():
        assert torch.equal(po[k], v)
    assert ctx.model.training is False
