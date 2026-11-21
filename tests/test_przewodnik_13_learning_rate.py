"""Testy modułu przewodnik.learning_rate."""

import dataclasses
import json
import math

import pytest
import torch

from przewodnik.krotki_trening import krotki_trening as prawdziwy_trening
from przewodnik.learning_rate import LICZBA_TOKENOW, PROMPT, PRZEBIEGI, zrob
from przewodnik.proba import maly_kontekst_i

NAZWY = ["za mały", "dobry", "za duży"]
KOMUNIKAT = "(generowanie niemożliwe: model rozbieżny)"


def wczytaj_przebiegi(pliki) -> dict:
    """Wczytuje JSON z listy zwróconej przez zrob."""
    return json.loads(pliki[1].read_text(encoding="utf-8"))


def atrapa_z_stratami(monkeypatch: pytest.MonkeyPatch, straty: list[float]) -> None:
    """Podstawia trening, który uczy naprawdę, ale zwraca zadaną listę strat."""

    def falszywy(model, *args, **kwargs):
        wynik = prawdziwy_trening(model, *args, **kwargs)
        return dataclasses.replace(wynik, train_loss=list(straty))

    monkeypatch.setattr("przewodnik.learning_rate.krotki_trening", falszywy)


def test_stale_modulu():
    assert PRZEBIEGI == (("za mały", 1e-5), ("dobry", 2e-3), ("za duży", 1e-1))
    assert PROMPT == "Wieczorem stary kot"
    assert LICZBA_TOKENOW == 40


def test_pliki_nazwy_kolejnosc_i_format(tmp_path):
    ctx = maly_kontekst_i(tmp_path)
    pliki = zrob(ctx)
    assert [p.name for p in pliki] == ["learning_rate.png", "learning_rate.json"]
    for plik in pliki:
        assert plik.parent == ctx.wyniki_dir
        assert plik.exists()
        assert plik.stat().st_size > 0
    assert pliki[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    surowy = pliki[1].read_bytes()
    assert surowy.endswith(b"\n")
    assert b"\r\n" not in surowy


def test_json_struktura(tmp_path):
    ctx = maly_kontekst_i(tmp_path)
    dane = wczytaj_przebiegi(zrob(ctx))
    assert dane["material"] == "learning_rate"
    assert dane["kroki"] == 6
    przebiegi = dane["przebiegi"]
    assert len(przebiegi) == 3
    assert [p["nazwa"] for p in przebiegi] == NAZWY
    assert [p["lr"] for p in przebiegi] == pytest.approx([1e-05, 0.002, 0.1])
    klucze = {
        "nazwa",
        "lr",
        "train_loss",
        "val_kroki",
        "val_loss",
        "koncowa_strata",
        "tekst",
        "czas_s",
    }
    for p in przebiegi:
        assert klucze <= set(p)
        assert len(p["train_loss"]) == 6
        assert p["czas_s"] > 0
        assert p["tekst"].startswith(PROMPT) or p["tekst"] == KOMUNIKAT


def test_ten_sam_start(tmp_path):
    ctx = maly_kontekst_i(tmp_path)
    przebiegi = wczytaj_przebiegi(zrob(ctx))["przebiegi"]
    starty = [p["train_loss"][0] for p in przebiegi]
    assert starty[1] == pytest.approx(starty[0], abs=1e-6)
    assert starty[2] == pytest.approx(starty[0], abs=1e-6)
    ln_v = math.log(320)
    assert 0.9 * ln_v <= starty[0] <= 1.1 * ln_v


def test_koncowa_strata_to_srednia_ostatnich_10(tmp_path):
    ctx = maly_kontekst_i(tmp_path / "a")
    for p in wczytaj_przebiegi(zrob(ctx))["przebiegi"]:
        srednia = sum(p["train_loss"]) / len(p["train_loss"])
        assert p["koncowa_strata"] == pytest.approx(srednia, abs=1e-6)

    ctx60 = dataclasses.replace(maly_kontekst_i(tmp_path / "b"), kroki_lr=60)
    for p in wczytaj_przebiegi(zrob(ctx60))["przebiegi"]:
        ostatnie = p["train_loss"][-10:]
        assert len(p["train_loss"]) == 60
        assert p["koncowa_strata"] == pytest.approx(sum(ostatnie) / 10, abs=1e-6)


def test_dobry_lr_lepszy_niz_za_maly(tmp_path):
    ctx = dataclasses.replace(maly_kontekst_i(tmp_path), kroki_lr=60)
    przebiegi = {p["nazwa"]: p for p in wczytaj_przebiegi(zrob(ctx))["przebiegi"]}
    assert przebiegi["dobry"]["koncowa_strata"] < przebiegi["za mały"]["koncowa_strata"]


def test_koncowa_strata_pomija_nan(tmp_path, monkeypatch):
    nan = float("nan")
    atrapa_z_stratami(monkeypatch, [nan, nan, nan, 1.0, 2.0, 3.0])
    ctx = maly_kontekst_i(tmp_path)
    przebiegi = wczytaj_przebiegi(zrob(ctx))["przebiegi"]
    assert len(przebiegi) == 3
    for p in przebiegi:
        assert p["koncowa_strata"] == pytest.approx(2.0)
        assert p["train_loss"][:3] == [None, None, None]
        assert p["train_loss"][3:] == [1.0, 2.0, 3.0]


def test_koncowa_strata_none_gdy_same_nan(tmp_path, monkeypatch):
    atrapa_z_stratami(monkeypatch, [float("nan")] * 6)
    ctx = maly_kontekst_i(tmp_path)
    przebiegi = wczytaj_przebiegi(zrob(ctx))["przebiegi"]
    assert len(przebiegi) == 3
    for p in przebiegi:
        assert p["koncowa_strata"] is None
        assert p["train_loss"] == [None] * 6


def test_za_duzy_lr_nie_przerywa(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "przewodnik.learning_rate.PRZEBIEGI",
        (("za mały", 1e-5), ("dobry", 2e-3), ("za duży", 1e3)),
    )
    ctx = maly_kontekst_i(tmp_path)
    przebiegi = wczytaj_przebiegi(zrob(ctx))["przebiegi"]
    assert len(przebiegi) == 3
    ostatni = przebiegi[2]
    assert ostatni["lr"] == 1000.0
    assert len(ostatni["train_loss"]) == 6
    assert all(x is None or isinstance(x, (int, float)) for x in ostatni["train_loss"])
    assert isinstance(ostatni["tekst"], str)


def test_rozbiezny_model_daje_komunikat(tmp_path, monkeypatch):
    def rzuca_runtime(*args, **kwargs):
        raise RuntimeError("nan")

    monkeypatch.setattr("przewodnik.learning_rate.probka_tekstu", rzuca_runtime)
    ctx = maly_kontekst_i(tmp_path / "a")
    przebiegi = wczytaj_przebiegi(zrob(ctx))["przebiegi"]
    assert len(przebiegi) == 3
    assert all(p["tekst"] == KOMUNIKAT for p in przebiegi)

    def rzuca_value(*args, **kwargs):
        raise ValueError("zle")

    monkeypatch.setattr("przewodnik.learning_rate.probka_tekstu", rzuca_value)
    with pytest.raises(ValueError):
        zrob(maly_kontekst_i(tmp_path / "b"))


def test_powtarzalnosc(tmp_path):
    przebiegi_a = wczytaj_przebiegi(zrob(maly_kontekst_i(tmp_path / "a", seed=0)))["przebiegi"]
    przebiegi_b = wczytaj_przebiegi(zrob(maly_kontekst_i(tmp_path / "b", seed=0)))["przebiegi"]
    assert len(przebiegi_a) == len(przebiegi_b) == 3
    for a, b in zip(przebiegi_a, przebiegi_b, strict=True):
        assert a["train_loss"] == b["train_loss"]
        assert a["tekst"] == b["tekst"]


def test_model_kontekstu_bez_zmian(tmp_path):
    ctx = maly_kontekst_i(tmp_path)
    przed = {nazwa: t.clone() for nazwa, t in ctx.model.state_dict().items()}
    zrob(ctx)
    po = ctx.model.state_dict()
    assert set(po) == set(przed)
    for nazwa, tensor in przed.items():
        assert torch.equal(po[nazwa], tensor)
    assert ctx.model.training is False
