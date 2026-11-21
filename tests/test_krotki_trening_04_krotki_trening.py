"""Testy modułu przewodnik.krotki_trening."""

import dataclasses
import math

import pytest
import torch

from nanowriter.model import GPT, GPTConfig
from przewodnik.krotki_trening import WynikTreningu, krotki_trening, nowy_model

KONFIG = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, d_model=32)


def dane_losowe() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, 64, (400,))


def dane_okresowe() -> torch.Tensor:
    return torch.arange(400) % 8


def trening(model: GPT, dane: torch.Tensor | None = None, **kwargs):
    """Wywołanie bazowe krotki_trening; argumenty nazwane nadpisują domyślne."""
    if dane is None:
        dane = dane_losowe()
    argumenty = {
        "kroki": 4,
        "lr": 0.003,
        "batch_size": 4,
        "device": "cpu",
        "seed": 0,
        "co_ile": 2,
    }
    argumenty.update(kwargs)
    return krotki_trening(model, dane, dane, **argumenty)


def stan(model: GPT) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in model.state_dict().items()}


def test_nowy_model_liczba_parametrow_i_typ():
    m = nowy_model(KONFIG, 0, "cpu")
    assert isinstance(m, GPT)
    assert m.config == KONFIG
    assert all(p.device.type == "cpu" for p in m.parameters())
    assert sum(p.numel() for p in m.parameters()) == 28032


def test_nowy_model_seed():
    a = nowy_model(KONFIG, 0, "cpu").state_dict()
    b = nowy_model(KONFIG, 0, "cpu").state_dict()
    c = nowy_model(KONFIG, 1, "cpu").state_dict()
    assert a.keys() == b.keys() == c.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert any(not torch.equal(a[k], c[k]) for k in a)


def test_krotki_trening_kroki_walidacja_i_na_krok():
    wolane: list[int] = []
    model = nowy_model(KONFIG, 0, "cpu")
    wynik = trening(
        model, kroki=4, co_ile=2, na_krok=lambda krok, strata: wolane.append(krok)
    )
    assert isinstance(wynik, WynikTreningu)
    assert wynik.kroki == [0, 1, 2, 3]
    assert len(wynik.train_loss) == 4
    assert wynik.val_kroki == [1, 3]
    assert len(wynik.val_loss) == 2
    assert wolane == [0, 2, 3]
    assert all(isinstance(k, int) for k in wynik.kroki)
    for strata in wynik.train_loss + wynik.val_loss:
        assert isinstance(strata, float)
        assert math.isfinite(strata)


def test_krotki_trening_granice_co_ile():
    przypadki = [
        (5, 2, [1, 3, 4], [0, 2, 4]),
        (3, 1, [0, 1, 2], [0, 1, 2]),
        (3, 10, [2], [0, 2]),
        (1, 1, [0], [0]),
    ]
    for kroki, co_ile, val_kroki, wolane in przypadki:
        zapis: list[int] = []
        model = nowy_model(KONFIG, 0, "cpu")
        wynik = trening(
            model,
            kroki=kroki,
            co_ile=co_ile,
            na_krok=lambda krok, strata, zapis=zapis: zapis.append(krok),
        )
        assert wynik.kroki == list(range(kroki))
        assert len(wynik.train_loss) == kroki
        assert wynik.val_kroki == val_kroki
        assert len(wynik.val_loss) == len(val_kroki)
        assert zapis == wolane


def test_krotki_trening_na_krok_dostaje_straty_krokow():
    pary: list[tuple[int, float]] = []
    model = nowy_model(KONFIG, 0, "cpu")
    wynik = trening(
        model,
        kroki=5,
        co_ile=2,
        na_krok=lambda krok, strata: pary.append((krok, strata)),
    )
    assert len(pary) == 3
    for krok, strata in pary:
        assert isinstance(krok, int)
        assert strata == pytest.approx(wynik.train_loss[krok], rel=1e-5)


def test_krotki_trening_bez_na_krok():
    model = nowy_model(KONFIG, 0, "cpu")
    wynik = trening(model, kroki=4, co_ile=2)
    assert len(wynik.train_loss) == 4


def test_krotki_trening_powtarzalnosc():
    a = trening(nowy_model(KONFIG, 0, "cpu"), kroki=6, co_ile=3, seed=0)
    b = trening(nowy_model(KONFIG, 0, "cpu"), kroki=6, co_ile=3, seed=0)
    c = trening(nowy_model(KONFIG, 0, "cpu"), kroki=6, co_ile=3, seed=1)
    assert a.train_loss == b.train_loss
    assert a.val_loss == b.val_loss
    assert a.train_loss != c.train_loss


def test_krotki_trening_uczy_sie():
    model = nowy_model(KONFIG, 0, "cpu")
    wynik = trening(
        model,
        dane_okresowe(),
        kroki=60,
        lr=0.01,
        batch_size=8,
        co_ile=20,
    )
    pierwsze = sum(wynik.train_loss[:10]) / 10
    ostatnie = sum(wynik.train_loss[-10:]) / 10
    assert ostatnie < pierwsze
    assert wynik.val_kroki == [19, 39, 59]
    assert len(wynik.val_loss) == 3


def test_krotki_trening_model_i_pola_wyniku():
    model = nowy_model(KONFIG, 0, "cpu")
    kopia = stan(model)
    wynik = trening(model, lr=0.003)
    assert wynik.model is model
    assert model.training is False
    po = model.state_dict()
    assert any(not torch.equal(kopia[k], po[k]) for k in kopia)
    assert wynik.lr == 0.003
    assert isinstance(wynik.czas_s, float)
    assert wynik.czas_s >= 0

    model.train()
    trening(model)
    assert model.training is False


def test_wynik_treningu_zamrozony():
    assert dataclasses.is_dataclass(WynikTreningu)
    assert [f.name for f in dataclasses.fields(WynikTreningu)] == [
        "model",
        "kroki",
        "train_loss",
        "val_kroki",
        "val_loss",
        "lr",
        "czas_s",
    ]
    wynik = trening(nowy_model(KONFIG, 0, "cpu"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        wynik.lr = 1.0  # type: ignore[misc]
    assert wynik == wynik  # noqa: PLR0124
    assert dataclasses.replace(wynik, lr=wynik.lr) != wynik


def test_krotki_trening_bledy_przed_treningiem():
    przypadki = [
        ({"kroki": 0}, "kroki"),
        ({"kroki": -1}, None),
        ({"co_ile": 0}, "co_ile"),
        ({"co_ile": -1}, None),
        ({"batch_size": 0}, "batch_size"),
        ({"batch_size": -1}, None),
    ]
    model = nowy_model(KONFIG, 0, "cpu")
    kopia = stan(model)
    for kwargs, nazwa in przypadki:
        with pytest.raises(ValueError) as info:
            trening(model, **kwargs)
        if nazwa is not None:
            assert nazwa in str(info.value)
        po = model.state_dict()
        assert all(torch.equal(kopia[k], po[k]) for k in kopia)

    dane_2d = torch.zeros(10, 40, dtype=torch.long)
    argumenty = {
        "kroki": 4,
        "lr": 0.003,
        "batch_size": 4,
        "device": "cpu",
        "seed": 0,
        "co_ile": 2,
    }
    for train_dane, val_dane in [(dane_2d, dane_losowe()), (dane_losowe(), dane_2d)]:
        with pytest.raises(ValueError):
            krotki_trening(model, train_dane, val_dane, **argumenty)
        po = model.state_dict()
        assert all(torch.equal(kopia[k], po[k]) for k in kopia)


def test_krotki_trening_ogromny_lr_nie_rzuca():
    model = nowy_model(KONFIG, 0, "cpu")
    wynik = trening(model, lr=1e3, kroki=10, co_ile=5)
    assert len(wynik.train_loss) == 10
    assert wynik.kroki == list(range(10))
    assert model.training is False
    assert wynik.val_kroki == [4, 9]


def test_krotki_trening_nic_nie_wypisuje(capsys):
    model = nowy_model(KONFIG, 0, "cpu")
    trening(model, kroki=4, co_ile=2)
    przechwycone = capsys.readouterr()
    assert przechwycone.out == ""
    assert przechwycone.err == ""
