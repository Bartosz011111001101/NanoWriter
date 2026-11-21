"""Testy modułu przewodnik.bez_maski."""

import json
import math

import pytest
import torch

from nanowriter.model import GPT, GPTConfig
from nanowriter.train import estimate_loss
from przewodnik.bez_maski import (
    LICZBA_TOKENOW,
    LR,
    PROMPT,
    _przywroc_maske,
    _wylacz_maske,
    zrob,
)
from przewodnik.krotki_trening import krotki_trening as prawdziwy_trening
from przewodnik.proba import maly_kontekst_i

PNG_NAGLOWEK = b"\x89PNG\r\n\x1a\n"


def maly_model() -> GPT:
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, d_model=32))


def trojkat(n: int) -> torch.Tensor:
    return torch.tril(torch.ones(n, n)).bool()


def stan(model: GPT) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in model.state_dict().items()}


@pytest.fixture
def monitor_i_wywolania(monkeypatch):
    """Podstawia monitor w miejsce krotki_trening i zwraca listę zapisanych wywołań."""
    wywolania: list[dict] = []

    def monitor(model, *args, **kwargs):
        wywolania.append(
            {
                "model": model,
                "stan": stan(model),
                "kwargs": dict(kwargs),
                "maski_pelne": [blok.attn.mask.all().item() for blok in model.blocks],
            }
        )
        return prawdziwy_trening(model, *args, **kwargs)

    monkeypatch.setattr("przewodnik.bez_maski.krotki_trening", monitor)
    return wywolania


def test_stale_modulu():
    assert PROMPT == "Wieczorem stary kot"
    assert LICZBA_TOKENOW == 40
    assert LR == 0.002


def test_wylacz_maske():
    model = maly_model()
    m0 = model.blocks[0].attn.mask
    m1 = model.blocks[1].attn.mask
    p0 = m0.data_ptr()
    p1 = m1.data_ptr()

    assert _wylacz_maske(model) is None

    assert model.blocks[0].attn.mask is m0
    assert model.blocks[1].attn.mask is m1
    assert m0.data_ptr() == p0
    assert m1.data_ptr() == p1
    for blok in model.blocks:
        maska = blok.attn.mask
        assert tuple(maska.shape) == (16, 16)
        assert maska.dtype == torch.bool
        assert maska.all()

    _wylacz_maske(model)
    for blok in model.blocks:
        assert blok.attn.mask.all()


def test_przywroc_maske():
    model = maly_model()
    bufory = [blok.attn.mask for blok in model.blocks]
    wskazniki = [maska.data_ptr() for maska in bufory]

    _wylacz_maske(model)
    _przywroc_maske(model)

    for blok, bufor, wskaznik in zip(model.blocks, bufory, wskazniki, strict=True):
        maska = blok.attn.mask
        assert torch.equal(maska, trojkat(16))
        assert maska.dtype == torch.bool
        assert maska is bufor
        assert maska.data_ptr() == wskaznik

    # Przywrócenie na modelu, którego maski nie wyłączono, nic nie zmienia.
    model2 = maly_model()
    _przywroc_maske(model2)
    for blok in model2.blocks:
        assert torch.equal(blok.attn.mask, trojkat(16))


def test_wylaczona_maska_zmienia_wyjscie_modelu():
    model = maly_model()
    model.eval()
    torch.manual_seed(0)
    idx = torch.randint(0, 64, (1, 16))
    idx2 = idx.clone()
    idx2[0, 10] = (idx[0, 10] + 1) % 64
    assert not torch.equal(idx, idx2)

    def logity(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            wynik = model(x)
        return wynik[0] if isinstance(wynik, tuple) else wynik

    a = logity(idx)
    b = logity(idx2)
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-5)

    _wylacz_maske(model)
    a = logity(idx)
    b = logity(idx2)
    assert not torch.allclose(a[:, :10], b[:, :10], atol=1e-5)

    _przywroc_maske(model)
    a = logity(idx)
    b = logity(idx2)
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-5)


def test_pliki_nazwy_kolejnosc_i_format(tmp_path):
    ctx = maly_kontekst_i(tmp_path)

    pliki = zrob(ctx)

    assert [p.name for p in pliki] == ["bez_maski.png", "bez_maski.json"]
    for plik in pliki:
        assert plik.parent == ctx.wyniki_dir
        assert plik.is_file()
        assert plik.stat().st_size > 0
    assert pliki[0].read_bytes().startswith(PNG_NAGLOWEK)
    tresc = pliki[1].read_bytes()
    assert tresc.endswith(b"\n")
    assert b"\r\n" not in tresc


def test_json_klucze_i_liczby(tmp_path):
    ctx = maly_kontekst_i(tmp_path)

    pliki = zrob(ctx)
    dane = json.loads(pliki[1].read_text(encoding="utf-8"))

    assert dane["material"] == "bez_maski"
    assert dane["kroki"] == 6
    assert dane["prompt"] == PROMPT

    z_maska = dane["z_maska"]
    bez_maski = dane["bez_maski"]
    assert set(z_maska) == {"train_loss", "val_loss_uczciwa", "tekst"}
    assert set(bez_maski) == {
        "train_loss",
        "val_loss_podgladajaca",
        "val_loss_uczciwa",
        "tekst",
    }
    assert len(z_maska["train_loss"]) == 6
    assert len(bez_maski["train_loss"]) == 6
    for przebieg in (z_maska, bez_maski):
        assert all(math.isfinite(x) for x in przebieg["train_loss"])
        assert isinstance(przebieg["tekst"], str)
        assert przebieg["tekst"].startswith(PROMPT)

    for liczba in (
        z_maska["val_loss_uczciwa"],
        bez_maski["val_loss_podgladajaca"],
        bez_maski["val_loss_uczciwa"],
    ):
        assert math.isfinite(liczba)
        assert liczba > 0

    assert abs(z_maska["train_loss"][0] - bez_maski["train_loss"][0]) <= 0.5
    ln_vocab = math.log(320)
    assert 0.9 * ln_vocab <= z_maska["train_loss"][0] <= 1.1 * ln_vocab


def test_ctx_model_nietkniety(tmp_path):
    ctx = maly_kontekst_i(tmp_path)
    przed = stan(ctx.model)

    zrob(ctx)

    po = ctx.model.state_dict()
    assert po.keys() == przed.keys()
    for klucz in przed:
        assert torch.equal(po[klucz], przed[klucz]), klucz
    assert ctx.model.training is False
    for blok in ctx.model.blocks:
        assert torch.equal(blok.attn.mask, trojkat(256))


def test_monitor_dwoch_przebiegow(tmp_path, monitor_i_wywolania):
    ctx = maly_kontekst_i(tmp_path)

    zrob(ctx)

    assert len(monitor_i_wywolania) == 2
    a, b = monitor_i_wywolania

    # Przebieg A ma maskę przyczynową (nie same True), przebieg B same True.
    assert not any(a["maski_pelne"])
    assert all(b["maski_pelne"])

    assert a["model"] is not ctx.model
    assert b["model"] is not ctx.model

    assert a["stan"].keys() == b["stan"].keys()
    for klucz in a["stan"]:
        if klucz.endswith("attn.mask"):
            continue
        assert torch.equal(a["stan"][klucz], b["stan"][klucz]), klucz

    co_ile = max(1, ctx.kroki_bez_maski // 10)
    assert co_ile == 1
    for wywolanie in (a, b):
        kw = wywolanie["kwargs"]
        assert kw["kroki"] == ctx.kroki_bez_maski
        assert kw["lr"] == 0.002
        assert kw["batch_size"] == ctx.batch_size
        assert kw["device"] == "cpu"
        assert kw["seed"] == ctx.seed
        assert kw["co_ile"] == co_ile
    assert a["kwargs"]["co_ile"] == b["kwargs"]["co_ile"]


def test_maska_modelu_b_przywrocona_po_zrob(tmp_path, monitor_i_wywolania):
    ctx = maly_kontekst_i(tmp_path)

    zrob(ctx)

    assert len(monitor_i_wywolania) == 2
    for wywolanie in monitor_i_wywolania:
        for blok in wywolanie["model"].blocks:
            assert torch.equal(blok.attn.mask, trojkat(256))


def test_walidacje_modelu_b_zgodne_z_przeliczeniem(tmp_path, monitor_i_wywolania):
    ctx = maly_kontekst_i(tmp_path)

    pliki = zrob(ctx)
    dane = json.loads(pliki[1].read_text(encoding="utf-8"))
    model_b = monitor_i_wywolania[1]["model"]

    # Maska już przywrócona: strata uczciwa.
    uczciwa = estimate_loss(
        model_b, ctx.dane_val, ctx.batch_size, 256, "cpu", max_batches=5
    )
    assert dane["bez_maski"]["val_loss_uczciwa"] == pytest.approx(uczciwa, abs=1e-5)

    # Wyłączamy maskę ręcznie: strata podglądająca.
    for blok in model_b.blocks:
        blok.attn.mask.fill_(True)
    podgladajaca = estimate_loss(
        model_b, ctx.dane_val, ctx.batch_size, 256, "cpu", max_batches=5
    )
    assert dane["bez_maski"]["val_loss_podgladajaca"] == pytest.approx(podgladajaca, abs=1e-4)

    assert abs(
        dane["bez_maski"]["val_loss_uczciwa"] - dane["bez_maski"]["val_loss_podgladajaca"]
    ) >= 1e-6


def test_powtarzalnosc(tmp_path):
    ctx_a = maly_kontekst_i(tmp_path / "a", seed=0)
    ctx_b = maly_kontekst_i(tmp_path / "b", seed=0)

    dane_a = json.loads(zrob(ctx_a)[1].read_text(encoding="utf-8"))
    dane_b = json.loads(zrob(ctx_b)[1].read_text(encoding="utf-8"))

    assert dane_a["z_maska"]["train_loss"] == dane_b["z_maska"]["train_loss"]
    assert dane_a["bez_maski"]["train_loss"] == dane_b["bez_maski"]["train_loss"]
    assert dane_a["z_maska"]["tekst"] == dane_b["z_maska"]["tekst"]
    assert dane_a["bez_maski"]["tekst"] == dane_b["bez_maski"]["tekst"]
