"""Testy modułu przewodnik.rozmiary."""

import dataclasses
import json
import math

import pytest
import torch

from nanowriter.model import GPT, GPTConfig
from przewodnik.proba import maly_kontekst_ii
from przewodnik.rozmiary import LICZBA_TOKENOW, OPISY_KORPUSOW, PROMPT, _podzial, zrob
from przewodnik.wspolne import probka_tekstu

CZESCI = ["embedding_tokenow", "embedding_pozycji", "uwaga", "mlp", "layernorm"]
KONFIG_10M = GPTConfig(vocab_size=8192, block_size=256, n_layer=6, n_head=5, d_model=320)


def liczba_parametrow(model: GPT) -> int:
    return sum(p.numel() for p in model.parameters())


def log_bez_val_loss(katalog):
    """Log JSON Lines z samymi wierszami train_loss."""
    sciezka = katalog / "bez_val.jsonl"
    with open(sciezka, "w", encoding="utf-8", newline="\n") as plik:
        for krok in (0, 10, 20):
            plik.write(json.dumps({"step": krok, "train_loss": 5.0, "lr": 0.001}) + "\n")
    return sciezka


@pytest.fixture(scope="module")
def wynik(tmp_path_factory):
    """Kontekst i wynik jednego wywołania zrob (współdzielone w module)."""
    ctx = maly_kontekst_ii(tmp_path_factory.mktemp("rozmiary"))
    pliki = zrob(ctx)
    return ctx, pliki


@pytest.fixture(scope="module")
def dane_json(wynik):
    ctx, _ = wynik
    return json.loads((ctx.wyniki_dir / "rozmiary.json").read_text(encoding="utf-8"))


def test_stale_modulu():
    assert PROMPT == "Wieczorem stary kot"
    assert LICZBA_TOKENOW == 60
    assert OPISY_KORPUSOW == (
        "książki z Wolnych Lektur, słownik 1024",
        "napisy filmowe i książki, słownik 8192",
    )


def test_podzial_model_1m():
    torch.manual_seed(0)
    podzial = _podzial(GPT(GPTConfig()))
    assert podzial == {
        "embedding_tokenow": 131072,
        "embedding_pozycji": 32768,
        "uwaga": 264192,
        "mlp": 526848,
        "layernorm": 2304,
    }
    assert sum(podzial.values()) == 957184


def test_podzial_model_10m():
    torch.manual_seed(0)
    podzial = _podzial(GPT(KONFIG_10M))
    assert sorted(podzial.keys()) == sorted(CZESCI)
    assert podzial["embedding_tokenow"] == 2621440
    assert sum(podzial.values()) == 10101760


def test_podzial_suma_rowna_liczbie_parametrow(wynik):
    ctx, _ = wynik
    model = ctx.model_1m
    assert sum(_podzial(model).values()) == sum(p.numel() for p in model.parameters())


def test_podzial_niezgodna_suma_runtime_error():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, d_model=32))
    model.dodatkowy = torch.nn.Parameter(torch.zeros(3))
    with pytest.raises(RuntimeError):
        _podzial(model)


def test_pliki_nazwy_kolejnosc_i_format(wynik):
    ctx, pliki = wynik
    assert [p.name for p in pliki] == ["rozmiary.png", "rozmiary_probki.png", "rozmiary.json"]
    for sciezka in pliki:
        assert sciezka.parent == ctx.wyniki_dir
        assert sciezka.exists()
        assert sciezka.stat().st_size > 0
    for sciezka in pliki[:2]:
        assert sciezka.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    zawartosc = pliki[2].read_bytes()
    assert zawartosc.endswith(b"\n")
    assert b"\r\n" not in zawartosc


def test_json_modele_i_podzial(wynik, dane_json):
    ctx, _ = wynik
    assert dane_json["material"] == "rozmiary"
    assert dane_json["prompt"] == PROMPT
    modele = dane_json["modele"]
    assert len(modele) == 2
    assert [m["nazwa"] for m in modele] == ["1M", "10M wstępny"]
    klucze = {"nazwa", "parametry", "podzial", "val_loss", "bity_na_znak", "uwaga_korpus", "probka"}
    for m in modele:
        assert klucze <= set(m.keys())
        assert sorted(m["podzial"].keys()) == sorted(CZESCI)
        assert sum(m["podzial"].values()) == m["parametry"]
    for m, model in zip(modele, (ctx.model_1m, ctx.model_wstepny), strict=True):
        assert m["parametry"] == liczba_parametrow(model)
        assert m["podzial"] == _podzial(model)


def test_bity_na_znak(dane_json):
    oczekiwane = 4.2 * 1200 / (3600 * math.log(2))
    for m in dane_json["modele"]:
        assert m["val_loss"] == pytest.approx(4.2)
        assert m["bity_na_znak"] == pytest.approx(oczekiwane, abs=1e-3)
        assert m["bity_na_znak"] == pytest.approx(2.0198, abs=1e-3)


def test_uwaga_korpus(dane_json):
    for i in (0, 1):
        assert dane_json["modele"][i]["uwaga_korpus"] == (
            "Strata z logu treningu na własnym zbiorze walidacyjnym tego modelu "
            f"({OPISY_KORPUSOW[i]}); zbiory są różne, więc porównanie bitów na znak jest "
            "przybliżone"
        )


def test_probki(wynik, dane_json, tmp_path):
    ctx, _ = wynik
    probki = [m["probka"] for m in dane_json["modele"]]
    assert all(p.startswith(PROMPT) for p in probki)
    oczekiwane = [
        probka_tekstu(
            ctx.model_1m, ctx.tokenizer_1m, PROMPT, LICZBA_TOKENOW, device="cpu", seed=ctx.seed
        ),
        probka_tekstu(
            ctx.model_wstepny,
            ctx.tokenizer_10m,
            PROMPT,
            LICZBA_TOKENOW,
            device="cpu",
            seed=ctx.seed,
        ),
    ]
    assert probki == oczekiwane

    ctx_a = maly_kontekst_ii(tmp_path / "a")
    ctx_b = maly_kontekst_ii(tmp_path / "b")
    zrob(ctx_a)
    zrob(ctx_b)
    json_a = json.loads((ctx_a.wyniki_dir / "rozmiary.json").read_text(encoding="utf-8"))
    json_b = json.loads((ctx_b.wyniki_dir / "rozmiary.json").read_text(encoding="utf-8"))
    assert [m["probka"] for m in json_a["modele"]] == [m["probka"] for m in json_b["modele"]]


def test_log_bez_val_loss_value_error(tmp_path):
    ctx = maly_kontekst_ii(tmp_path / "ctx")
    log = log_bez_val_loss(tmp_path)
    for pole in ("przebieg_1m", "przebieg_wstepny"):
        przebieg = dataclasses.replace(getattr(ctx, pole), log=log)
        zepsuty = dataclasses.replace(ctx, **{pole: przebieg})
        with pytest.raises(ValueError):
            zrob(zepsuty)


def test_modele_bez_zmian(tmp_path):
    ctx = maly_kontekst_ii(tmp_path)
    przed = {
        nazwa: {k: v.clone() for k, v in model.state_dict().items()}
        for nazwa, model in (("1m", ctx.model_1m), ("wstepny", ctx.model_wstepny))
    }
    zrob(ctx)
    for nazwa, model in (("1m", ctx.model_1m), ("wstepny", ctx.model_wstepny)):
        po = model.state_dict()
        assert po.keys() == przed[nazwa].keys()
        assert all(torch.equal(po[k], przed[nazwa][k]) for k in po)
        assert model.training is False
