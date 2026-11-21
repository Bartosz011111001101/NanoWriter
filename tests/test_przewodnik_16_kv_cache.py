"""Testy modułu przewodnik.kv_cache."""

import copy
import dataclasses
import json

import pytest
import torch
from PIL import Image

from nanowriter.model import GPT, GPTConfig
from nanowriter.sample import generate as prawdziwy_generate
from przewodnik.kv_cache import (
    DLUGOSC_PROMPTU,
    DLUGOSCI_CZASU,
    POWTORZENIA,
    _czas_generowania,
    zrob,
)
from przewodnik.proba import maly_kontekst_i
from przewodnik.wspolne import ZDANIE, etykieta_tokenu, zakoduj_zdanie

NAZWY_PLIKOW = ["kv_cache.gif", "kv_cache.png", "kv_cache_czas.png", "kv_cache.json"]
KLUCZE_KROKU = {
    "krok",
    "token",
    "dlugosc_cache",
    "bajty_cache",
    "tokeny_z_cache",
    "tokeny_bez_cache",
    "roznica_logitow",
}


@pytest.fixture(scope="module")
def wynik_zrob(tmp_path_factory):
    """Jedno pełne zrob(ctx) współdzielone przez testy: (ctx, ids, pliki, dane JSON)."""
    ctx = maly_kontekst_i(tmp_path_factory.mktemp("kv_cache"))
    ids = zakoduj_zdanie(ctx.tokenizer, ZDANIE, 256)
    pliki = zrob(ctx)
    dane = json.loads(pliki[3].read_text(encoding="utf-8"))
    return ctx, ids, pliki, dane


def model_z_kontekstem(ctx, block_size: int) -> GPT:
    torch.manual_seed(0)
    config = GPTConfig(
        vocab_size=ctx.tokenizer.vocab_size,
        block_size=block_size,
        n_layer=2,
        n_head=2,
        d_model=32,
    )
    return GPT(config).eval()


def zainstaluj_licznik(monkeypatch) -> list:
    """Podmienia generate w kv_cache na licznik, który zapisuje wywołania i deleguje dalej."""
    wywolania: list = []

    def licznik(*args, **kwargs):
        wywolania.append((args, kwargs, kwargs["generator"].get_state().clone()))
        return prawdziwy_generate(*args, **kwargs)

    monkeypatch.setattr("przewodnik.kv_cache.generate", licznik)
    return wywolania


def liczba_tokenow(wywolanie) -> int:
    args, kwargs, _ = wywolanie
    return args[2] if len(args) > 2 else kwargs["max_new_tokens"]


def test_stale_modulu():
    assert DLUGOSCI_CZASU == (16, 48, 96, 160)
    assert DLUGOSC_PROMPTU == 4
    assert POWTORZENIA == 2


def test_pliki_nazwy_kolejnosc_i_format(wynik_zrob):
    ctx, _, pliki, _ = wynik_zrob
    assert [p.name for p in pliki] == NAZWY_PLIKOW
    for plik in pliki:
        assert plik.parent == ctx.wyniki_dir
        assert plik.exists()
        assert plik.stat().st_size > 0
    assert pliki[0].read_bytes().startswith(b"GIF8")
    assert pliki[1].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pliki[2].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    zawartosc = pliki[3].read_bytes()
    assert zawartosc.endswith(b"\n")
    assert b"\r\n" not in zawartosc


def test_json_klucze_i_kroki(wynik_zrob):
    ctx, ids, _, dane = wynik_zrob
    liczba = ids.shape[1]
    assert dane["material"] == "kv_cache"
    assert dane["zdanie"] == ZDANIE
    assert "kroki" in dane
    assert "czasy" in dane
    assert len(dane["kroki"]) == liczba
    for i in range(liczba):
        krok = dane["kroki"][i]
        assert set(krok) == KLUCZE_KROKU
        assert krok["krok"] == i
        assert krok["token"] == etykieta_tokenu(ctx.tokenizer, int(ids[0, i]))
        assert krok["dlugosc_cache"] == i + 1
        assert krok["tokeny_z_cache"] == 1
        assert krok["tokeny_bez_cache"] == i + 1
        assert krok["bajty_cache"] == 2 * 2 * (i + 1) * 32 * 4
        assert 0 <= krok["roznica_logitow"] <= 1e-4


def test_gif_liczba_klatek(wynik_zrob):
    _, ids, pliki, _ = wynik_zrob
    with Image.open(pliki[0]) as gif:
        assert gif.n_frames == ids.shape[1]


def test_czasy_cpu(wynik_zrob):
    _, _, _, dane = wynik_zrob
    czasy = dane["czasy"]
    assert len(czasy) == 1
    wpis = czasy[0]
    assert set(wpis) == {"urzadzenie", "dlugosci", "bez_cache_s", "z_cache_s"}
    assert wpis["urzadzenie"] == "cpu"
    assert wpis["dlugosci"] == [16, 48, 96, 160]
    assert len(wpis["bez_cache_s"]) == 4
    assert len(wpis["z_cache_s"]) == 4
    assert all(c > 0 for c in wpis["bez_cache_s"])
    assert all(c > 0 for c in wpis["z_cache_s"])


def test_granica_dlugosci_wzgledem_block_size(tmp_path, monkeypatch):
    ctx = maly_kontekst_i(tmp_path)
    ctx = dataclasses.replace(ctx, model=model_z_kontekstem(ctx, 100))
    monkeypatch.setattr("przewodnik.kv_cache.DLUGOSCI_CZASU", (16, 96, 97))
    monkeypatch.setattr("przewodnik.kv_cache.POWTORZENIA", 1)
    pliki = zrob(ctx)
    czasy = json.loads(pliki[3].read_text(encoding="utf-8"))["czasy"]
    assert czasy[0]["dlugosci"] == [16, 96]
    assert len(czasy[0]["bez_cache_s"]) == 2
    assert len(czasy[0]["z_cache_s"]) == 2


def test_brak_pasujacych_dlugosci_value_error(tmp_path, monkeypatch):
    ctx = maly_kontekst_i(tmp_path)
    ctx = dataclasses.replace(ctx, model=model_z_kontekstem(ctx, 19))
    monkeypatch.setattr("przewodnik.kv_cache.DLUGOSCI_CZASU", (16, 48))
    with pytest.raises(ValueError):
        zrob(ctx)


def test_czas_generowania_wartosc(tmp_path):
    ctx = maly_kontekst_i(tmp_path)
    ids = zakoduj_zdanie(ctx.tokenizer, ZDANIE, 256)
    prompt = ids[:, :4]
    for use_cache in (True, False):
        czas = _czas_generowania(ctx.model, prompt, 8, "cpu", 0, use_cache)
        assert isinstance(czas, float)
        assert czas > 0
    assert ctx.model.training is False


@pytest.mark.parametrize("use_cache", [True, False])
def test_czas_generowania_wywolania_generate(tmp_path, monkeypatch, use_cache):
    ctx = maly_kontekst_i(tmp_path)
    ids = zakoduj_zdanie(ctx.tokenizer, ZDANIE, 256)
    wywolania = zainstaluj_licznik(monkeypatch)
    monkeypatch.setattr("przewodnik.kv_cache.POWTORZENIA", 3)

    _czas_generowania(ctx.model, ids[:, :4], 8, "cpu", 5, use_cache)

    assert len(wywolania) == 4
    oczekiwany_stan = torch.Generator(device="cpu").manual_seed(5).get_state()
    for _, kwargs, stan in wywolania:
        assert kwargs["use_cache"] is use_cache
        assert kwargs["block_size"] == 256
        assert torch.equal(stan, oczekiwany_stan)
    assert liczba_tokenow(wywolania[0]) <= 8
    for wywolanie in wywolania[1:]:
        assert liczba_tokenow(wywolanie) == 8


def test_model_bez_zmian(tmp_path):
    ctx = maly_kontekst_i(tmp_path)
    migawka = copy.deepcopy(ctx.model.state_dict())
    tryb_przed = ctx.model.training
    assert tryb_przed is False
    zrob(ctx)
    po = ctx.model.state_dict()
    assert set(po) == set(migawka)
    for nazwa, tensor in migawka.items():
        assert torch.equal(po[nazwa], tensor)
    assert ctx.model.training is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="brak GPU")
def test_gpu_dwa_wpisy(tmp_path, monkeypatch):
    ctx = maly_kontekst_i(tmp_path)
    ctx.model.to("cuda")
    ctx = dataclasses.replace(ctx, device="cuda")
    monkeypatch.setattr("przewodnik.kv_cache.DLUGOSCI_CZASU", (16,))
    pliki = zrob(ctx)
    czasy = json.loads(pliki[3].read_text(encoding="utf-8"))["czasy"]
    assert [w["urzadzenie"] for w in czasy] == ["cuda", "cpu"]
