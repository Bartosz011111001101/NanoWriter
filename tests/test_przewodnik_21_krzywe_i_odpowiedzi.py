"""Testy modułu przewodnik.krzywe_i_odpowiedzi."""

import dataclasses
import json

import pytest
import torch

from nanowriter.sample import chat_reply as prawdziwy_chat
from przewodnik.krzywe_i_odpowiedzi import MAX_NOWYCH_TOKENOW, zrob
from przewodnik.proba import maly_kontekst_ii
from przewodnik.wspolne import PROMPTY_ROZMOWY

BRAK = "(brak odpowiedzi)"
NAZWY = ["krzywe_pretrening_douczanie.png", "odpowiedzi.png", "krzywe_i_odpowiedzi.json"]
PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    return maly_kontekst_ii(tmp_path_factory.mktemp("krzywe"))


@pytest.fixture(scope="module")
def pliki(ctx):
    return zrob(ctx)


@pytest.fixture(scope="module")
def dane(pliki):
    return json.loads(pliki[2].read_text(encoding="utf-8"))


def odpowiedz_odniesienia(ctx, model, prompt: str) -> str:
    g = torch.Generator(device="cpu")
    g.manual_seed(ctx.seed)
    tekst = prawdziwy_chat(
        model,
        ctx.tokenizer_10m,
        [],
        prompt,
        device="cpu",
        block_size=256,
        max_new_tokens=40,
        generator=g,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
    )
    return tekst or BRAK


def test_stala():
    assert MAX_NOWYCH_TOKENOW == 40


def test_pliki_nazwy_kolejnosc_i_format(ctx, pliki):
    assert [p.name for p in pliki] == NAZWY
    for plik in pliki:
        assert plik.parent == ctx.wyniki_dir
        assert plik.exists()
        assert plik.stat().st_size > 0
    assert pliki[0].read_bytes().startswith(PNG)
    assert pliki[1].read_bytes().startswith(PNG)
    bajty = pliki[2].read_bytes()
    assert bajty.endswith(b"\n")
    assert b"\r\n" not in bajty


def test_json_krzywe(ctx, dane):
    assert dane["material"] == "krzywe_i_odpowiedzi"
    assert dane["seed"] == ctx.seed
    for klucz in ("pretrening", "douczanie"):
        krzywe = dane[klucz]
        assert set(krzywe) == {"kroki", "train_loss", "val_kroki", "val_loss"}
        assert krzywe["kroki"] == [0, 10, 20, 30, 40]
        assert krzywe["val_kroki"] == [20, 40]
        assert krzywe["train_loss"] == pytest.approx([6.0, 5.5, 5.0, 4.5, 4.0])
        assert krzywe["val_loss"] == pytest.approx([5.2, 4.2])


def test_json_odpowiedzi(dane):
    odpowiedzi = dane["odpowiedzi"]
    assert len(odpowiedzi) == 4
    assert [o["prompt"] for o in odpowiedzi] == list(PROMPTY_ROZMOWY)
    for o in odpowiedzi:
        assert set(o) == {"prompt", "wstepny", "douczony"}
        for klucz in ("wstepny", "douczony"):
            assert o[klucz] != ""
            assert "\n" not in o[klucz]


def test_odpowiedzi_zgodne_z_chat_reply(ctx, dane):
    for o in dane["odpowiedzi"]:
        assert o["wstepny"] == odpowiedz_odniesienia(ctx, ctx.model_wstepny, o["prompt"])
        assert o["douczony"] == odpowiedz_odniesienia(ctx, ctx.model_douczony, o["prompt"])


def test_wywolania_chat_reply(ctx, monkeypatch):
    wywolania = []

    def rejestr(*args, **kwargs):
        wywolania.append((args, kwargs, kwargs["generator"].get_state().clone()))
        return prawdziwy_chat(*args, **kwargs)

    monkeypatch.setattr("przewodnik.krzywe_i_odpowiedzi.chat_reply", rejestr)
    zrob(ctx)

    assert len(wywolania) == 8
    swiezy = torch.Generator(device="cpu")
    swiezy.manual_seed(ctx.seed)
    stan_swiezego = swiezy.get_state()
    liczniki = {"wstepny": 0, "douczony": 0}
    for args, kwargs, stan in wywolania:
        if args[0] is ctx.model_wstepny:
            liczniki["wstepny"] += 1
        else:
            assert args[0] is ctx.model_douczony
            liczniki["douczony"] += 1
        assert args[1] is ctx.tokenizer_10m
        assert args[2] == []
        assert args[3] in PROMPTY_ROZMOWY
        assert kwargs["max_new_tokens"] == 40
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_k"] == 40
        assert kwargs["top_p"] == 0.9
        assert kwargs["block_size"] == 256
        assert kwargs["device"] == "cpu"
        assert torch.equal(stan, stan_swiezego)
    assert liczniki == {"wstepny": 4, "douczony": 4}


def test_brak_odpowiedzi_dla_pustego_wyniku(ctx, monkeypatch):
    monkeypatch.setattr("przewodnik.krzywe_i_odpowiedzi.chat_reply", lambda *args, **kwargs: "")
    pliki = zrob(ctx)
    odpowiedzi = json.loads(pliki[2].read_text(encoding="utf-8"))["odpowiedzi"]
    assert len(odpowiedzi) == 4
    for o in odpowiedzi:
        assert o["wstepny"] == BRAK
        assert o["douczony"] == BRAK


def test_powtarzalnosc_i_inny_seed(tmp_path):
    ctx_a = maly_kontekst_ii(tmp_path / "a")
    ctx_b = maly_kontekst_ii(tmp_path / "b")
    ctx_c = maly_kontekst_ii(tmp_path / "c", seed=1)
    json_a = zrob(ctx_a)[2]
    json_b = zrob(ctx_b)[2]
    json_c = zrob(ctx_c)[2]
    odp_a = json.loads(json_a.read_text(encoding="utf-8"))["odpowiedzi"]
    odp_b = json.loads(json_b.read_text(encoding="utf-8"))["odpowiedzi"]
    odp_c = json.loads(json_c.read_text(encoding="utf-8"))["odpowiedzi"]
    assert odp_a == odp_b
    assert json_a.read_bytes() == json_b.read_bytes()
    assert odp_c != odp_a


def test_log_bez_train_loss_value_error(tmp_path):
    ctx = maly_kontekst_ii(tmp_path / "k")
    pusty = tmp_path / "pusty.log"
    pusty.write_text("", encoding="utf-8")

    zly_wstepny = dataclasses.replace(
        ctx, przebieg_wstepny=dataclasses.replace(ctx.przebieg_wstepny, log=pusty)
    )
    with pytest.raises(ValueError) as wyjatek1:
        zrob(zly_wstepny)

    zly_douczony = dataclasses.replace(
        ctx, przebieg_douczania=dataclasses.replace(ctx.przebieg_douczania, log=pusty)
    )
    with pytest.raises(ValueError) as wyjatek2:
        zrob(zly_douczony)

    assert str(wyjatek1.value) != str(wyjatek2.value)


def test_log_bez_val_loss_dziala(tmp_path):
    ctx = maly_kontekst_ii(tmp_path / "k")
    log = tmp_path / "bez_val.log"
    wiersze = [{"step": s, "train_loss": 5.0, "lr": 0.001} for s in (0, 10, 20)]
    log.write_text("".join(json.dumps(w) + "\n" for w in wiersze), encoding="utf-8")
    ctx_bez_val = dataclasses.replace(
        ctx, przebieg_wstepny=dataclasses.replace(ctx.przebieg_wstepny, log=log)
    )

    pliki = zrob(ctx_bez_val)

    assert pliki[0].exists()
    assert pliki[1].exists()
    dane = json.loads(pliki[2].read_text(encoding="utf-8"))
    assert dane["pretrening"]["val_kroki"] == []
    assert dane["pretrening"]["val_loss"] == []


def test_modele_bez_zmian(ctx):
    modele = (ctx.model_wstepny, ctx.model_douczony)
    migawki = [{k: v.clone() for k, v in m.state_dict().items()} for m in modele]

    zrob(ctx)

    for model, migawka in zip(modele, migawki, strict=True):
        po = model.state_dict()
        assert po.keys() == migawka.keys()
        for klucz, tensor in migawka.items():
            assert torch.equal(po[klucz], tensor)
        assert model.training is False
