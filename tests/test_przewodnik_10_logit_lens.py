import dataclasses
import json
import math
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from nanowriter.model import GPT, GPTConfig
from przewodnik.logit_lens import zrob
from przewodnik.proba import maly_kontekst_i
from przewodnik.wspolne import ZDANIE, etykieta_tokenu, zakoduj_zdanie

NAZWY = ["logit_lens.gif", "logit_lens.png", "logit_lens_podsumowanie.png", "logit_lens.json"]


def wszystkie_haki_puste(model: GPT) -> bool:
    moduly = [model.ln_f, model.lm_head]
    for blok in model.blocks:
        moduly += [blok, blok.attn, blok.mlp]
    return all(not m._forward_hooks and not m._forward_pre_hooks for m in moduly)


def normy_odniesienia(model: GPT, ids: torch.Tensor) -> list[dict[str, float]]:
    """Normy wkładów liczone własnymi hakami, niezależnie od kodu materiału."""
    out_attn: list[torch.Tensor] = [None] * len(model.blocks)
    out_mlp: list[torch.Tensor] = [None] * len(model.blocks)
    out_blok: list[torch.Tensor] = [None] * len(model.blocks)

    def zapis(cel, i):
        def hak(_modul, _args, wyjscie):
            cel[i] = wyjscie.detach()

        return hak

    uchwyty = []
    for i, blok in enumerate(model.blocks):
        uchwyty.append(blok.attn.register_forward_hook(zapis(out_attn, i)))
        uchwyty.append(blok.mlp.register_forward_hook(zapis(out_mlp, i)))
        uchwyty.append(blok.register_forward_hook(zapis(out_blok, i)))
    try:
        with torch.no_grad():
            model(ids)
    finally:
        for u in uchwyty:
            u.remove()
    return [
        {
            "uwaga": float(out_attn[i][0].norm(dim=-1).mean()),
            "mlp": float(out_mlp[i][0].norm(dim=-1).mean()),
            "reszta": float(out_blok[i][0].norm(dim=-1).mean()),
        }
        for i in range(len(model.blocks))
    ]


@pytest.fixture(scope="module")
def wynik(tmp_path_factory):
    ctx = maly_kontekst_i(tmp_path_factory.mktemp("logit_lens"))
    ids = zakoduj_zdanie(ctx.tokenizer, ZDANIE, 256)
    T = ids.shape[1]
    with torch.no_grad():
        logity = ctx.model(ids)[0]
        prob = torch.softmax(logity, dim=-1)
        x0 = ctx.model.tok_emb(ids) + ctx.model.pos_emb(torch.arange(T))
        logity0 = ctx.model.lm_head(ctx.model.ln_f(x0))[0]
    normy = normy_odniesienia(ctx.model, ids)
    pliki = zrob(ctx)
    dane = json.loads(pliki[3].read_text(encoding="utf-8"))
    return SimpleNamespace(
        ctx=ctx, ids=ids, T=T, prob=prob, logity0=logity0, normy=normy, pliki=pliki, dane=dane
    )


def test_pliki_nazwy_kolejnosc_i_format(wynik):
    assert [p.name for p in wynik.pliki] == NAZWY
    for p in wynik.pliki:
        assert p.parent == wynik.ctx.wyniki_dir
        assert p.exists()
        assert p.stat().st_size > 0

    gif, png1, png2, js = wynik.pliki
    assert gif.read_bytes()[:4] == b"GIF8"
    assert png1.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert png2.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    surowe = js.read_bytes()
    assert surowe.endswith(b"\n")
    assert b"\r\n" not in surowe


def test_json_struktura(wynik):
    dane, ctx, T = wynik.dane, wynik.ctx, wynik.T
    assert dane["material"] == "logit_lens"
    assert dane["zdanie"] == ZDANIE
    assert dane["tokeny"] == [etykieta_tokenu(ctx.tokenizer, i) for i in wynik.ids[0].tolist()]
    assert len(dane["etapy"]) == 3
    assert [e["etap"] for e in dane["etapy"]] == [0, 1, 2]
    assert dane["etapy"][0]["opis"] == "po embeddingach (zanim cokolwiek policzono)"
    assert dane["etapy"][1]["opis"] == "po bloku 1 z 2"
    assert dane["etapy"][2]["opis"] == "po bloku 2 z 2"
    for etap in dane["etapy"]:
        assert len(etap["pozycje"]) == T
        for poz in etap["pozycje"]:
            assert "top3" in poz
            assert "p_prawdziwego" in poz
            top3 = poz["top3"]
            assert len(top3) == 3
            for el in top3:
                assert "token" in el
                assert "p" in el
                assert 0 < el["p"] <= 1
            ps = [el["p"] for el in top3]
            assert ps == sorted(ps, reverse=True)


def test_p_prawdziwego_i_srednie(wynik):
    dane, T = wynik.dane, wynik.T
    for etap in dane["etapy"]:
        assert etap["pozycje"][T - 1]["p_prawdziwego"] is None
        for t in range(T - 1):
            assert 0 < etap["pozycje"][t]["p_prawdziwego"] <= 1
    assert len(dane["srednia_p_prawdziwego"]) == 3
    for s, etap in enumerate(dane["etapy"]):
        wartosci = [etap["pozycje"][t]["p_prawdziwego"] for t in range(T - 1)]
        assert dane["srednia_p_prawdziwego"][s] == pytest.approx(
            sum(wartosci) / len(wartosci), abs=1e-6
        )


def test_ostatni_etap_zgodny_z_modelem(wynik):
    etap = wynik.dane["etapy"][2]
    ctx, ids, prob, T = wynik.ctx, wynik.ids, wynik.prob, wynik.T
    for t in range(T):
        top = etap["pozycje"][t]["top3"][0]
        assert top["token"] == etykieta_tokenu(ctx.tokenizer, int(prob[t].argmax()))
        assert top["p"] == pytest.approx(float(prob[t].max()), abs=1e-4)
        if t < T - 1:
            assert etap["pozycje"][t]["p_prawdziwego"] == pytest.approx(
                float(prob[t, ids[0, t + 1]]), abs=1e-4
            )


def test_etap_zero_zgodny_z_odniesieniem(wynik):
    etap = wynik.dane["etapy"][0]
    ctx, ids, logity0, T = wynik.ctx, wynik.ids, wynik.logity0, wynik.T
    for t in range(T):
        p0 = torch.softmax(logity0[t], dim=-1)
        top = etap["pozycje"][t]["top3"][0]
        assert top["token"] == etykieta_tokenu(ctx.tokenizer, int(logity0[t].argmax()))
        assert top["p"] == pytest.approx(float(p0.max()), abs=1e-4)
        if t < T - 1:
            assert etap["pozycje"][t]["p_prawdziwego"] == pytest.approx(
                float(p0[ids[0, t + 1]]), abs=1e-4
            )


def test_normy_blokow(wynik):
    normy = wynik.dane["normy_blokow"]
    assert len(normy) == 2
    for w in normy:
        assert {"blok", "uwaga", "mlp", "reszta"} <= set(w)
    numery = [w["blok"] for w in normy]
    assert numery in ([0, 1], [1, 2])
    for w, ref in zip(normy, wynik.normy, strict=True):
        for klucz in ("uwaga", "mlp", "reszta"):
            assert w[klucz] > 0
            assert w[klucz] == pytest.approx(ref[klucz], abs=1e-4, rel=1e-4)
            assert math.isfinite(w[klucz])


def test_gif_liczba_klatek(wynik):
    with Image.open(wynik.pliki[0]) as gif:
        assert gif.n_frames == 3


def test_model_z_trzema_warstwami(tmp_path):
    ctx = maly_kontekst_i(tmp_path)
    torch.manual_seed(0)
    model3 = GPT(
        GPTConfig(
            vocab_size=ctx.tokenizer.vocab_size, block_size=256, n_layer=3, n_head=2, d_model=32
        )
    ).eval()
    ctx3 = dataclasses.replace(ctx, model=model3)

    pliki = zrob(ctx3)

    dane = json.loads(pliki[3].read_text(encoding="utf-8"))
    assert len(dane["etapy"]) == 4
    assert len(dane["normy_blokow"]) == 3
    assert len(dane["srednia_p_prawdziwego"]) == 4
    with Image.open(pliki[0]) as gif:
        assert gif.n_frames == 4
    assert dane["etapy"][-1]["opis"] == "po bloku 3 z 3"


def test_haki_zdjete_a_model_bez_zmian(tmp_path):
    ctx = maly_kontekst_i(tmp_path)
    ids = zakoduj_zdanie(ctx.tokenizer, ZDANIE, 256)
    przed = {k: v.clone() for k, v in ctx.model.state_dict().items()}
    with torch.no_grad():
        logity_przed = ctx.model(ids)

    zrob(ctx)

    assert wszystkie_haki_puste(ctx.model)
    po = ctx.model.state_dict()
    assert set(po) == set(przed)
    for k, v in przed.items():
        assert torch.equal(po[k], v)
    assert ctx.model.training is False
    for blok in ctx.model.blocks:
        assert blok.attn.trace is None
    with torch.no_grad():
        assert torch.equal(ctx.model(ids), logity_przed)


def test_haki_zdjete_po_wyjatku_modelu(tmp_path):
    ctx = maly_kontekst_i(tmp_path)

    def f(_modul, _args, _wyjscie):
        raise RuntimeError("boom")

    uchwyt = ctx.model.blocks[-1].register_forward_hook(f)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            zrob(ctx)
    finally:
        uchwyt.remove()
    assert wszystkie_haki_puste(ctx.model)
