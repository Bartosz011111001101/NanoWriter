import dataclasses
import json
import math

import pytest
import torch
from PIL import Image

from przewodnik.proba import maly_kontekst_i
from przewodnik.uwaga import _wybierz_glowice, zrob
from przewodnik.wspolne import ZDANIE, etykieta_tokenu, zakoduj_zdanie

PNG_NAGLOWEK = b"\x89PNG\r\n\x1a\n"
NAZWY = ["uwaga_krok_po_kroku.gif", "uwaga_krok_po_kroku.png", "uwaga_mapy.png", "uwaga.json"]


@dataclasses.dataclass
class Wynik:
    ctx: object
    ids: torch.Tensor
    sciezki: list
    dane: dict
    scores_ref: list
    weights_ref: list
    entropia_ref: list
    logity_ref: torch.Tensor
    stan_przed: dict
    trace_po: list
    training_po: bool


def _slad_odniesienia(ctx, ids):
    """Liczy odniesienie ze śladu modelu: logity, scores, weights i entropie."""
    for blok in ctx.model.blocks:
        blok.attn.trace = {}
    with torch.no_grad():
        logity = ctx.model(ids)
    scores = [blok.attn.trace["scores"][0] for blok in ctx.model.blocks]
    weights = [blok.attn.trace["weights"][0] for blok in ctx.model.blocks]
    for blok in ctx.model.blocks:
        blok.attn.trace = None
    entropie = []
    for w in weights:
        wiersz = []
        for h in range(w.shape[0]):
            wh = w[h]
            ent = -(wh * torch.log(wh.clamp(min=1e-12))).sum(-1)
            wiersz.append(float(ent[1:].mean()))
        entropie.append(wiersz)
    return logity, scores, weights, entropie


@pytest.fixture(scope="module")
def wynik(tmp_path_factory):
    ctx = maly_kontekst_i(tmp_path_factory.mktemp("uwaga"))
    ids = zakoduj_zdanie(ctx.tokenizer, ZDANIE, 256)
    logity_ref, scores_ref, weights_ref, entropia_ref = _slad_odniesienia(ctx, ids)
    stan_przed = {k: v.clone() for k, v in ctx.model.state_dict().items()}
    sciezki = zrob(ctx)
    dane = json.loads(sciezki[3].read_text(encoding="utf-8"))
    return Wynik(
        ctx=ctx,
        ids=ids,
        sciezki=sciezki,
        dane=dane,
        scores_ref=scores_ref,
        weights_ref=weights_ref,
        entropia_ref=entropia_ref,
        logity_ref=logity_ref,
        stan_przed=stan_przed,
        trace_po=[blok.attn.trace for blok in ctx.model.blocks],
        training_po=ctx.model.training,
    )


def _t(wynik):
    return wynik.ids.shape[1]


def test_pliki_nazwy_kolejnosc_i_format(wynik):
    sciezki = wynik.sciezki

    assert [p.name for p in sciezki] == NAZWY
    for p in sciezki:
        assert p.parent == wynik.ctx.wyniki_dir
        assert p.exists()
        assert p.stat().st_size > 0
    assert sciezki[0].read_bytes().startswith(b"GIF8")
    assert sciezki[1].read_bytes().startswith(PNG_NAGLOWEK)
    assert sciezki[2].read_bytes().startswith(PNG_NAGLOWEK)
    surowe = sciezki[3].read_bytes()
    assert surowe.endswith(b"\n")
    assert b"\r\n" not in surowe


def test_json_klucze(wynik):
    dane = wynik.dane
    t = _t(wynik)
    ctx = wynik.ctx

    assert dane["material"] == "uwaga"
    assert dane["zdanie"] == ZDANIE
    assert dane["tokeny"] == [etykieta_tokenu(ctx.tokenizer, i) for i in wynik.ids[0].tolist()]
    for klucz in ("wybrana_glowica", "entropia", "wyniki_qk", "maska_przyczynowa", "wagi"):
        assert klucz in dane
    assert set(dane["wybrana_glowica"]) == {"warstwa", "glowica"}

    assert len(dane["wagi"]) == 2
    for warstwa in dane["wagi"]:
        assert len(warstwa) == 2
        for glowica in warstwa:
            assert len(glowica) == t
            assert all(len(wiersz) == t for wiersz in glowica)
    assert len(dane["entropia"]) == 2
    assert all(len(wiersz) == 2 for wiersz in dane["entropia"])
    for macierz in (dane["wyniki_qk"], dane["maska_przyczynowa"]):
        assert len(macierz) == t
        assert all(len(wiersz) == t for wiersz in macierz)


def test_wagi_wiersze_sumuja_sie_do_jedynki_i_przyszlosc_zerowa(wynik):
    wagi = wynik.dane["wagi"]
    t = _t(wynik)

    for w in range(2):
        for h in range(2):
            for r in range(t):
                wiersz = wagi[w][h][r]
                assert math.isclose(sum(wiersz), 1.0, abs_tol=1e-3)
                assert all(x >= 0 for x in wiersz)
                for j in range(r + 1, t):
                    assert wiersz[j] == 0


def test_wagi_zgodne_ze_sladem_modelu(wynik):
    for w in range(2):
        for h in range(2):
            wagi = torch.tensor(wynik.dane["wagi"][w][h])
            assert torch.allclose(wagi, wynik.weights_ref[w][h], atol=1e-4)


def test_maska_przyczynowa(wynik):
    maska = wynik.dane["maska_przyczynowa"]
    t = _t(wynik)

    for r in range(t):
        for j in range(t):
            assert isinstance(maska[r][j], bool)
            assert maska[r][j] == (j <= r)


def test_entropia_i_wybrana_glowica(wynik):
    entropia = wynik.dane["entropia"]

    for w in range(2):
        for h in range(2):
            assert entropia[w][h] >= 0
            assert entropia[w][h] == pytest.approx(wynik.entropia_ref[w][h], abs=1e-3)

    najmniejsza = min(min(wiersz) for wiersz in entropia)
    w0, h0 = next((w, h) for w in range(2) for h in range(2) if entropia[w][h] == najmniejsza)
    assert wynik.dane["wybrana_glowica"] == {"warstwa": w0, "glowica": h0}


def test_wyniki_qk_wybranej_glowicy(wynik):
    dane = wynik.dane
    t = _t(wynik)
    w0 = dane["wybrana_glowica"]["warstwa"]
    h0 = dane["wybrana_glowica"]["glowica"]
    qk = dane["wyniki_qk"]

    assert len(qk) == t
    assert all(len(wiersz) == t for wiersz in qk)
    assert all(x is not None for wiersz in qk for x in wiersz)
    assert all(math.isfinite(x) for wiersz in qk for x in wiersz)
    assert torch.allclose(torch.tensor(qk), wynik.scores_ref[w0][h0], atol=1e-4)
    for r in range(t):
        softmax = torch.softmax(torch.tensor(qk[r][: r + 1]), dim=-1)
        oczekiwane = torch.tensor(dane["wagi"][w0][h0][r][: r + 1])
        assert torch.allclose(softmax, oczekiwane, atol=1e-3)


def test_gif_liczba_klatek(wynik):
    with Image.open(wynik.sciezki[0]) as gif:
        assert gif.n_frames == 3 * _t(wynik)


def test_slad_zdjety_i_logity_bez_zmian(wynik):
    ctx = wynik.ctx

    assert all(slad is None for slad in wynik.trace_po)
    assert all(blok.attn.trace is None for blok in ctx.model.blocks)
    stan_po = ctx.model.state_dict()
    assert set(stan_po) == set(wynik.stan_przed)
    for klucz, tensor in wynik.stan_przed.items():
        assert torch.equal(stan_po[klucz], tensor)
    assert ctx.model.training is False
    assert wynik.training_po is False
    with torch.no_grad():
        logity_po = ctx.model(wynik.ids)
    assert torch.equal(logity_po, wynik.logity_ref)


def test_slad_zdjety_po_wyjatku_modelu(tmp_path):
    ctx = maly_kontekst_i(tmp_path)

    def rzuc(*args, **kwargs):
        raise RuntimeError("boom")

    uchwyt = ctx.model.ln_f.register_forward_hook(rzuc)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            zrob(ctx)
    finally:
        uchwyt.remove()

    assert all(blok.attn.trace is None for blok in ctx.model.blocks)


def test_za_krotkie_zdanie_value_error(tmp_path, monkeypatch):
    ctx = maly_kontekst_i(tmp_path)
    monkeypatch.setattr(
        "przewodnik.uwaga.zakoduj_zdanie", lambda *args, **kwargs: torch.tensor([[5]])
    )

    with pytest.raises(ValueError):
        zrob(ctx)

    assert all(blok.attn.trace is None for blok in ctx.model.blocks)


def test_wybierz_glowice():
    assert _wybierz_glowice([[0.5, 0.2], [0.9, 0.2]]) == (0, 1)
    assert _wybierz_glowice([[1.0]]) == (0, 0)
    assert _wybierz_glowice([[0.3, 0.3], [0.3, 0.3]]) == (0, 0)
    assert _wybierz_glowice([[0.5], [0.1], [0.1]]) == (1, 0)
    assert _wybierz_glowice([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]]) == (1, 2)
    wynik = _wybierz_glowice([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]])
    assert isinstance(wynik, tuple)
    assert len(wynik) == 2
    assert all(isinstance(x, int) for x in wynik)
