import math

import torch

from nanowriter.model import GPT, CausalSelfAttention, GPTConfig, KVCache

KONFIG = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, d_model=32)
KLUCZE = {"scores", "masked", "weights"}


def nowy_gpt() -> GPT:
    torch.manual_seed(0)
    return GPT(KONFIG).eval()


def nowa_uwaga() -> CausalSelfAttention:
    torch.manual_seed(0)
    return CausalSelfAttention(KONFIG).eval()


def nowe_idx() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, 64, (2, 6))


def nowe_x(b: int = 2, t: int = 5) -> torch.Tensor:
    torch.manual_seed(2)
    return torch.randn(b, t, 32)


def przyszlosc(t: int) -> torch.Tensor:
    return torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)


def ustaw_slady(model: GPT) -> list[dict[str, torch.Tensor]]:
    slady: list[dict[str, torch.Tensor]] = [{} for _ in model.blocks]
    for blok, slad in zip(model.blocks, slady, strict=True):
        blok.attn.trace = slad
    return slady


def test_trace_domyslnie_none():
    assert CausalSelfAttention(GPTConfig()).trace is None
    assert CausalSelfAttention(GPTConfig()).head_dim == 32

    model = nowy_gpt()
    for blok in model.blocks:
        assert blok.attn.trace is None

    with torch.no_grad():
        model(nowe_idx())
    for blok in model.blocks:
        assert blok.attn.trace is None


def test_trace_nie_zmienia_wyniku_gpt():
    model = nowy_gpt()
    idx = nowe_idx()
    with torch.no_grad():
        a = model(idx)
        for blok in model.blocks:
            blok.attn.trace = {}
        b = model(idx)
        for blok in model.blocks:
            blok.attn.trace = None
        c = model(idx)

    assert a.shape == (2, 6, 64)
    assert torch.equal(a, b)
    assert torch.equal(a, c)


def test_trace_nie_zmienia_wyniku_uwagi():
    attn = nowa_uwaga()
    x = nowe_x()
    with torch.no_grad():
        a = attn(x)
        assert a.shape == (2, 5, 32)
        attn.trace = {}
        b = attn(x)
    assert torch.equal(a, b)

    attn = nowa_uwaga()
    x1 = nowe_x(1, 1)
    with torch.no_grad():
        wynik = attn(x1)
        assert wynik.shape == (1, 1, 32)
        attn.trace = {}
        attn(x1)
    trace = attn.trace
    assert trace is not None
    assert trace["weights"].shape == (1, 2, 1, 1)
    assert torch.allclose(trace["weights"], torch.ones(1, 2, 1, 1), atol=1e-6)
    assert torch.equal(trace["masked"], trace["scores"])


def test_trace_klucze_ksztalty_dtype():
    attn = nowa_uwaga()
    attn.trace = {}
    with torch.no_grad():
        attn(nowe_x(2, 5))
    assert set(attn.trace) == KLUCZE
    for tensor in attn.trace.values():
        assert tensor.shape == (2, 2, 5, 5)
        assert tensor.dtype == torch.float32

    attn.trace = {}
    with torch.no_grad():
        attn(nowe_x(1, 16))
    assert set(attn.trace) == KLUCZE
    for tensor in attn.trace.values():
        assert tensor.shape == (1, 2, 16, 16)


def test_trace_wagi_suma_i_maska():
    attn = nowa_uwaga()
    attn.trace = {}
    with torch.no_grad():
        attn(nowe_x(2, 5))
    s, m, w = attn.trace["scores"], attn.trace["masked"], attn.trace["weights"]
    p = przyszlosc(5)

    assert torch.allclose(w.sum(-1), torch.ones(2, 2, 5), atol=1e-5)
    assert (w[..., p] == 0).all()
    assert torch.isneginf(m[..., p]).all()
    assert torch.isfinite(m[..., ~p]).all()
    assert torch.equal(m[..., ~p], s[..., ~p])
    assert torch.isfinite(s).all()
    assert (w[..., ~p] > 0).all()


def test_trace_wagi_to_softmax_masked():
    attn = nowa_uwaga()
    attn.trace = {}
    with torch.no_grad():
        attn(nowe_x(2, 5))
    assert torch.allclose(
        attn.trace["weights"], torch.softmax(attn.trace["masked"], dim=-1), atol=1e-6
    )


def test_trace_scores_i_wynik_zgodne_z_przeliczeniem():
    attn = nowa_uwaga()
    x = nowe_x(2, 5)
    with torch.no_grad():
        q, k, v = attn.qkv(x).split(32, dim=-1)
        q = q.reshape(2, 5, 2, 16).transpose(1, 2)
        k = k.reshape(2, 5, 2, 16).transpose(1, 2)
        v = v.reshape(2, 5, 2, 16).transpose(1, 2)
        oczekiwane = q @ k.transpose(-2, -1) / math.sqrt(16)

        attn.trace = {}
        y = attn(x)
        trace = attn.trace
        assert torch.allclose(trace["scores"], oczekiwane, atol=1e-5)
        przeliczone = attn.proj((trace["weights"] @ v).transpose(1, 2).reshape(2, 5, 32))
        assert torch.allclose(y, przeliczone, atol=1e-5)


def test_trace_odlaczony_od_grafu():
    attn = nowa_uwaga()
    attn.trace = {}
    y = attn(nowe_x())
    for klucz in KLUCZE:
        assert attn.trace[klucz].requires_grad is False
        assert attn.trace[klucz].grad_fn is None
    assert y.requires_grad is True


def test_trace_slownik_nie_jest_podmieniany_i_klucze_nadpisywane():
    attn = nowa_uwaga()
    s: dict[str, torch.Tensor] = {}
    attn.trace = s
    with torch.no_grad():
        attn(nowe_x(2, 5))
    assert attn.trace is s
    assert s["weights"].shape == (2, 2, 5, 5)

    with torch.no_grad():
        attn(nowe_x(1, 3))
    assert attn.trace is s
    assert s["weights"].shape == (1, 2, 3, 3)
    assert s["scores"].shape == (1, 2, 3, 3)
    assert s["masked"].shape == (1, 2, 3, 3)
    assert set(s) == KLUCZE

    attn.trace = {}
    with torch.no_grad():
        attn(nowe_x(1, 3))
    assert torch.equal(s["weights"], attn.trace["weights"])


def test_trace_none_nie_zmienia_starego_slownika():
    attn = nowa_uwaga()
    s: dict[str, torch.Tensor] = {}
    attn.trace = s
    with torch.no_grad():
        attn(nowe_x(2, 5))
    kopia = {k: v.clone() for k, v in s.items()}

    attn.trace = None
    with torch.no_grad():
        attn(nowe_x(2, 4))

    assert set(s) == set(kopia)
    for klucz in kopia:
        assert torch.equal(s[klucz], kopia[klucz])
    assert attn.trace is None


def test_trace_w_gpt_kazdy_blok_osobno():
    model = nowy_gpt()
    slady = ustaw_slady(model)
    with torch.no_grad():
        model(nowe_idx())

    p = przyszlosc(6)
    for slad in slady:
        assert set(slad) == KLUCZE
        for tensor in slad.values():
            assert tensor.shape == (2, 2, 6, 6)
        w = slad["weights"]
        assert torch.allclose(w.sum(-1), torch.ones(2, 2, 6), atol=1e-5)
        assert (w[..., p] == 0).all()

    assert not torch.allclose(slady[0]["weights"], slady[1]["weights"])


def test_trace_z_cache_jedna_nowa_pozycja():
    model = nowy_gpt()
    c = KVCache()
    idx = nowe_idx()
    slady = ustaw_slady(model)

    with torch.no_grad():
        model(idx[:, :5], cache=c)
        for slad in slady:
            assert slad["weights"].shape == (2, 2, 5, 5)

        model(idx[:, 5:6], cache=c)

    for blok, slad in zip(model.blocks, slady, strict=True):
        assert blok.attn.trace is slad
        for klucz in KLUCZE:
            assert slad[klucz].shape == (2, 2, 1, 6)
        assert torch.isfinite(slad["masked"]).all()
        assert (slad["weights"] > 0).all()
        assert torch.allclose(slad["weights"].sum(-1), torch.ones(2, 2, 1), atol=1e-5)


def test_trace_z_cache_kilka_nowych_pozycji_i_zgodnosc_z_przebiegiem_bez_cache():
    model = nowy_gpt()
    idx = nowe_idx()[:, :5]

    with torch.no_grad():
        slady_pelne = ustaw_slady(model)
        model(idx)
        pelne = [slad["weights"].clone() for slad in slady_pelne]
        for p in pelne:
            assert p.shape == (2, 2, 5, 5)

        c = KVCache()
        slady = ustaw_slady(model)
        model(idx[:, :3], cache=c)
        model(idx[:, 3:5], cache=c)

    for i, slad in enumerate(slady):
        for klucz in KLUCZE:
            assert slad[klucz].shape == (2, 2, 2, 5)
        masked, weights = slad["masked"], slad["weights"]
        assert torch.isneginf(masked[..., 0, 4]).all()
        assert torch.isfinite(masked[..., 0, :4]).all()
        assert torch.isfinite(masked[..., 1, :]).all()
        assert (weights[..., 0, 4] == 0).all()
        assert torch.allclose(weights, pelne[i][..., 3:5, :], atol=1e-5)


def test_state_dict_i_parametry_bez_trace():
    model = GPT(KONFIG)
    assert not any("trace" in nazwa for nazwa in model.state_dict())
    assert not any("trace" in nazwa for nazwa, _ in model.named_parameters())
    assert not any("trace" in nazwa for nazwa, _ in model.named_buffers())

    przed = set(model.state_dict())
    model.eval()
    ustaw_slady(model)
    with torch.no_grad():
        model(nowe_idx())
    assert set(model.state_dict()) == przed

    assert sum(p.numel() for p in GPT(GPTConfig()).parameters()) == 957184
