"""Testy KVCache, filter_logits i generate (próbkowanie, KV cache)."""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from nanowriter.model import GPT, Block, CausalSelfAttention, GPTConfig, KVCache
from nanowriter.sample import filter_logits, generate

SMALL = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, d_model=32)
NEG_INF = float("-inf")


def _gpt() -> GPT:
    torch.manual_seed(0)
    return GPT(SMALL).eval()


def _idx(batch: int, length: int) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, 64, (batch, length))


def _dummy(row0: list[float] | torch.Tensor) -> nn.Embedding:
    """Atrapa: wszystkie wiersze zerowe, wiersz 0 ma podane logity."""
    row = torch.as_tensor(row0, dtype=torch.float32)
    v = row.shape[0]
    model = nn.Embedding(v, v)
    with torch.no_grad():
        model.weight.zero_()
        model.weight[0] = row
    return model


def _random_model(v: int) -> nn.Embedding:
    torch.manual_seed(0)
    return nn.Embedding(v, v)


class _Recorder(nn.Module):
    def __init__(self, gpt: GPT) -> None:
        super().__init__()
        self.gpt = gpt
        self.calls: list[tuple[int, KVCache | None]] = []
        self.cache_lengths: list[int | None] = []

    def forward(self, idx: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        self.calls.append((idx.shape[1], cache))
        self.cache_lengths.append(None if cache is None else cache.length)
        return self.gpt(idx, cache=cache)


def _reference(model, idx, n, generator, temperature, top_k, top_p):
    with torch.no_grad():
        seq = idx.clone()
        for _ in range(n):
            logits = model(seq)[:, -1, :]
            logits = logits / temperature
            logits = filter_logits(logits, top_k, top_p)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1, generator=generator)
            seq = torch.cat([seq, nxt], dim=1)
    return seq


def _kv(t: int, b: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(b, 2, t, 4), torch.randn(b, 2, t, 4)


# ---------------------------------------------------------------- KVCache


def test_kvcache_empty():
    assert KVCache().length == 0
    assert type(KVCache().length) is int
    assert not isinstance(KVCache(), nn.Module)


def test_kvcache_update_first_layer():
    c = KVCache()
    k, v = _kv(3)
    rk, rv = c.update(0, k, v)
    assert rk.shape == rv.shape == (1, 2, 3, 4)
    assert torch.equal(rk, k)
    assert torch.equal(rv, v)
    assert c.length == 3


def test_kvcache_update_appends():
    c = KVCache()
    k, v = _kv(3)
    c.update(0, k, v)
    k2, v2 = _kv(1)
    rk, rv = c.update(0, k2, v2)
    assert rk.shape == (1, 2, 4, 4)
    assert torch.equal(rk, torch.cat([k, k2], dim=2))
    assert torch.equal(rv, torch.cat([v, v2], dim=2))
    assert c.length == 4


def test_kvcache_layers_separate():
    c = KVCache()
    c.update(0, *_kv(3))
    k1, v1 = _kv(5)
    rk, rv = c.update(1, k1, v1)
    assert rk.shape == (1, 2, 5, 4)
    assert torch.equal(rk, k1)
    assert torch.equal(rv, v1)
    assert c.length == 3
    rk, rv = c.update(1, *_kv(1))
    assert rk.shape == (1, 2, 6, 4)
    assert c.length == 3


@pytest.mark.parametrize(
    ("prefilled", "layer"),
    [(False, 1), (True, 2), (False, -1), (True, -1)],
)
def test_kvcache_update_missing_layer_raises(prefilled, layer):
    c = KVCache()
    if prefilled:
        c.update(0, *_kv(3))
    with pytest.raises(ValueError):
        c.update(layer, *_kv(3))


def test_kvcache_reset():
    c = KVCache()
    c.update(0, *_kv(3))
    c.update(1, *_kv(3))
    c.reset()
    assert c.length == 0
    with pytest.raises(ValueError):
        c.update(1, *_kv(1))
    k, v = _kv(1)
    rk, rv = c.update(0, k, v)
    assert rk.shape == (1, 2, 1, 4)
    assert torch.equal(rk, k)
    assert torch.equal(rv, v)
    assert c.length == 1


# ------------------------------------------------------ uwaga i blok z cache


def test_attention_cache_chunks_match_full():
    torch.manual_seed(0)
    a = CausalSelfAttention(SMALL)
    x = torch.randn(2, 7, 32)
    c = KVCache()
    with torch.no_grad():
        out = torch.cat(
            [a(x[:, :4], cache=c, layer=0), a(x[:, 4:], cache=c, layer=0)], dim=1
        )
        full = a(x)
    assert torch.allclose(out, full, atol=1e-5)
    assert c.length == 7


def test_attention_cache_single_token():
    torch.manual_seed(0)
    a = CausalSelfAttention(SMALL)
    x = torch.randn(1, 1, 32)
    with torch.no_grad():
        out = a(x, cache=KVCache())
        full = a(x)
    assert out.shape == (1, 1, 32)
    assert torch.allclose(out, full, atol=1e-5)


def test_attention_cache_layers_independent():
    torch.manual_seed(0)
    a0 = CausalSelfAttention(SMALL)
    a1 = CausalSelfAttention(SMALL)
    x = torch.randn(1, 7, 32)
    c = KVCache()
    with torch.no_grad():
        o0a = a0(x[:, :4], cache=c, layer=0)
        o1a = a1(x[:, :4], cache=c, layer=1)
        o0b = a0(x[:, 4:], cache=c, layer=0)
        o1b = a1(x[:, 4:], cache=c, layer=1)
        full0 = a0(x)
        full1 = a1(x)
    assert torch.allclose(torch.cat([o0a, o0b], dim=1), full0, atol=1e-5)
    assert torch.allclose(torch.cat([o1a, o1b], dim=1), full1, atol=1e-5)
    assert c.length == 7


def test_block_cache_chunks_match_full():
    torch.manual_seed(0)
    b = Block(SMALL)
    x = torch.randn(1, 3, 32)
    c = KVCache()
    with torch.no_grad():
        out = torch.cat(
            [b(x[:, :2], cache=c, layer=0), b(x[:, 2:], cache=c, layer=0)], dim=1
        )
        full = b(x)
    assert torch.allclose(out, full, atol=1e-5)
    assert c.length == 3

    torch.manual_seed(0)
    default_block = Block(GPTConfig())
    with torch.no_grad():
        res = default_block(torch.randn(1, 3, 128), cache=KVCache())
    assert res.shape == (1, 3, 128)


# -------------------------------------------------------- GPT.forward z cache


def test_gpt_forward_without_cache_unchanged():
    m = _gpt()
    idx = _idx(2, 16)
    with torch.no_grad():
        a = m(idx)
        b = m(idx, cache=None)
    assert torch.allclose(a, b, atol=1e-5)
    assert a.shape == (2, 16, 64)
    assert a.dtype == torch.float32


@pytest.mark.parametrize(
    "splits", [[5, 4, 3], [12], [1] * 12, [11, 1], [1, 11]]
)
def test_gpt_cache_chunks_match_full(splits):
    m = _gpt()
    idx = _idx(2, 12)
    c = KVCache()
    outs = []
    start = 0
    with torch.no_grad():
        for n in splits:
            out = m(idx[:, start : start + n], cache=c)
            start += n
            assert out.shape == (2, n, 64)
            assert out.dtype == torch.float32
            assert c.length == start
            outs.append(out)
        full = m(idx)
    assert torch.allclose(torch.cat(outs, dim=1), full, atol=1e-5)


def test_gpt_cache_token_by_token_full_block():
    m = _gpt()
    idx = _idx(1, 16)
    c = KVCache()
    with torch.no_grad():
        outs = [m(idx[:, t : t + 1], cache=c) for t in range(16)]
        full = m(idx)
    assert torch.allclose(torch.cat(outs, dim=1), full, atol=1e-5)
    assert c.length == 16


def test_gpt_cache_fills_every_layer():
    m = _gpt()
    idx = _idx(2, 5)
    c = KVCache()
    with torch.no_grad():
        m(idx, cache=c)
    rk, rv = c.update(1, torch.randn(2, 2, 1, 16), torch.randn(2, 2, 1, 16))
    assert rk.shape == (2, 2, 6, 16)
    assert rv.shape == (2, 2, 6, 16)
    k = torch.randn(2, 2, 1, 16)
    v = torch.randn(2, 2, 1, 16)
    rk, rv = c.update(2, k, v)
    assert rk.shape == (2, 2, 1, 16)
    assert torch.equal(rk, k)
    assert torch.equal(rv, v)
    assert c.length == 5


def test_gpt_cache_reaches_block_size():
    m = _gpt()
    idx = _idx(1, 16)
    c = KVCache()
    with torch.no_grad():
        m(idx[:, :10], cache=c)
        m(idx[:, 10:], cache=c)
    assert c.length == 16


@pytest.mark.parametrize(("filled", "extra"), [(16, 1), (10, 7)])
def test_gpt_cache_overflow_raises(filled, extra):
    m = _gpt()
    c = KVCache()
    with torch.no_grad():
        m(_idx(1, filled), cache=c)
        assert c.length == filled
        with pytest.raises(ValueError, match="16"):
            m(torch.zeros(1, extra, dtype=torch.long), cache=c)
    assert c.length == filled


@pytest.mark.parametrize(
    ("shape", "use_cache"),
    [((1, 0), True), ((5,), True), ((1, 0), False), ((1, 17), False)],
)
def test_gpt_cache_invalid_idx_raises(shape, use_cache):
    m = _gpt()
    idx = torch.zeros(*shape, dtype=torch.long)
    with torch.no_grad(), pytest.raises(ValueError):
        if use_cache:
            m(idx, cache=KVCache())
        else:
            m(idx)


def test_gpt_state_dict_keys_unchanged():
    m = GPT(SMALL)
    with torch.no_grad():
        m(_idx(1, 5), cache=KVCache())
    expected = {
        "tok_emb.weight",
        "pos_emb.weight",
        "ln_f.weight",
        "ln_f.bias",
        "lm_head.weight",
    }
    for i in range(2):
        for name in (
            "ln1.weight",
            "ln1.bias",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "ln2.weight",
            "ln2.bias",
            "mlp.fc.weight",
            "mlp.fc.bias",
            "mlp.proj.weight",
            "mlp.proj.bias",
        ):
            expected.add(f"blocks.{i}.{name}")
    assert set(m.state_dict().keys()) == expected
    assert [n for n, _ in m.named_buffers()] == ["blocks.0.attn.mask", "blocks.1.attn.mask"]
    assert sum(p.numel() for p in m.parameters()) == 28032


# ------------------------------------------------------------ filter_logits


def test_filter_top_k_example():
    out = filter_logits(torch.tensor([[1.0, 3.0, 2.0, 0.0]]), top_k=2)
    assert out.shape == (1, 4)
    assert out.dtype == torch.float32
    assert torch.isneginf(out[0, 0])
    assert torch.isneginf(out[0, 3])
    assert out[0, 1].item() == 3.0
    assert out[0, 2].item() == 2.0


@pytest.mark.parametrize(
    ("row", "top_k", "expected"),
    [
        ([1.0, 3.0, 3.0, 2.0], 1, [NEG_INF, 3.0, 3.0, NEG_INF]),
        ([3.0, 2.0, 2.0, 0.0], 2, [3.0, 2.0, 2.0, NEG_INF]),
    ],
)
def test_filter_top_k_ties_kept(row, top_k, expected):
    out = filter_logits(torch.tensor([row]), top_k=top_k)
    assert out.tolist() == [expected]


@pytest.mark.parametrize("top_k", [4, 10])
def test_filter_top_k_at_least_vocab_unchanged(top_k):
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.0]])
    out = filter_logits(logits, top_k=top_k)
    assert torch.equal(out, logits)
    assert not torch.isneginf(out).any()


@pytest.mark.parametrize(
    ("top_p", "kept"), [(0.4, [0]), (0.6, [0, 1]), (0.9, [0, 1, 2])]
)
def test_filter_top_p_examples(top_p, kept):
    logits = torch.tensor([0.5, 0.3, 0.2]).log()[None]
    out = filter_logits(logits, top_p=top_p)
    for i in range(3):
        if i in kept:
            assert out[0, i].item() == logits[0, i].item()
        else:
            assert torch.isneginf(out[0, i])


def test_filter_top_p_unsorted_row():
    logits = torch.tensor([0.2, 0.5, 0.3]).log()[None]
    out = filter_logits(logits, top_p=0.6)
    assert torch.isneginf(out[0, 0])
    assert out[0, 1].item() == logits[0, 1].item()
    assert out[0, 2].item() == logits[0, 2].item()


def test_filter_top_p_keeps_most_likely():
    out = filter_logits(torch.tensor([[0.0, 5.0, 1.0, 2.0]]), top_p=1e-6)
    assert out[0, 1].item() == 5.0
    assert torch.isneginf(out[0, 0])
    assert torch.isneginf(out[0, 2])
    assert torch.isneginf(out[0, 3])


def test_filter_top_p_one_unchanged():
    logits = torch.tensor([[0.0, 10.0, -10.0, 1.0]])
    out = filter_logits(logits, top_p=1.0)
    assert torch.equal(out, logits)
    assert not torch.isneginf(out).any()


def test_filter_top_k_then_top_p():
    logits = torch.tensor([0.4, 0.3, 0.2, 0.1]).log()[None]
    out = filter_logits(logits, top_k=2, top_p=0.5)
    assert out[0, 0].item() == logits[0, 0].item()
    for i in (1, 2, 3):
        assert torch.isneginf(out[0, i])


def test_filter_none_returns_copy():
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.0]])
    out = filter_logits(logits)
    assert torch.equal(out, logits)
    assert out is not logits
    out[0, 0] = 100.0
    assert logits.tolist() == [[1.0, 3.0, 2.0, 0.0]]


@pytest.mark.parametrize(
    "kwargs", [{"top_k": 2}, {"top_p": 0.5}, {"top_k": 3, "top_p": 0.5}]
)
def test_filter_does_not_modify_input(kwargs):
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.0], [0.0, 1.0, 2.0, 3.0]])
    before = logits.clone()
    filter_logits(logits, **kwargs)
    assert torch.equal(logits, before)


def test_filter_rows_independent():
    out = filter_logits(
        torch.tensor([[1.0, 3.0, 2.0, 0.0], [0.0, 1.0, 2.0, 3.0]]), top_k=2
    )
    assert out.tolist() == [[NEG_INF, 3.0, 2.0, NEG_INF], [NEG_INF, NEG_INF, 2.0, 3.0]]

    logits = torch.tensor([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]]).log()
    out = filter_logits(logits, top_p=0.6)
    assert out[0, 0].item() == logits[0, 0].item()
    assert out[0, 1].item() == logits[0, 1].item()
    assert torch.isneginf(out[0, 2])
    assert torch.isneginf(out[1, 0])
    assert out[1, 1].item() == logits[1, 1].item()
    assert out[1, 2].item() == logits[1, 2].item()


@pytest.mark.parametrize(
    "kwargs",
    [{"top_k": 0}, {"top_k": -1}, {"top_p": 0.0}, {"top_p": -0.1}, {"top_p": 1.5}],
)
def test_filter_invalid_args(kwargs):
    with pytest.raises(ValueError):
        filter_logits(torch.tensor([[1.0, 3.0, 2.0, 0.0]]), **kwargs)


@pytest.mark.parametrize("logits", [torch.tensor([1.0, 3.0, 2.0]), torch.zeros(1, 2, 3)])
def test_filter_invalid_dims(logits):
    with pytest.raises(ValueError):
        filter_logits(logits)


def test_filter_valid_boundaries():
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.0]])
    assert filter_logits(logits, top_k=1).tolist() == [[NEG_INF, 3.0, NEG_INF, NEG_INF]]
    assert torch.equal(filter_logits(logits, top_p=1.0), logits)


# ------------------------------------------------------- generate: próbkowanie


def test_generate_defaults_match_reference():
    model = _random_model(8)
    idx = torch.tensor([[1, 2], [3, 4]])
    expected = _reference(model, idx, 10, torch.Generator().manual_seed(0), 1.0, None, None)
    out = generate(model, idx, 10, generator=torch.Generator().manual_seed(0))
    assert torch.equal(out, expected)
    out = generate(
        model,
        idx,
        10,
        generator=torch.Generator().manual_seed(0),
        temperature=1.0,
        top_k=None,
        top_p=None,
        use_cache=False,
    )
    assert torch.equal(out, expected)


@pytest.mark.parametrize(
    ("temperature", "top_k", "top_p"),
    [(0.7, None, None), (1.0, 3, None), (1.0, None, 0.8), (0.7, 3, 0.8), (1.5, 5, 0.9)],
)
def test_generate_temperature_and_filters_match_reference(temperature, top_k, top_p):
    model = _random_model(8)
    idx = torch.tensor([[1], [5]])
    expected = _reference(
        model, idx, 20, torch.Generator().manual_seed(0), temperature, top_k, top_p
    )
    out = generate(
        model,
        idx,
        20,
        generator=torch.Generator().manual_seed(0),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    assert torch.equal(out, expected)


def test_generate_temperature_zero_is_argmax():
    model = _random_model(8)
    idx = torch.tensor([[1], [3]])
    rows = []
    for start in (1, 3):
        seq = [start]
        for _ in range(10):
            seq.append(int(model.weight[seq[-1]].argmax()))
        rows.append(seq)
    expected = torch.tensor(rows)
    for gen in (torch.Generator().manual_seed(0), torch.Generator().manual_seed(1), None):
        out = generate(model, idx, 10, generator=gen, temperature=0)
        assert out.shape == (2, 11)
        assert out.dtype == torch.int64
        assert torch.equal(out, expected)


def test_generate_temperature_zero_does_not_use_generator():
    model = _random_model(8)
    g = torch.Generator().manual_seed(0)
    state = g.get_state().clone()
    generate(model, torch.tensor([[1]]), 10, generator=g, temperature=0)
    assert torch.equal(g.get_state(), state)


def test_generate_top_k_one_equals_argmax():
    model = _random_model(8)
    idx = torch.tensor([[1], [3]])
    a = generate(model, idx, 15, generator=torch.Generator().manual_seed(0), top_k=1)
    b = generate(model, idx, 15, temperature=0)
    assert torch.equal(a, b)


def test_generate_top_k_restricts_tokens():
    model = _dummy([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    idx = torch.zeros(2000, 1, dtype=torch.long)
    out = generate(model, idx, 1, generator=torch.Generator().manual_seed(0), top_k=3)
    new = out[:, 1]
    assert set(new.tolist()) == {3, 4, 5}
    share = (new == 5).float().mean().item()
    assert 0.62 <= share <= 0.71


def test_generate_top_p_restricts_tokens():
    model = _dummy(torch.tensor([0.5, 0.3, 0.15, 0.05]).log())
    idx = torch.zeros(2000, 1, dtype=torch.long)
    out = generate(model, idx, 1, generator=torch.Generator().manual_seed(0), top_p=0.6)
    new = out[:, 1]
    assert set(new.tolist()) == {0, 1}
    share = (new == 0).float().mean().item()
    assert 0.57 <= share <= 0.68


@pytest.mark.parametrize(
    ("temperature", "low", "high"), [(0.5, 0.87, 0.93), (2.0, 0.59, 0.68)]
)
def test_generate_low_temperature_concentrates(temperature, low, high):
    model = _dummy([math.log(3), 0.0, -100.0])
    idx = torch.zeros(2000, 1, dtype=torch.long)
    out = generate(
        model, idx, 1, generator=torch.Generator().manual_seed(0), temperature=temperature
    )
    new = out[:, 1]
    share = (new == 0).float().mean().item()
    assert low <= share <= high
    assert (new == 2).sum().item() == 0


def test_generate_new_args_keyword_only():
    with pytest.raises(TypeError):
        generate(_random_model(5), torch.tensor([[1]]), 2, None, None, 0.5)


@pytest.mark.parametrize("max_new_tokens", [0, 3])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": -0.5},
        {"top_k": 0},
        {"top_k": -1},
        {"top_p": 0.0},
        {"top_p": -0.1},
        {"top_p": 1.5},
    ],
)
def test_generate_invalid_new_args(kwargs, max_new_tokens):
    with pytest.raises(ValueError):
        generate(_random_model(5), torch.tensor([[1]]), max_new_tokens, **kwargs)
    with pytest.raises(ValueError):
        generate(_gpt(), torch.tensor([[1]]), 0, use_cache=True, **kwargs)


@pytest.mark.parametrize(
    "kwargs", [{"temperature": 0}, {"top_k": 1}, {"top_p": 1.0}, {"top_k": 100}]
)
def test_generate_valid_boundary_args(kwargs):
    out = generate(
        _random_model(5),
        torch.tensor([[1]]),
        3,
        generator=torch.Generator().manual_seed(0),
        **kwargs,
    )
    assert out.shape == (1, 4)
    assert ((out[:, 1:] >= 0) & (out[:, 1:] < 5)).all()


def test_generate_valid_boundary_zero_new_tokens():
    idx = _idx(1, 3)
    out = generate(
        _gpt(), idx, 0, temperature=0, top_k=2, top_p=0.5, use_cache=True
    )
    assert torch.equal(out, idx)
    assert out is not idx


# ------------------------------------------------- generate: KV cache (bramka)


def _run(prompt, n, block_size, use_cache, temperature, top_k, top_p, model=None):
    return generate(
        _gpt() if model is None else model,
        prompt,
        n,
        generator=torch.Generator().manual_seed(0),
        block_size=block_size,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        use_cache=use_cache,
    )


@pytest.mark.parametrize(
    ("batch", "length", "n", "block_size", "temperature", "top_k", "top_p"),
    [
        (1, 3, 10, 16, 1.0, None, None),
        (2, 3, 30, 16, 0.8, 10, 0.9),
        (1, 3, 30, 16, 1.0, None, None),
        (2, 20, 10, 16, 0.8, 10, 0.9),
        (1, 16, 5, 16, 1.0, None, None),
        (1, 1, 15, 16, 1.0, None, None),
        (1, 1, 16, 16, 1.0, None, None),
        (2, 3, 30, 16, 0, None, None),
        (2, 3, 10, None, 0.8, 10, 0.9),
    ],
)
def test_generate_cache_identical(batch, length, n, block_size, temperature, top_k, top_p):
    prompt = _idx(batch, length)
    args = (prompt, n, block_size)
    without = _run(*args, False, temperature, top_k, top_p)
    with_cache = _run(*args, True, temperature, top_k, top_p)
    assert torch.equal(without, with_cache)
    assert with_cache.shape == (batch, length + n)
    assert with_cache.dtype == torch.int64
    assert torch.equal(with_cache[:, :length], prompt)
    new = with_cache[:, length:]
    assert ((new >= 0) & (new < 64)).all()


def test_generate_cache_same_seed_repeatable():
    prompt = _idx(2, 3)
    model = _gpt()

    def run(seed):
        return generate(
            model,
            prompt,
            30,
            generator=torch.Generator().manual_seed(seed),
            block_size=16,
            temperature=0.8,
            top_k=10,
            top_p=0.9,
            use_cache=True,
        )

    assert torch.equal(run(0), run(0))
    assert not torch.equal(run(0), run(1))


def test_generate_cache_feeds_model():
    prompt = _idx(1, 3)
    rec = _Recorder(_gpt())
    generate(
        rec, prompt, 20, generator=torch.Generator().manual_seed(0), block_size=16, use_cache=True
    )
    assert [c[0] for c in rec.calls] == [3] + [1] * 13 + [16] * 6
    assert rec.cache_lengths == [0, *range(3, 16), 0, 0, 0, 0, 0, 0]
    caches = [c[1] for c in rec.calls]
    assert all(isinstance(c, KVCache) for c in caches)
    assert all(c is caches[0] for c in caches)

    rec2 = _Recorder(_gpt())
    generate(
        rec2,
        prompt,
        20,
        generator=torch.Generator().manual_seed(0),
        block_size=16,
        use_cache=False,
    )
    assert [c[0] for c in rec2.calls] == [*range(3, 17), 16, 16, 16, 16, 16, 16]
    assert all(c[1] is None for c in rec2.calls)

    rec3 = _Recorder(_gpt())
    for _ in range(2):
        generate(rec3, prompt, 2, block_size=16, use_cache=True)
    assert rec3.calls[0][1] is not rec3.calls[2][1]


def test_generate_cache_does_not_modify_idx():
    prompt = _idx(1, 3)
    before = prompt.clone()
    generate(
        _gpt(),
        prompt,
        20,
        generator=torch.Generator().manual_seed(0),
        block_size=16,
        use_cache=True,
    )
    assert torch.equal(prompt, before)
    assert prompt.shape == (1, 3)


@pytest.mark.parametrize("train", [True, False])
def test_generate_cache_keeps_model_mode(train):
    model = _gpt()
    model.train(train)
    generate(model, _idx(1, 3), 5, block_size=16, use_cache=True)
    assert model.training is train


# ------------------------------------------------------------------------ GPU


@pytest.mark.skipif(not torch.cuda.is_available(), reason="brak GPU")
def test_cache_on_cuda():
    torch.manual_seed(0)
    m = GPT(SMALL).to("cuda").eval()
    idx = torch.randint(0, 64, (2, 12)).to("cuda")
    c = KVCache()
    outs = []
    start = 0
    with torch.no_grad():
        for n in (5, 4, 3):
            outs.append(m(idx[:, start : start + n], cache=c))
            start += n
        full = m(idx)
    out = torch.cat(outs, dim=1)
    assert torch.allclose(out, full, atol=1e-4)
    assert out.device.type == "cuda"
    assert c.length == 12

    prompt = torch.randint(0, 64, (2, 3)).to("cuda")

    def run(use_cache):
        return generate(
            m,
            prompt,
            30,
            generator=torch.Generator(device="cuda").manual_seed(0),
            block_size=16,
            temperature=0.8,
            top_k=10,
            top_p=0.9,
            use_cache=use_cache,
        )

    a = run(True)
    b = run(False)
    assert torch.equal(a, b)
    assert a.device.type == "cuda"
    assert a.shape == (2, 33)
