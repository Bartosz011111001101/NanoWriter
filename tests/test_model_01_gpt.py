"""Testy GPT, GPTConfig i klocków transformera (zadanie 04, prompt 01)."""

import dataclasses
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from nanowriter.model import GPT, MLP, Block, CausalSelfAttention, GPTConfig

SMALL = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, d_model=32)


def _idx(batch: int, length: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(0, 64, (batch, length))


def _close(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-5) -> bool:
    return torch.allclose(a, b, atol=atol)


# --- GPTConfig ---


def test_config_defaults():
    cfg = GPTConfig()
    assert cfg.vocab_size == 1024
    assert cfg.block_size == 256
    assert cfg.n_layer == 4
    assert cfg.n_head == 4
    assert cfg.d_model == 128


def test_config_is_frozen_dataclass():
    assert dataclasses.is_dataclass(GPTConfig)
    cfg = GPTConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.n_layer = 3  # type: ignore[misc]


def test_config_asdict_roundtrip():
    assert GPTConfig(**dataclasses.asdict(SMALL)) == SMALL


@pytest.mark.parametrize("value", [0, -1])
@pytest.mark.parametrize(
    "field", ["vocab_size", "block_size", "n_layer", "n_head", "d_model"]
)
def test_config_field_below_one_raises(field, value):
    with pytest.raises(ValueError, match=field):
        GPTConfig(**{field: value})


def test_config_d_model_not_divisible_raises():
    with pytest.raises(ValueError, match="d_model"):
        GPTConfig(d_model=130, n_head=4)


def test_config_minimal_values_valid():
    GPTConfig(vocab_size=1)
    GPTConfig(vocab_size=1, block_size=1, n_layer=1, n_head=1, d_model=1)


# --- Budowa i atrybuty ---


def test_modules_are_nn_module():
    for module in (GPT(SMALL), CausalSelfAttention(SMALL), MLP(SMALL), Block(SMALL)):
        assert isinstance(module, nn.Module)


def test_attention_attributes():
    a = CausalSelfAttention(SMALL)
    assert a.n_head == 2
    assert a.head_dim == 16
    assert isinstance(a.qkv, nn.Linear)
    assert (a.qkv.in_features, a.qkv.out_features) == (32, 96)
    assert a.qkv.bias is not None
    assert isinstance(a.proj, nn.Linear)
    assert (a.proj.in_features, a.proj.out_features) == (32, 32)
    assert a.proj.bias is not None
    assert a.mask.dtype == torch.bool
    assert a.mask.shape == (16, 16)
    assert torch.equal(a.mask, torch.tril(torch.ones(16, 16, dtype=torch.bool)))
    assert CausalSelfAttention(GPTConfig()).head_dim == 32


def test_mlp_attributes():
    m = MLP(SMALL)
    assert isinstance(m.fc, nn.Linear)
    assert (m.fc.in_features, m.fc.out_features) == (32, 128)
    assert m.fc.bias is not None
    assert isinstance(m.proj, nn.Linear)
    assert (m.proj.in_features, m.proj.out_features) == (128, 32)
    assert m.proj.bias is not None
    assert isinstance(m.act, nn.GELU)
    assert m.act.approximate == "none"
    assert MLP(GPTConfig()).fc.out_features == 512


def test_block_attributes():
    b = Block(SMALL)
    assert isinstance(b.ln1, nn.LayerNorm)
    assert isinstance(b.ln2, nn.LayerNorm)
    assert tuple(b.ln1.normalized_shape) == (32,)
    assert tuple(b.ln2.normalized_shape) == (32,)
    assert isinstance(b.attn, CausalSelfAttention)
    assert isinstance(b.mlp, MLP)
    assert tuple(Block(GPTConfig()).ln1.normalized_shape) == (128,)


def test_gpt_attributes():
    m = GPT(SMALL)
    assert m.config == SMALL
    assert isinstance(m.tok_emb, nn.Embedding)
    assert list(m.tok_emb.weight.shape) == [64, 32]
    assert isinstance(m.pos_emb, nn.Embedding)
    assert list(m.pos_emb.weight.shape) == [16, 32]
    assert isinstance(m.blocks, nn.ModuleList)
    assert len(m.blocks) == 2
    assert all(isinstance(b, Block) for b in m.blocks)
    assert isinstance(m.ln_f, nn.LayerNorm)
    assert tuple(m.ln_f.normalized_shape) == (32,)
    assert isinstance(m.lm_head, nn.Linear)
    assert (m.lm_head.in_features, m.lm_head.out_features) == (32, 64)
    assert m.lm_head.bias is None


def test_weight_tying():
    for cfg in (SMALL, GPTConfig()):
        m = GPT(cfg)
        assert m.lm_head.weight is m.tok_emb.weight


def test_param_count():
    assert sum(p.numel() for p in GPT(SMALL).parameters()) == 28032
    assert sum(p.numel() for p in GPT(GPTConfig()).parameters()) == 957184


def test_state_dict_has_no_mask():
    for module in (GPT(SMALL), CausalSelfAttention(SMALL)):
        assert not any("mask" in key for key in module.state_dict())


def test_state_dict_save_load_same_logits(tmp_path):
    torch.manual_seed(0)
    m1 = GPT(SMALL)
    path = tmp_path / "m.pt"
    torch.save(m1.state_dict(), path)
    torch.manual_seed(1)
    m2 = GPT(SMALL)
    m2.load_state_dict(torch.load(path, weights_only=True))
    m1.eval()
    m2.eval()
    idx = _idx(2, 16)
    with torch.no_grad():
        assert _close(m1(idx), m2(idx))
    assert m2.lm_head.weight is m2.tok_emb.weight


# --- Inicjalizacja ---


def test_init_biases_zero_and_layernorm_default():
    torch.manual_seed(0)
    m = GPT(SMALL)
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and mod.bias is not None:
            assert torch.count_nonzero(mod.bias) == 0, name
        if isinstance(mod, nn.LayerNorm):
            assert torch.all(mod.weight == 1), name
            assert torch.count_nonzero(mod.bias) == 0, name


def test_init_std():
    torch.manual_seed(0)
    m = GPT(GPTConfig())
    proj_std = 0.02 / math.sqrt(8)
    checks = [
        (m.tok_emb.weight, 0.02),
        (m.pos_emb.weight, 0.02),
        (m.blocks[0].attn.qkv.weight, 0.02),
        (m.blocks[0].mlp.fc.weight, 0.02),
    ]
    for i in range(4):
        checks.append((m.blocks[i].attn.proj.weight, proj_std))
        checks.append((m.blocks[i].mlp.proj.weight, proj_std))
    for w, expected in checks:
        assert w.std().item() == pytest.approx(expected, rel=0.1)
        assert abs(w.mean().item()) < 0.002


# --- Klocki ---


def test_block_modules_shapes():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 32)
    for module in (CausalSelfAttention(SMALL), MLP(SMALL), Block(SMALL)):
        out = module(x)
        assert out.shape == (2, 5, 32)
        assert out.dtype == torch.float32
    big = GPTConfig()
    assert MLP(big)(torch.randn(2, 5, 128)).shape == (2, 5, 128)
    assert Block(big)(torch.randn(1, 3, 128)).shape == (1, 3, 128)


def test_attention_matches_reference():
    torch.manual_seed(0)
    a = CausalSelfAttention(SMALL)
    x = torch.randn(2, 7, 32)
    with torch.no_grad():
        q, k, v = a.qkv(x).split(32, dim=-1)
        q, k, v = (t.view(2, 7, 2, 16).transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        expected = a.proj(o.transpose(1, 2).reshape(2, 7, 32))
        assert _close(a(x), expected)


def test_attention_single_token_attends_to_itself():
    torch.manual_seed(0)
    a = CausalSelfAttention(SMALL)
    x = torch.randn(1, 1, 32)
    with torch.no_grad():
        out = a(x)
        v = a.qkv(x)[..., 64:96]
        assert out.shape == (1, 1, 32)
        assert _close(out, a.proj(v))


def test_attention_causal():
    torch.manual_seed(0)
    a = CausalSelfAttention(SMALL)
    x = torch.randn(1, 8, 32)
    x2 = x.clone()
    x2[0, 5] = torch.randn(32)
    with torch.no_grad():
        out, out2 = a(x), a(x2)
    assert _close(out[0, :5], out2[0, :5])
    assert not _close(out[0, 5:], out2[0, 5:])


def test_mlp_matches_formula():
    torch.manual_seed(0)
    m = MLP(SMALL)
    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        assert _close(m(x), m.proj(F.gelu(m.fc(x))))


def test_block_matches_prenorm_formula():
    torch.manual_seed(0)
    b = Block(SMALL)
    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        h = x + b.attn(b.ln1(x))
        expected = h + b.mlp(b.ln2(h))
        assert _close(b(x), expected)


def test_gpt_forward_matches_composition():
    torch.manual_seed(0)
    m = GPT(SMALL).eval()
    idx = _idx(2, 9)
    with torch.no_grad():
        h = m.tok_emb(idx) + m.pos_emb(torch.arange(9))
        for blk in m.blocks:
            h = blk(h)
        expected = m.lm_head(m.ln_f(h))
        assert _close(m(idx), expected)


# --- GPT.forward ---


@pytest.mark.parametrize("batch,length", [(1, 1), (2, 5), (3, 16)])
def test_forward_shapes_and_dtype(batch, length):
    torch.manual_seed(0)
    m = GPT(SMALL)
    logits = m(_idx(batch, length))
    assert logits.shape == (batch, length, 64)
    assert logits.dtype == torch.float32


def test_forward_rejects_too_long():
    m = GPT(SMALL)
    with pytest.raises(ValueError) as exc:
        m(torch.zeros(1, 17, dtype=torch.long))
    assert "17" in str(exc.value)
    assert "16" in str(exc.value)


def test_forward_rejects_empty_sequence():
    m = GPT(SMALL)
    with pytest.raises(ValueError):
        m(torch.zeros(1, 0, dtype=torch.long))


@pytest.mark.parametrize("shape", [(5,), (1, 2, 3)])
def test_forward_rejects_wrong_dims(shape):
    m = GPT(SMALL)
    with pytest.raises(ValueError):
        m(torch.zeros(*shape, dtype=torch.long))


def test_causal_mask_hides_future():
    torch.manual_seed(0)
    m = GPT(SMALL).eval()
    idx = torch.randint(0, 64, (1, 16))
    changed = idx.clone()
    changed[0, 10] = (changed[0, 10] + 1) % 64
    with torch.no_grad():
        a, b = m(idx), m(changed)
    assert _close(a[:, :10], b[:, :10])
    assert not _close(a[:, 10:], b[:, 10:])


@pytest.mark.parametrize("t", [0, 1, 15])
def test_causal_mask_each_position(t):
    torch.manual_seed(0)
    m = GPT(SMALL).eval()
    idx = torch.randint(0, 64, (1, 16))
    changed = idx.clone()
    changed[0, t] = (changed[0, t] + 1) % 64
    with torch.no_grad():
        a, b = m(idx), m(changed)
    if t > 0:
        assert _close(a[0, :t], b[0, :t])
    assert not _close(a[0, t], b[0, t])


@pytest.mark.parametrize("k", [1, 7, 16])
def test_prefix_consistency(k):
    torch.manual_seed(0)
    m = GPT(SMALL).eval()
    idx = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        assert _close(m(idx[:, :k]), m(idx)[:, :k])


def test_batch_items_independent():
    torch.manual_seed(0)
    m = GPT(SMALL).eval()
    idx = torch.randint(0, 64, (3, 16))
    changed = idx.clone()
    changed[1] = (idx[1] + torch.randint(1, 64, (16,))) % 64
    with torch.no_grad():
        a, b = m(idx), m(changed)
        assert _close(a[0], b[0])
        assert _close(a[2], b[2])
        assert _close(m(idx[0:1]), a[0:1])


def test_eval_and_train_same_logits():
    torch.manual_seed(0)
    m = GPT(SMALL)
    idx = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        m.train()
        a = m(idx)
        m.eval()
        b = m(idx)
    assert _close(a, b)


# --- Trening ---


def test_initial_loss_near_ln_vocab():
    torch.manual_seed(0)
    m = GPT(SMALL)
    idx = torch.randint(0, 64, (4, 16))
    targets = torch.randint(0, 64, (4, 16))
    loss = F.cross_entropy(m(idx).reshape(-1, 64), targets.reshape(-1)).item()
    assert 0.9 * math.log(64) <= loss <= 1.1 * math.log(64)


def test_all_params_get_gradient():
    torch.manual_seed(0)
    m = GPT(SMALL)
    m.train()
    idx = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))
    loss = F.cross_entropy(m(idx).reshape(-1, 64), targets.reshape(-1))
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert p.grad.abs().sum() > 0, name


def test_overfits_single_batch():
    torch.manual_seed(0)
    m = GPT(SMALL)
    m.train()
    d = torch.randint(0, 64, (4, 17), generator=torch.Generator().manual_seed(0))
    d[:, 0] = torch.arange(4)
    x, y = d[:, :-1], d[:, 1:]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    losses = []
    for _ in range(300):
        opt.zero_grad()
        loss = F.cross_entropy(m(x).reshape(-1, 64), y.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[0] > 3.0
    assert losses[-1] < 0.1


# --- Urządzenie ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="brak GPU")
def test_forward_on_cuda():
    torch.manual_seed(0)
    m = GPT(SMALL).to("cuda").eval()
    idx = torch.randint(0, 64, (2, 16)).to("cuda")
    with torch.no_grad():
        logits = m(idx)
        assert logits.device.type == "cuda"
        assert logits.shape == (2, 16, 64)
        assert logits.dtype == torch.float32
        cpu_logits = m.cpu()(idx.cpu())
    assert _close(logits.cpu(), cpu_logits, atol=1e-4)
