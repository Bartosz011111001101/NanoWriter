"""Testy pełnego treningu: TrainConfig, lr_at, make_optimizer, train_step, estimate_loss,
bits_per_char, checkpointy i train (wznowienie, log, walidacja)."""

import dataclasses
import json
import math

import pytest
import torch
from torch import nn

from nanowriter.dataset import get_batch
from nanowriter.model import GPT, GPTConfig
from nanowriter.train import (
    TrainConfig,
    bits_per_char,
    compute_loss,
    estimate_loss,
    load_checkpoint,
    load_gpt,
    lr_at,
    make_optimizer,
    save_checkpoint,
    train,
    train_step,
)

LN4 = math.log(4)
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="brak GPU")

DATA = torch.randint(0, 16, (400,), generator=torch.Generator().manual_seed(0))
VAL = torch.randint(0, 16, (200,), generator=torch.Generator().manual_seed(1))
CFG = TrainConfig(
    batch_size=4,
    block_size=8,
    max_steps=20,
    learning_rate=1e-2,
    warmup_steps=4,
    lr_schedule="cosine",
    min_lr=1e-3,
    grad_clip=1.0,
    weight_decay=0.1,
    eval_every=5,
    eval_batches=2,
)


def _gpt_m() -> GPT:
    return GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, d_model=32))


def _gpt_t(seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=16, block_size=8, n_layer=1, n_head=2, d_model=16))


def _e0() -> nn.Embedding:
    model = nn.Embedding(4, 4)
    with torch.no_grad():
        model.weight.zero_()
    return model


def _idx() -> torch.Tensor:
    return torch.randint(0, 16, (2, 8), generator=torch.Generator().manual_seed(7))


def _logits(model: nn.Module) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(_idx())


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _ck_state():
    """Model, optymalizator i generator po jednym kroku (niepusty stan optymalizatora)."""
    torch.manual_seed(0)
    model = _gpt_t(0)
    opt = make_optimizer(model, CFG)
    x, y = get_batch(DATA, 4, 8, _gen(0))
    train_step(model, opt, x, y)
    gen = _gen(3)
    torch.rand(5, generator=gen)
    return model, opt, gen


def _read_log(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


# ------------------------------------------------------------- TrainConfig


def test_train_config_all_defaults():
    c = TrainConfig()
    assert c.batch_size == 32
    assert c.block_size == 8
    assert c.max_steps == 5000
    assert c.learning_rate == 0.01
    assert c.seed == 0
    assert c.log_every == 500
    assert c.weight_decay == 0.01
    assert c.beta1 == 0.9
    assert c.beta2 == 0.999
    assert c.warmup_steps == 0
    assert c.lr_schedule == "constant"
    assert c.min_lr == 0.0
    assert c.grad_clip == 0.0
    assert c.bf16 is False
    assert c.eval_every == 0
    assert c.eval_batches == 20
    assert c.checkpoint_every == 0
    assert [f.name for f in dataclasses.fields(TrainConfig)] == [
        "batch_size",
        "block_size",
        "max_steps",
        "learning_rate",
        "seed",
        "log_every",
        "weight_decay",
        "beta1",
        "beta2",
        "warmup_steps",
        "lr_schedule",
        "min_lr",
        "grad_clip",
        "bf16",
        "eval_every",
        "eval_batches",
        "checkpoint_every",
    ]


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"weight_decay": -0.1}, "weight_decay"),
        ({"beta1": 1.0}, "beta1"),
        ({"beta1": -0.1}, "beta1"),
        ({"beta2": 1.0}, "beta2"),
        ({"beta2": -0.01}, "beta2"),
        ({"warmup_steps": -1}, "warmup_steps"),
        ({"lr_schedule": "linear"}, "lr_schedule"),
        ({"min_lr": -0.1}, "min_lr"),
        ({"min_lr": 0.1, "learning_rate": 0.01}, "min_lr"),
        ({"grad_clip": -1.0}, "grad_clip"),
        ({"eval_every": -1}, "eval_every"),
        ({"eval_batches": 0}, "eval_batches"),
        ({"checkpoint_every": -1}, "checkpoint_every"),
    ],
)
def test_train_config_new_fields_invalid(kwargs, name):
    with pytest.raises(ValueError, match=name):
        TrainConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_steps": 0},
        {"weight_decay": 0.0},
        {"beta1": 0.0, "beta2": 0.0},
        {"warmup_steps": 0},
        {"lr_schedule": "cosine"},
        {"min_lr": 0.0},
        {"min_lr": 0.01, "learning_rate": 0.01},
        {"grad_clip": 0.0},
        {"eval_every": 0},
        {"eval_batches": 1},
        {"checkpoint_every": 0},
        {"bf16": True},
    ],
)
def test_train_config_new_fields_boundaries_valid(kwargs):
    assert isinstance(TrainConfig(**kwargs), TrainConfig)


def test_train_config_still_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        TrainConfig().warmup_steps = 5  # type: ignore[misc]


# ------------------------------------------------------------------- lr_at


def test_lr_at_docstring_cosine_example():
    config = TrainConfig(
        learning_rate=1.0, warmup_steps=2, max_steps=6, lr_schedule="cosine", min_lr=0.0
    )
    expected = [0.5, 1.0, 1.0, 0.853553, 0.5, 0.146447]
    for step, value in enumerate(expected):
        assert lr_at(step, config) == pytest.approx(value, abs=1e-6)


@pytest.mark.parametrize("step", [0, 1, 100, 4999, 5000, 10000])
def test_lr_at_constant(step):
    assert lr_at(step, TrainConfig(learning_rate=0.01)) == pytest.approx(0.01)


def test_lr_at_warmup_then_constant():
    config = TrainConfig(learning_rate=0.1, warmup_steps=4)
    expected = [0.025, 0.05, 0.075, 0.1, 0.1, 0.1]
    for step, value in enumerate(expected):
        assert lr_at(step, config) == pytest.approx(value, abs=1e-9)


def test_lr_at_cosine_without_warmup():
    config = TrainConfig(
        learning_rate=1.0, warmup_steps=0, max_steps=4, lr_schedule="cosine", min_lr=0.0
    )
    expected = [1.0, 0.853553, 0.5, 0.146447]
    for step, value in enumerate(expected):
        assert lr_at(step, config) == pytest.approx(value, abs=1e-6)


def test_lr_at_cosine_reaches_min_lr_after_max_steps():
    c = TrainConfig(
        learning_rate=1.0, warmup_steps=2, max_steps=6, lr_schedule="cosine", min_lr=0.1
    )
    assert lr_at(6, c) == pytest.approx(0.1, abs=1e-9)
    assert lr_at(100, c) == pytest.approx(0.1, abs=1e-9)
    assert lr_at(2, c) == pytest.approx(1.0, abs=1e-9)


def test_lr_at_warmup_equals_max_steps():
    c = TrainConfig(
        learning_rate=1.0, warmup_steps=3, max_steps=3, lr_schedule="cosine", min_lr=0.1
    )
    assert lr_at(0, c) == pytest.approx(1 / 3, abs=1e-9)
    assert lr_at(1, c) == pytest.approx(2 / 3, abs=1e-9)
    assert lr_at(2, c) == pytest.approx(1.0, abs=1e-9)
    assert lr_at(3, c) == pytest.approx(1.0, abs=1e-9)
    assert lr_at(4, c) == pytest.approx(0.1, abs=1e-9)


def test_lr_at_negative_step():
    with pytest.raises(ValueError):
        lr_at(-1, TrainConfig())


def test_lr_at_returns_float():
    assert isinstance(lr_at(0, TrainConfig()), float)
    assert isinstance(lr_at(3, CFG), float)


# ---------------------------------------------------------- make_optimizer


def test_make_optimizer_gpt_groups():
    model = _gpt_m()
    opt = make_optimizer(model, TrainConfig(weight_decay=0.1))
    assert isinstance(opt, torch.optim.AdamW)
    assert len(opt.param_groups) == 2
    g0, g1 = opt.param_groups
    assert len(g0["params"]) == 10
    assert g0["weight_decay"] == 0.1
    assert all(p.dim() >= 2 for p in g0["params"])
    assert len(g1["params"]) == 18
    assert g1["weight_decay"] == 0.0
    assert all(p.dim() < 2 for p in g1["params"])
    for group in opt.param_groups:
        assert group["lr"] == 0.01
        assert tuple(group["betas"]) == (0.9, 0.999)


def test_make_optimizer_custom_lr_and_betas():
    opt = make_optimizer(
        _gpt_m(),
        TrainConfig(learning_rate=0.003, beta1=0.8, beta2=0.95, weight_decay=0.1),
    )
    for group in opt.param_groups:
        assert group["lr"] == 0.003
        assert tuple(group["betas"]) == (0.8, 0.95)


def test_make_optimizer_each_param_once():
    model = _gpt_m()
    opt = make_optimizer(model, TrainConfig(weight_decay=0.1))
    ids = [id(p) for group in opt.param_groups for p in group["params"]]
    assert len(ids) == 28
    assert len(set(ids)) == 28
    assert set(ids) == {id(p) for p in model.parameters()}
    assert ids.count(id(model.tok_emb.weight)) == 1


def test_make_optimizer_embedding_single_group():
    model = nn.Embedding(4, 4)
    opt = make_optimizer(model, TrainConfig())
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["weight_decay"] == 0.01
    params = opt.param_groups[0]["params"]
    assert len(params) == 1
    assert params[0] is model.weight


def test_make_optimizer_only_1d_params():
    opt = make_optimizer(nn.LayerNorm(4), TrainConfig(weight_decay=0.1))
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["weight_decay"] == 0.0
    assert len(opt.param_groups[0]["params"]) == 2


def test_make_optimizer_skips_frozen():
    model = _gpt_m()
    model.pos_emb.weight.requires_grad_(False)
    opt = make_optimizer(model, TrainConfig(weight_decay=0.1))
    assert len(opt.param_groups[0]["params"]) == 9
    assert len(opt.param_groups[1]["params"]) == 18
    ids = {id(p) for group in opt.param_groups for p in group["params"]}
    assert id(model.pos_emb.weight) not in ids


def test_make_optimizer_matches_stage1_adamw():
    torch.manual_seed(0)
    a = nn.Embedding(8, 8)
    torch.manual_seed(0)
    b = nn.Embedding(8, 8)
    x = torch.tensor([[0, 1, 2, 3]])
    y = torch.tensor([[1, 2, 3, 4]])
    opt_a = make_optimizer(a, TrainConfig())
    opt_b = torch.optim.AdamW(b.parameters(), lr=0.01)
    losses_a = [train_step(a, opt_a, x, y) for _ in range(3)]
    losses_b = [train_step(b, opt_b, x, y) for _ in range(3)]
    for la, lb in zip(losses_a, losses_b, strict=True):
        assert la == pytest.approx(lb, abs=1e-7)
    assert torch.allclose(a.weight, b.weight, atol=1e-7)


# -------------------------------------------------------------- train_step


def test_train_step_docstring_example():
    model = _e0()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.tensor([[0, 1]])
    y = torch.tensor([[1, 2]])
    result = train_step(model, optimizer, x, y)
    assert isinstance(result, float)
    assert result == pytest.approx(LN4, abs=1e-6)
    assert not torch.all(model.weight == 0)


def test_train_step_grad_clip_limits_update():
    model = _e0()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    x = torch.tensor([[0, 1, 2, 3]])
    y = torch.tensor([[1, 2, 3, 0]])
    result = train_step(model, optimizer, x, y, grad_clip=0.01)
    assert result == pytest.approx(LN4, abs=1e-6)
    norm = model.weight.norm().item()
    assert norm <= 0.01 + 1e-6
    assert norm > 0.005


def test_train_step_without_clip_update_is_larger():
    x = torch.tensor([[0, 1, 2, 3]])
    y = torch.tensor([[1, 2, 3, 0]])
    model = _e0()
    train_step(model, torch.optim.SGD(model.parameters(), lr=1.0), x, y, grad_clip=0.0)
    assert model.weight.norm().item() == pytest.approx(0.4330127, abs=1e-4)
    model2 = _e0()
    train_step(model2, torch.optim.SGD(model2.parameters(), lr=1.0), x, y)
    assert model2.weight.norm().item() == pytest.approx(0.4330127, abs=1e-4)


def test_train_step_bf16_on_cpu():
    torch.manual_seed(0)
    model = _gpt_t(0)
    opt = make_optimizer(model, TrainConfig())
    x, y = get_batch(DATA, 2, 8, _gen(0))
    result = train_step(model, opt, x, y, bf16=True)
    assert isinstance(result, float)
    assert math.isfinite(result)
    assert 0.5 * math.log(16) < result < 1.5 * math.log(16)


def test_train_step_keyword_only_args():
    model = _e0()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.tensor([[0, 1]])
    y = torch.tensor([[1, 2]])
    with pytest.raises(TypeError):
        train_step(model, optimizer, x, y, 1.0)  # type: ignore[misc]


# ----------------------------------------------------------- estimate_loss


def _manual_window_losses(model, data, block_size, n_windows):
    losses = []
    with torch.no_grad():
        for i in range(n_windows):
            x = data[block_size * i : block_size * i + block_size][None]
            y = data[block_size * i + 1 : block_size * i + block_size + 1][None]
            losses.append(compute_loss(model(x), y).item())
    return losses


def _emb_and_data():
    torch.manual_seed(0)
    model = nn.Embedding(5, 5)
    data = torch.randint(0, 5, (23,), generator=_gen(0))
    return model, data


def test_estimate_loss_docstring_example():
    result = estimate_loss(_e0(), torch.arange(9) % 4, 2, 4, "cpu")
    assert isinstance(result, float)
    assert result == pytest.approx(LN4, abs=1e-6)


def test_estimate_loss_matches_manual_windows():
    model, data = _emb_and_data()
    losses = _manual_window_losses(model, data, 4, 5)
    expected = sum(losses) / 5
    assert estimate_loss(model, data, 2, 4, "cpu") == pytest.approx(expected, abs=1e-6)


def test_estimate_loss_max_batches():
    model, data = _emb_and_data()
    losses = _manual_window_losses(model, data, 4, 5)
    assert estimate_loss(model, data, 2, 4, "cpu", max_batches=1) == pytest.approx(
        sum(losses[:2]) / 2, abs=1e-6
    )
    assert estimate_loss(model, data, 2, 4, "cpu", max_batches=2) == pytest.approx(
        sum(losses[:4]) / 4, abs=1e-6
    )
    assert estimate_loss(model, data, 2, 4, "cpu", max_batches=10) == pytest.approx(
        sum(losses) / 5, abs=1e-6
    )


def test_estimate_loss_single_window():
    model, _ = _emb_and_data()
    data = torch.tensor([0, 1, 2, 3, 4])
    with torch.no_grad():
        expected = compute_loss(model(data[:4][None]), data[1:5][None]).item()
    assert estimate_loss(model, data, 3, 4, "cpu") == pytest.approx(expected, abs=1e-6)


def test_estimate_loss_too_short():
    with pytest.raises(ValueError):
        estimate_loss(_e0(), torch.arange(4), 2, 4, "cpu")


@pytest.mark.parametrize(
    ("data", "batch_size", "block_size", "max_batches"),
    [
        (torch.zeros((2, 10), dtype=torch.long), 2, 4, None),
        (torch.arange(20) % 4, 0, 4, None),
        (torch.arange(20) % 4, 2, 0, None),
        (torch.arange(20) % 4, 2, 4, 0),
    ],
)
def test_estimate_loss_invalid_args(data, batch_size, block_size, max_batches):
    with pytest.raises(ValueError):
        estimate_loss(_e0(), data, batch_size, block_size, "cpu", max_batches=max_batches)


def test_estimate_loss_deterministic_and_no_rng():
    model = _gpt_t(0)
    first = estimate_loss(model, VAL, 4, 8, "cpu")
    second = estimate_loss(model, VAL, 4, 8, "cpu")
    assert first == second
    torch.manual_seed(5)
    estimate_loss(model, VAL, 4, 8, "cpu")
    a = torch.rand(3)
    torch.manual_seed(5)
    b = torch.rand(3)
    assert torch.equal(a, b)


def test_estimate_loss_restores_mode():
    model = _gpt_t(0)
    model.train()
    estimate_loss(model, VAL, 4, 8, "cpu")
    assert model.training is True
    model.eval()
    estimate_loss(model, VAL, 4, 8, "cpu")
    assert model.training is False


def test_estimate_loss_no_gradients():
    model = _gpt_t(0)
    assert all(p.grad is None for p in model.parameters())
    estimate_loss(model, VAL, 4, 8, "cpu")
    assert all(p.grad is None for p in model.parameters())


# ------------------------------------------------------------ bits_per_char


def test_bits_per_char_examples():
    first = bits_per_char(2.0, 100, 250)
    assert isinstance(first, float)
    assert first == pytest.approx(1.154156, abs=1e-6)
    assert bits_per_char(math.log(2), 3, 3) == pytest.approx(1.0, abs=1e-12)
    assert bits_per_char(1.5, 0, 10) == 0.0


@pytest.mark.parametrize(
    ("loss", "n_tokens", "n_chars"),
    [(1.0, 10, 0), (1.0, 10, -1), (1.0, -1, 10)],
)
def test_bits_per_char_invalid(loss, n_tokens, n_chars):
    with pytest.raises(ValueError):
        bits_per_char(loss, n_tokens, n_chars)


# -------------------------------------------------------------- Checkpoint


def test_checkpoint_keys_and_values(tmp_path):
    model, opt, gen = _ck_state()
    path = tmp_path / "c.pt"
    save_checkpoint(path, model, opt, 10, gen, CFG)
    ck = load_checkpoint(path)
    assert set(ck) == {"model", "optimizer", "step", "generator", "config", "model_config"}
    assert ck["step"] == 10
    assert type(ck["step"]) is int
    assert ck["config"] == dataclasses.asdict(CFG)
    assert ck["model_config"] == dataclasses.asdict(model.config)
    assert torch.equal(ck["generator"], gen.get_state())
    state = model.state_dict()
    assert set(ck["model"]) == set(state)
    for key, tensor in ck["model"].items():
        assert tensor.device.type == "cpu"
        assert torch.equal(tensor, state[key].cpu())


def test_checkpoint_creates_parent_dirs_and_no_tmp(tmp_path):
    model, opt, gen = _ck_state()
    path = tmp_path / "a" / "b" / "c.pt"
    save_checkpoint(path, model, opt, 1, gen, CFG)
    assert path.exists()
    assert [f.name for f in path.parent.iterdir()] == ["c.pt"]


def test_checkpoint_overwrites(tmp_path):
    model, opt, gen = _ck_state()
    path = tmp_path / "c.pt"
    save_checkpoint(path, model, opt, 10, gen, CFG)
    save_checkpoint(path, model, opt, 20, gen, CFG)
    assert load_checkpoint(path)["step"] == 20
    assert [f.name for f in tmp_path.iterdir()] == ["c.pt"]


def test_checkpoint_model_config_none_for_embedding(tmp_path):
    model = nn.Embedding(4, 4)
    opt = make_optimizer(model, TrainConfig())
    path = tmp_path / "c.pt"
    save_checkpoint(path, model, opt, 0, torch.Generator(), TrainConfig())
    assert load_checkpoint(path)["model_config"] is None


def test_checkpoint_optimizer_state_restorable(tmp_path):
    model, opt, gen = _ck_state()
    path = tmp_path / "c.pt"
    save_checkpoint(path, model, opt, 1, gen, CFG)
    ck = load_checkpoint(path)
    model2 = _gpt_t(1)
    opt2 = make_optimizer(model2, CFG)
    model2.load_state_dict(ck["model"])
    opt2.load_state_dict(ck["optimizer"])
    for p, p2 in zip(model.parameters(), model2.parameters(), strict=True):
        assert torch.equal(opt2.state[p2]["exp_avg"], opt.state[p]["exp_avg"])
        assert torch.equal(opt2.state[p2]["exp_avg_sq"], opt.state[p]["exp_avg_sq"])
    groups = opt.state_dict()["param_groups"]
    groups2 = opt2.state_dict()["param_groups"]
    assert len(groups) == len(groups2)
    for g, g2 in zip(groups, groups2, strict=True):
        assert g2["lr"] == g["lr"]
        assert g2["weight_decay"] == g["weight_decay"]
        assert tuple(g2["betas"]) == tuple(g["betas"])


def test_load_checkpoint_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "brak.pt")


def test_load_gpt_same_logits_and_eval(tmp_path):
    model, opt, gen = _ck_state()
    path = tmp_path / "c.pt"
    save_checkpoint(path, model, opt, 1, gen, CFG)
    g = load_gpt(path, "cpu")
    assert isinstance(g, GPT)
    assert g.training is False
    assert g.config == model.config
    assert all(p.device.type == "cpu" for p in g.parameters())
    idx = _idx()
    model.eval()
    with torch.no_grad():
        assert torch.allclose(g(idx), model(idx), atol=1e-6)


def test_load_gpt_rejects_non_gpt_checkpoint(tmp_path):
    model = nn.Embedding(4, 4)
    opt = make_optimizer(model, TrainConfig())
    path = tmp_path / "c.pt"
    save_checkpoint(path, model, opt, 0, torch.Generator(), TrainConfig())
    with pytest.raises(ValueError):
        load_gpt(path, "cpu")


@needs_cuda
def test_checkpoint_from_cuda_loads_on_cpu(tmp_path):
    model = _gpt_t(0).to("cuda")
    opt = make_optimizer(model, CFG)
    x, y = get_batch(DATA, 4, 8, _gen(0))
    train_step(model, opt, x.to("cuda"), y.to("cuda"))
    path = tmp_path / "c.pt"
    save_checkpoint(path, model, opt, 1, torch.Generator(), CFG)
    ck = load_checkpoint(path)
    assert all(t.device.type == "cpu" for t in ck["model"].values())
    assert ck["generator"].device.type == "cpu"
    g = load_gpt(path, "cuda")
    assert all(p.device.type == "cuda" for p in g.parameters())
    assert g.training is False


# ------------------------------------------------------------------- train


def _run_resume(tmp_path, k, device="cpu"):
    """Zwraca (a, b1, b2, ev_a, ev_b, mA, m2, ścieżka checkpointu)."""
    ev_a: list[tuple[int, float]] = []
    m_a = _gpt_t(0).to(device)
    a = train(
        m_a, DATA, CFG, device, val_data=VAL, on_eval=lambda s, v: ev_a.append((s, v))
    )
    ev_b: list[tuple[int, float]] = []
    path = tmp_path / "ck.pt"
    m1 = _gpt_t(0).to(device)
    b1 = train(
        m1,
        DATA,
        CFG,
        device,
        val_data=VAL,
        on_eval=lambda s, v: ev_b.append((s, v)),
        checkpoint_path=path,
        until_step=k,
    )
    m2 = _gpt_t(1).to(device)
    b2 = train(
        m2,
        DATA,
        CFG,
        device,
        val_data=VAL,
        on_eval=lambda s, v: ev_b.append((s, v)),
        checkpoint_path=path,
        resume_from=path,
    )
    return a, b1, b2, ev_a, ev_b, m_a, m2, path


@pytest.mark.parametrize("k", [1, 7, 10, 19])
def test_train_resume_equals_uninterrupted(tmp_path, k):
    a, b1, b2, ev_a, ev_b, m_a, m2, path = _run_resume(tmp_path, k)
    assert len(a) == 20
    assert len(b1) == k
    assert len(b2) == 20 - k
    assert a == b1 + b2
    assert [s for s, _ in ev_a] == [4, 9, 14, 19]
    assert ev_a == ev_b
    assert torch.allclose(_logits(m_a), _logits(m2), atol=1e-6)
    assert load_checkpoint(path)["step"] == 20


def test_train_resume_log_equals_uninterrupted(tmp_path):
    log_a = tmp_path / "a" / "log.jsonl"
    log_b = tmp_path / "b" / "log.jsonl"
    path = tmp_path / "ck.pt"
    a = train(_gpt_t(0), DATA, CFG, "cpu", val_data=VAL, log_path=log_a)
    train(
        _gpt_t(0),
        DATA,
        CFG,
        "cpu",
        val_data=VAL,
        checkpoint_path=path,
        log_path=log_b,
        until_step=10,
    )
    train(
        _gpt_t(1),
        DATA,
        CFG,
        "cpu",
        val_data=VAL,
        checkpoint_path=path,
        resume_from=path,
        log_path=log_b,
    )
    lines_a = _read_log(log_a)
    lines_b = _read_log(log_b)
    assert len(lines_a) == 6
    assert len(lines_b) == 6
    assert lines_a == lines_b
    kinds = [(r["step"], "train_loss" in r) for r in lines_a]
    assert kinds == [(0, True), (4, False), (9, False), (14, False), (19, True), (19, False)]
    assert lines_a[0]["train_loss"] == a[0]
    assert lines_a[4]["train_loss"] == a[19]
    for record in lines_a:
        assert set(record) in ({"step", "train_loss", "lr"}, {"step", "val_loss"})


def test_train_validation_does_not_change_losses():
    plain = train(_gpt_t(0), DATA, CFG, "cpu")
    events: list[tuple[int, float]] = []
    with_val = train(
        _gpt_t(0),
        DATA,
        CFG,
        "cpu",
        val_data=VAL,
        on_eval=lambda s, v: events.append((s, v)),
    )
    assert plain == with_val
    assert events


def test_train_on_eval_steps():
    cfg = TrainConfig(batch_size=4, block_size=8, max_steps=12, eval_every=5, eval_batches=2)
    events: list[tuple[int, float]] = []
    train(
        _gpt_t(0),
        DATA,
        cfg,
        "cpu",
        val_data=VAL,
        on_eval=lambda s, v: events.append((s, v)),
    )
    assert [s for s, _ in events] == [4, 9, 11]
    for _, value in events:
        assert isinstance(value, float)
        assert math.isfinite(value)


def test_train_on_eval_value_is_estimate_loss():
    cfg = TrainConfig(batch_size=4, block_size=8, max_steps=10, eval_every=5, eval_batches=2)
    events: list[tuple[int, float]] = []
    model = _gpt_t(0)
    train(model, DATA, cfg, "cpu", val_data=VAL, on_eval=lambda s, v: events.append((s, v)))
    step, value = events[-1]
    assert step == 9
    expected = estimate_loss(model, VAL, 4, 8, "cpu", max_batches=2)
    assert value == pytest.approx(expected, abs=1e-6)


def test_train_no_eval_without_val_data_or_eval_every():
    events: list[tuple[int, float]] = []

    def on_eval(step: int, value: float) -> None:
        events.append((step, value))

    train(_gpt_t(0), DATA, CFG, "cpu", on_eval=on_eval)
    assert events == []
    train(
        _gpt_t(0),
        DATA,
        dataclasses.replace(CFG, eval_every=0),
        "cpu",
        val_data=VAL,
        on_eval=on_eval,
    )
    assert events == []


def test_train_mode_after_validation():
    model = _gpt_t(0)
    model.eval()
    train(model, DATA, CFG, "cpu", val_data=VAL)
    assert model.training is True


def test_train_log_content_and_lr(tmp_path):
    cfg = dataclasses.replace(CFG, max_steps=6, log_every=2, eval_every=3)
    log = tmp_path / "log.jsonl"
    losses = train(_gpt_t(0), DATA, cfg, "cpu", val_data=VAL, log_path=log)
    records = _read_log(log)
    assert len(records) == 6
    kinds = [(r["step"], "train_loss" in r) for r in records]
    assert kinds == [(0, True), (2, True), (2, False), (4, True), (5, True), (5, False)]
    for record in records:
        if "train_loss" in record:
            assert set(record) == {"step", "train_loss", "lr"}
            assert record["train_loss"] == losses[record["step"]]
            assert record["lr"] == pytest.approx(lr_at(record["step"], cfg), abs=1e-12)
        else:
            assert set(record) == {"step", "val_loss"}
            assert isinstance(record["val_loss"], float)
            assert math.isfinite(record["val_loss"])


def test_train_log_overwritten_without_resume(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text("stare\n", encoding="utf-8")
    cfg = dataclasses.replace(CFG, max_steps=6, log_every=2)
    train(_gpt_t(0), DATA, cfg, "cpu", log_path=log)
    text = log.read_text(encoding="utf-8")
    assert "stare" not in text
    records = _read_log(log)
    assert [r["step"] for r in records] == [0, 2, 4, 5]


def test_train_log_creates_parent_dirs(tmp_path):
    log = tmp_path / "x" / "y" / "log.jsonl"
    cfg = dataclasses.replace(CFG, max_steps=3, log_every=2)
    train(_gpt_t(0), DATA, cfg, "cpu", log_path=log)
    assert log.exists()
    assert len(_read_log(log)) == 2


def test_train_log_lines_written_per_event(tmp_path):
    log = tmp_path / "log.jsonl"
    cfg = dataclasses.replace(CFG, max_steps=6, log_every=2)
    seen: list[int] = []

    def on_log(step: int, loss: float) -> None:
        if log.exists():
            seen.append(len(log.read_text(encoding="utf-8").splitlines()))
        else:
            seen.append(0)

    train(_gpt_t(0), DATA, cfg, "cpu", on_log=on_log, log_path=log)
    assert len(seen) == 4
    for index, count in enumerate(seen):
        assert index <= count <= index + 1


def test_train_checkpoint_every(tmp_path):
    path = tmp_path / "ck.pt"
    cfg = TrainConfig(batch_size=4, block_size=8, max_steps=10, log_every=1, checkpoint_every=4)
    seen: list[int | None] = []

    def on_log(step: int, loss: float) -> None:
        seen.append(load_checkpoint(path)["step"] if path.exists() else None)

    train(_gpt_t(0), DATA, cfg, "cpu", on_log=on_log, checkpoint_path=path)
    assert seen == [None, None, None, None, 4, 4, 4, 4, 8, 8]
    assert load_checkpoint(path)["step"] == 10


def test_train_checkpoint_at_end_of_call(tmp_path):
    path = tmp_path / "ck.pt"
    model = _gpt_t(0)
    losses = train(model, DATA, CFG, "cpu", checkpoint_path=path, until_step=7)
    assert len(losses) == 7
    ck = load_checkpoint(path)
    assert ck["step"] == 7
    assert ck["model_config"] == dataclasses.asdict(model.config)


def test_train_no_checkpoint_path_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train(_gpt_t(0), DATA, dataclasses.replace(CFG, checkpoint_every=2), "cpu")
    assert list(tmp_path.iterdir()) == []


def test_train_zero_steps_no_checkpoint(tmp_path):
    path = tmp_path / "ck.pt"
    cfg = TrainConfig(batch_size=4, block_size=8, max_steps=0)
    calls: list[int] = []
    result = train(
        _gpt_t(0),
        DATA,
        cfg,
        "cpu",
        on_log=lambda s, v: calls.append(s),
        checkpoint_path=path,
    )
    assert result == []
    assert calls == []
    assert not path.exists()


def test_train_until_step_zero(tmp_path):
    path = tmp_path / "ck.pt"
    result = train(_gpt_t(0), DATA, CFG, "cpu", checkpoint_path=path, until_step=0)
    assert result == []
    assert not path.exists()


def test_train_until_step_beyond_max_steps():
    a = train(_gpt_t(0), DATA, CFG, "cpu", until_step=100)
    b = train(_gpt_t(0), DATA, CFG, "cpu")
    assert a == b
    assert len(a) == 20


def test_train_until_step_negative():
    with pytest.raises(ValueError):
        train(_gpt_t(0), DATA, CFG, "cpu", until_step=-1)


def test_train_resume_after_last_step(tmp_path):
    path = tmp_path / "ck.pt"
    train(_gpt_t(0), DATA, CFG, "cpu", checkpoint_path=path)
    calls: list[int] = []
    result = train(
        _gpt_t(1),
        DATA,
        CFG,
        "cpu",
        resume_from=path,
        checkpoint_path=path,
        on_log=lambda s, v: calls.append(s),
    )
    assert result == []
    assert calls == []
    assert load_checkpoint(path)["step"] == 20


def test_train_resume_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        train(_gpt_t(0), DATA, CFG, "cpu", resume_from=tmp_path / "brak.pt")


def test_train_on_log_steps_with_resume(tmp_path):
    path = tmp_path / "ck.pt"
    cfg = dataclasses.replace(CFG, max_steps=10, log_every=3)
    first: list[int] = []
    second: list[int] = []
    train(
        _gpt_t(0),
        DATA,
        cfg,
        "cpu",
        on_log=lambda s, v: first.append(s),
        checkpoint_path=path,
        until_step=5,
    )
    train(
        _gpt_t(1),
        DATA,
        cfg,
        "cpu",
        on_log=lambda s, v: second.append(s),
        resume_from=path,
    )
    assert first == [0, 3]
    assert second == [6, 9]


def test_train_matches_stage1_manual_loop():
    cfg = TrainConfig(
        batch_size=4, block_size=8, max_steps=15, learning_rate=1e-2, seed=3, log_every=5
    )
    data = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7] * 30)
    torch.manual_seed(0)
    model_a = nn.Embedding(8, 8)
    losses_a = train(model_a, data, cfg, "cpu")
    torch.manual_seed(0)
    model_b = nn.Embedding(8, 8)
    opt = torch.optim.AdamW(model_b.parameters(), lr=1e-2)
    gen = torch.Generator().manual_seed(3)
    losses_b = []
    for _ in range(15):
        x, y = get_batch(data, 4, 8, gen, "cpu")
        losses_b.append(train_step(model_b, opt, x, y))
    assert len(losses_a) == len(losses_b) == 15
    for la, lb in zip(losses_a, losses_b, strict=True):
        assert la == pytest.approx(lb, abs=1e-6)
    assert torch.allclose(model_a.weight, model_b.weight, atol=1e-6)


def test_train_returns_floats_and_reproducible():
    first = train(_gpt_t(0), DATA, CFG, "cpu")
    second = train(_gpt_t(0), DATA, CFG, "cpu")
    assert first == second
    assert len(first) == 20
    for value in first:
        assert isinstance(value, float)
        assert math.isfinite(value)


def test_train_prints_nothing(tmp_path, capsys):
    path = tmp_path / "ck.pt"
    train(
        _gpt_t(0),
        DATA,
        CFG,
        "cpu",
        val_data=VAL,
        checkpoint_path=path,
        log_path=tmp_path / "log.jsonl",
    )
    load_gpt(path, "cpu")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@needs_cuda
def test_train_bf16_cuda(tmp_path):
    path = tmp_path / "ck.pt"
    model = _gpt_t(0).to("cuda")
    cfg = dataclasses.replace(CFG, max_steps=5, bf16=True)
    losses = train(model, DATA, cfg, "cuda", val_data=VAL, checkpoint_path=path)
    assert len(losses) == 5
    for value in losses:
        assert isinstance(value, float)
        assert math.isfinite(value)
    assert load_checkpoint(path)["generator"].device.type == "cpu"


@needs_cuda
def test_train_resume_cuda(tmp_path):
    a, b1, b2, _, _, _, _, _ = _run_resume(tmp_path, 10, device="cuda")
    assert len(b1) == 10
    assert len(b2) == 10
    assert all(math.isfinite(v) for v in b1 + b2)
    for expected, actual in zip(a, b1 + b2, strict=True):
        assert actual == pytest.approx(expected, abs=1e-4)
