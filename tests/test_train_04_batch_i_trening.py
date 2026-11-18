"""Testy get_batch, TrainConfig, compute_loss, train_step i train."""

import dataclasses
import math

import pytest
import torch
from torch import nn

from nanowriter.dataset import get_batch
from nanowriter.train import TrainConfig, compute_loss, train, train_step

LN4 = 1.3862944
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="brak GPU")


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _zero_embedding() -> nn.Embedding:
    model = nn.Embedding(4, 4)
    with torch.no_grad():
        model.weight.zero_()
    return model


def _cyclic_data() -> torch.Tensor:
    return torch.tensor([0, 1, 2, 3] * 50)


# ---------------------------------------------------------------- get_batch


def test_get_batch_docstring_example():
    x, y = get_batch(torch.arange(10), 2, 4, _gen(0))
    assert x.shape == (2, 4)
    assert y.shape == (2, 4)
    assert torch.equal(y, x + 1)
    for row in x:
        s = int(row[0])
        assert 0 <= s <= 5
        assert row.tolist() == [s, s + 1, s + 2, s + 3]


def test_get_batch_single_start():
    x, y = get_batch(torch.arange(5), 3, 4, _gen(0))
    assert torch.equal(x, torch.tensor([[0, 1, 2, 3]] * 3))
    assert torch.equal(y, torch.tensor([[1, 2, 3, 4]] * 3))


def test_get_batch_b1_t1():
    x, y = get_batch(torch.arange(2), 1, 1, _gen(0))
    assert x.shape == (1, 1)
    assert y.shape == (1, 1)
    assert torch.equal(x, torch.tensor([[0]]))
    assert torch.equal(y, torch.tensor([[1]]))


def test_get_batch_too_short():
    g = _gen(0)
    with pytest.raises(ValueError):
        get_batch(torch.arange(4), 1, 4, g)
    with pytest.raises(ValueError) as info:
        get_batch(torch.arange(3), 1, 5, g)
    assert "3" in str(info.value)
    assert "5" in str(info.value)


@pytest.mark.parametrize(
    ("data", "batch_size", "block_size"),
    [
        (torch.zeros(2, 10, dtype=torch.long), 1, 4),
        (torch.arange(10), 0, 4),
        (torch.arange(10), 1, 0),
    ],
    ids=["data_2d", "batch_size_zero", "block_size_zero"],
)
def test_get_batch_invalid_args(data, batch_size, block_size):
    g = _gen(0)
    with pytest.raises(ValueError):
        get_batch(data, batch_size, block_size, g)


def test_get_batch_shift():
    data = torch.randint(0, 100, (50,), generator=_gen(1))
    x, y = get_batch(data, 4, 8, _gen(0))
    assert torch.equal(y[:, :-1], x[:, 1:])
    for b in range(4):
        found = any(
            torch.equal(x[b], data[s : s + 8]) and torch.equal(y[b], data[s + 1 : s + 9])
            for s in range(42)
        )
        assert found


@pytest.mark.parametrize("dt", [torch.int64, torch.int32, torch.uint16])
def test_get_batch_dtype(dt):
    data = torch.arange(20).to(dt)
    x, y = get_batch(data, 3, 4, _gen(0))
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
    assert x.shape == (3, 4)
    assert y.shape == (3, 4)
    assert torch.equal(y, x + 1)


def test_get_batch_default_device_cpu():
    x, y = get_batch(torch.arange(10), 2, 4, _gen(0))
    assert x.device.type == "cpu"
    assert y.device.type == "cpu"


def test_get_batch_same_seed_same_batch():
    x1, y1 = get_batch(torch.arange(1000), 8, 4, _gen(0))
    x2, y2 = get_batch(torch.arange(1000), 8, 4, _gen(0))
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_get_batch_different_seeds_differ():
    x1, _ = get_batch(torch.arange(1000), 8, 4, _gen(0))
    x2, _ = get_batch(torch.arange(1000), 8, 4, _gen(1))
    assert not torch.equal(x1, x2)


def test_get_batch_ignores_global_rng():
    torch.manual_seed(0)
    a = get_batch(torch.arange(1000), 8, 4, _gen(5))
    torch.manual_seed(123)
    b = get_batch(torch.arange(1000), 8, 4, _gen(5))
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_get_batch_start_range():
    x, _ = get_batch(torch.arange(10), 1000, 4, _gen(0))
    assert set(x[:, 0].tolist()) == {0, 1, 2, 3, 4, 5}


def test_get_batch_does_not_modify_data():
    data = torch.arange(20)
    before = data.clone()
    get_batch(data, 4, 4, _gen(0))
    assert torch.equal(data, before)
    assert data.dtype == torch.int64


@needs_cuda
def test_get_batch_cuda():
    x, y = get_batch(torch.arange(10), 2, 4, _gen(0), "cuda")
    assert x.device.type == "cuda"
    assert y.device.type == "cuda"
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64


# -------------------------------------------------------------- TrainConfig


def test_train_config_defaults():
    c = TrainConfig()
    assert c.batch_size == 32
    assert c.block_size == 8
    assert c.max_steps == 5000
    assert c.learning_rate == 0.01
    assert c.seed == 0
    assert c.log_every == 500


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"block_size": 0}, "block_size"),
        ({"max_steps": -1}, "max_steps"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"learning_rate": -0.1}, "learning_rate"),
        ({"log_every": 0}, "log_every"),
    ],
)
def test_train_config_invalid(kwargs, field):
    with pytest.raises(ValueError) as info:
        TrainConfig(**kwargs)
    assert field in str(info.value)


def test_train_config_boundaries_valid():
    TrainConfig(max_steps=0)
    TrainConfig(batch_size=1, block_size=1, log_every=1)


def test_train_config_frozen():
    c = TrainConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.batch_size = 1  # type: ignore[misc]


# ------------------------------------------------------------- compute_loss


def test_compute_loss_uniform():
    logits = torch.zeros(1, 2, 4, requires_grad=True)
    targets = torch.tensor([[0, 3]])
    loss = compute_loss(logits, targets)
    assert loss.item() == pytest.approx(LN4, abs=1e-5)
    assert loss.shape == torch.Size([])
    loss.backward()
    assert logits.grad is not None


def test_compute_loss_mean_over_positions():
    expected = 0.4100376
    logits = torch.tensor([[[0.0, 0.0], [2.0, 0.0]]])
    targets = torch.tensor([[0, 0]])
    assert compute_loss(logits, targets).item() == pytest.approx(expected, abs=1e-5)
    logits2 = torch.tensor([[[0.0, 0.0]], [[2.0, 0.0]]])
    targets2 = torch.tensor([[0], [0]])
    assert compute_loss(logits2, targets2).item() == pytest.approx(expected, abs=1e-5)


def test_compute_loss_noncontiguous_logits():
    logits = torch.zeros(1, 4, 2).transpose(1, 2)
    assert not logits.is_contiguous()
    targets = torch.tensor([[0, 3]])
    assert compute_loss(logits, targets).item() == pytest.approx(LN4, abs=1e-5)


@pytest.mark.parametrize(
    ("logits_shape", "targets_shape"),
    [((1, 2, 4), (1, 3)), ((2, 4), (2,)), ((1, 2, 4), (2, 1))],
    ids=["T_mismatch", "logits_2d", "B_mismatch"],
)
def test_compute_loss_bad_shapes(logits_shape, targets_shape):
    with pytest.raises(ValueError):
        compute_loss(torch.zeros(*logits_shape), torch.zeros(*targets_shape, dtype=torch.long))


# --------------------------------------------------------------- train_step


def test_train_step_docstring_example():
    model = _zero_embedding()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.tensor([[0, 1]])
    y = torch.tensor([[1, 2]])
    loss = train_step(model, optimizer, x, y)
    assert isinstance(loss, float)
    assert loss == pytest.approx(LN4, abs=1e-5)
    assert not torch.all(model.weight == 0)


def test_train_step_zeroes_grad_and_steps():
    model = _zero_embedding()
    model.weight.grad = torch.full((4, 4), 1000.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    x = torch.tensor([[0, 1]])
    y = torch.tensor([[1, 2]])
    loss = train_step(model, optimizer, x, y)
    assert loss == pytest.approx(LN4, abs=1e-5)
    expected = torch.tensor(
        [
            [-0.125, 0.375, -0.125, -0.125],
            [-0.125, -0.125, 0.375, -0.125],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert torch.allclose(model.weight.detach(), expected, atol=1e-5)


def test_train_step_loss_decreases_on_same_batch():
    torch.manual_seed(0)
    model = nn.Embedding(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.tensor([[0, 1, 2, 3]])
    y = torch.tensor([[1, 2, 3, 0]])
    losses = [train_step(model, optimizer, x, y) for _ in range(50)]
    assert losses[-1] < losses[0]


# -------------------------------------------------------------------- train


@pytest.mark.parametrize(
    ("max_steps", "log_every", "expected_steps", "expected_len"),
    [(5, 2, [0, 2, 4], 5), (4, 2, [0, 2, 3], 4), (1, 500, [0], 1)],
)
def test_train_on_log_steps(max_steps, log_every, expected_steps, expected_len):
    calls: list[tuple[int, float]] = []
    losses = train(
        nn.Embedding(4, 4),
        _cyclic_data(),
        TrainConfig(max_steps=max_steps, log_every=log_every),
        "cpu",
        on_log=lambda step, loss: calls.append((step, loss)),
    )
    assert [step for step, _ in calls] == expected_steps
    assert len(losses) == expected_len


def test_train_on_log_receives_listed_loss():
    calls: list[tuple[int, float]] = []
    losses = train(
        nn.Embedding(4, 4),
        _cyclic_data(),
        TrainConfig(max_steps=5, log_every=2),
        "cpu",
        on_log=lambda step, loss: calls.append((step, loss)),
    )
    assert calls
    for step, loss in calls:
        assert loss == losses[step]


def test_train_zero_steps():
    calls: list[tuple[int, float]] = []
    losses = train(
        nn.Embedding(4, 4),
        torch.tensor([0, 1, 2]),
        TrainConfig(max_steps=0),
        "cpu",
        on_log=lambda step, loss: calls.append((step, loss)),
    )
    assert losses == []
    assert calls == []


def test_train_too_short_data():
    with pytest.raises(ValueError):
        train(nn.Embedding(4, 4), torch.tensor([0, 1, 2]), TrainConfig(max_steps=1), "cpu")


def test_train_returns_list_of_floats():
    losses = train(
        nn.Embedding(4, 4),
        _cyclic_data(),
        TrainConfig(max_steps=10, log_every=5),
        "cpu",
        on_log=None,
    )
    assert isinstance(losses, list)
    assert len(losses) == 10
    assert all(isinstance(v, float) for v in losses)


def test_train_reproducible():
    runs = []
    for _ in range(2):
        torch.manual_seed(0)
        model = nn.Embedding(4, 4)
        runs.append(train(model, _cyclic_data(), TrainConfig(max_steps=20, seed=0), "cpu"))
    assert runs[0] == runs[1]


def test_train_loss_decreases():
    torch.manual_seed(0)
    model = nn.Embedding(4, 4)
    losses = train(model, _cyclic_data(), TrainConfig(max_steps=200), "cpu")
    first = math.fsum(losses[:20]) / 20
    last = math.fsum(losses[-20:]) / 20
    assert last < first


def test_train_sets_train_mode():
    model = nn.Embedding(4, 4)
    model.eval()
    train(model, _cyclic_data(), TrainConfig(max_steps=1), "cpu")
    assert model.training is True


@needs_cuda
def test_train_cuda():
    torch.manual_seed(0)
    model = nn.Embedding(4, 4).to("cuda")
    losses = train(model, _cyclic_data(), TrainConfig(max_steps=5), "cuda")
    assert len(losses) == 5
    assert all(isinstance(v, float) for v in losses)
    assert all(p.device.type == "cuda" for p in model.parameters())
