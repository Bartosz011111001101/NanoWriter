import math

import pytest
import torch
from torch import nn

from nanowriter.sample import generate


def _random_model(vocab: int) -> nn.Embedding:
    torch.manual_seed(0)
    return nn.Embedding(vocab, vocab)


def _deterministic_model(vocab: int) -> nn.Embedding:
    model = nn.Embedding(vocab, vocab)
    with torch.no_grad():
        model.weight.zero_()
        for i in range(vocab):
            model.weight[i, (i + 1) % vocab] = 100.0
    return model


class _RecordingModel(nn.Module):
    def __init__(self, vocab: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, vocab)
        self.lengths: list[int] = []
        self.grad_enabled: list[bool] = []

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        self.lengths.append(idx.shape[1])
        self.grad_enabled.append(torch.is_grad_enabled())
        return self.emb(idx)


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_generate_docstring_example():
    model = _random_model(5)
    idx = torch.tensor([[1, 2]])

    out = generate(model, idx, 4, generator=_gen(0))

    assert out.shape == (1, 6)
    assert out.dtype == torch.int64
    assert out[0, :2].tolist() == [1, 2]
    assert ((out >= 0) & (out < 5)).all()


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("length", [1, 3])
@pytest.mark.parametrize("max_new_tokens", [1, 10])
def test_generate_length(batch, length, max_new_tokens):
    model = _random_model(8)
    idx = torch.randint(0, 8, (batch, length))

    out = generate(model, idx, max_new_tokens, generator=_gen(0))

    assert out.shape == (batch, length + max_new_tokens)
    assert torch.equal(out[:, :length], idx)
    new = out[:, length:]
    assert ((new >= 0) & (new < 8)).all()
    assert out.dtype == torch.int64


def test_generate_does_not_modify_idx():
    model = _random_model(5)
    idx = torch.tensor([[1, 2]])
    before = idx.clone()

    generate(model, idx, 5, generator=_gen(0))

    assert torch.equal(idx, before)
    assert idx.shape == (1, 2)


def test_generate_deterministic_model():
    model = _deterministic_model(5)

    out_a = generate(model, torch.tensor([[0]]), 4)
    out_b = generate(model, torch.tensor([[3]]), 4)

    assert out_a.tolist() == [[0, 1, 2, 3, 4]]
    assert out_b.tolist() == [[3, 4, 0, 1, 2]]


def test_generate_uses_last_position():
    model = _deterministic_model(5)

    out = generate(model, torch.tensor([[0, 3]]), 2)

    assert out.tolist() == [[0, 3, 4, 0]]


def test_generate_per_row():
    model = _deterministic_model(5)

    out = generate(model, torch.tensor([[0], [2]]), 3)

    assert out.tolist() == [[0, 1, 2, 3], [2, 3, 4, 0]]


def test_generate_samples_from_softmax():
    model = nn.Embedding(3, 3)
    with torch.no_grad():
        model.weight[0] = torch.tensor([math.log(3), 0.0, -100.0])
    idx = torch.zeros(2000, 1, dtype=torch.long)

    out = generate(model, idx, 1, generator=_gen(0))

    new = out[:, 1]
    share0 = (new == 0).float().mean().item()
    share1 = (new == 1).float().mean().item()
    assert 0.70 <= share0 <= 0.80
    assert 0.20 <= share1 <= 0.30
    assert (new == 2).sum().item() == 0


def test_generate_same_seed_same_result():
    model = _random_model(5)
    idx = torch.tensor([[1, 2]])

    a = generate(model, idx, 20, generator=_gen(0))
    b = generate(model, idx, 20, generator=_gen(0))

    assert torch.equal(a, b)


def test_generate_different_seeds_differ():
    model = _random_model(5)
    idx = torch.tensor([[1]])

    a = generate(model, idx, 50, generator=_gen(0))
    b = generate(model, idx, 50, generator=_gen(1))

    assert not torch.equal(a, b)


def test_generate_none_generator_uses_global_seed():
    model = _random_model(5)
    idx = torch.tensor([[1]])

    torch.manual_seed(42)
    a = generate(model, idx, 20)
    torch.manual_seed(42)
    b = generate(model, idx, 20)

    assert torch.equal(a, b)
    assert a.shape == (1, 21)


def test_generate_block_size_limits_context():
    model = _RecordingModel(5)
    idx = torch.tensor([[1]])

    out = generate(model, idx, 10, generator=_gen(0), block_size=2)

    assert model.lengths == [1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    assert out.shape == (1, 11)


def test_generate_no_block_size_full_context():
    model = _RecordingModel(5)
    idx = torch.tensor([[1]])

    generate(model, idx, 10, generator=_gen(0))

    assert model.lengths == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_generate_block_size_boundaries():
    idx = torch.tensor([[1, 2, 3]])

    model_equal = _RecordingModel(5)
    generate(model_equal, idx, 2, generator=_gen(0), block_size=3)
    assert model_equal.lengths == [3, 3]

    model_large = _RecordingModel(5)
    generate(model_large, idx, 3, generator=_gen(0), block_size=100)
    assert model_large.lengths == [3, 4, 5]


def test_generate_block_size_one_same_for_bigram():
    model = _random_model(5)
    idx = torch.tensor([[1, 2]])

    without = generate(model, idx, 4, generator=_gen(0))
    with_block = generate(model, idx, 4, generator=_gen(0), block_size=1)

    assert torch.equal(without, with_block)


def test_generate_zero_new_tokens():
    model = _random_model(5)
    idx = torch.tensor([[1, 2]])

    out = generate(model, idx, 0)

    assert torch.equal(out, idx)
    assert out is not idx
    out[0, 0] = 4
    assert idx.tolist() == [[1, 2]]


@pytest.mark.parametrize(
    ("idx", "kwargs"),
    [
        (torch.tensor([[1]]), {"max_new_tokens": -1}),
        (torch.tensor([1, 2]), {}),
        (torch.zeros(1, 1, 1, dtype=torch.long), {}),
        (torch.zeros(1, 0, dtype=torch.long), {}),
        (torch.tensor([[1]]), {"block_size": 0}),
        (torch.tensor([[1]]), {"block_size": -1}),
    ],
    ids=[
        "max_new_tokens_negative",
        "idx_1d",
        "idx_3d",
        "idx_empty_context",
        "block_size_zero",
        "block_size_negative",
    ],
)
def test_generate_invalid_args(idx, kwargs):
    model = _random_model(5)
    max_new_tokens = kwargs.pop("max_new_tokens", 2)

    with pytest.raises(ValueError):
        generate(model, idx, max_new_tokens, **kwargs)


def test_generate_keyword_only():
    model = _random_model(5)

    with pytest.raises(TypeError):
        generate(model, torch.tensor([[1]]), 2, _gen(0))  # type: ignore[misc]  # pylint: disable=too-many-function-args


def test_generate_no_grad():
    model = _RecordingModel(5)

    out = generate(model, torch.tensor([[1]]), 3, generator=_gen(0))

    assert model.grad_enabled == [False, False, False]
    assert out.requires_grad is False


@pytest.mark.parametrize("train_mode", [True, False])
def test_generate_keeps_model_mode(train_mode):
    model = _random_model(5)
    model.train(train_mode)

    generate(model, torch.tensor([[1]]), 3, generator=_gen(0))

    assert model.training is train_mode


@pytest.mark.skipif(not torch.cuda.is_available(), reason="brak GPU")
def test_generate_cuda():
    model = _random_model(5).to("cuda")
    idx = torch.tensor([[1, 2]], device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(0)

    out = generate(model, idx, 4, generator=generator)

    assert out.device.type == "cuda"
    assert out.shape == (1, 6)
    assert out.dtype == torch.int64
