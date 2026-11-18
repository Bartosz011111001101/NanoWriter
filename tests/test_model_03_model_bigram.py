"""Testy BigramLanguageModel."""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from nanowriter.model import BigramLanguageModel


def _make(vocab_size):
    torch.manual_seed(0)
    return BigramLanguageModel(vocab_size)


def _count_params(model):
    return sum(p.numel() for p in model.parameters())


def test_is_nn_module_with_fields():
    model = _make(65)
    assert isinstance(model, nn.Module)
    assert model.vocab_size == 65
    assert isinstance(model.token_embedding, nn.Embedding)
    assert model.token_embedding.weight.shape == torch.Size([65, 65])


def test_forward_shape_and_dtype():
    model = _make(102)
    model.eval()
    idx = torch.randint(0, 102, (2, 5))
    logits = model(idx)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 5, 102)
    assert logits.dtype == torch.float32


def test_forward_b1_t1():
    model = _make(102)
    model.eval()
    logits = model(torch.tensor([[7]]))
    assert logits.shape == (1, 1, 102)


def test_vocab_size_one():
    model = _make(1)
    model.eval()
    idx = torch.zeros(1, 3, dtype=torch.long)
    logits = model(idx)
    assert logits.shape == (1, 3, 1)
    assert _count_params(model) == 1


def test_logits_are_table_rows():
    model = _make(10)
    model.eval()
    idx = torch.tensor([[3, 0, 9], [5, 5, 1]])
    logits = model(idx)
    for b in range(idx.shape[0]):
        for t in range(idx.shape[1]):
            row = model.token_embedding.weight[idx[b, t]]
            assert torch.allclose(logits[b, t], row, atol=1e-6)


def test_parameter_count():
    assert _count_params(_make(102)) == 10404
    assert _count_params(_make(64)) == 4096


def test_single_parameter_no_buffers():
    model = _make(10)
    names = [name for name, _ in model.named_parameters()]
    assert names == ["token_embedding.weight"]
    assert list(model.buffers()) == []


def test_init_std_and_mean():
    model = _make(102)
    weight = model.token_embedding.weight
    assert 0.015 <= weight.std().item() <= 0.025
    assert abs(weight.mean().item()) < 0.005


def test_initial_loss_near_ln_vocab():
    torch.manual_seed(0)
    model = BigramLanguageModel(64)
    model.eval()
    idx = torch.randint(0, 64, (4, 16))
    targets = torch.randint(0, 64, (4, 16))
    logits = model(idx)
    loss = F.cross_entropy(logits.reshape(-1, 64), targets.reshape(-1)).item()
    expected = math.log(64)
    assert 0.9 * expected <= loss <= 1.1 * expected


def test_only_current_token_matters():
    model = _make(64)
    model.eval()
    torch.manual_seed(0)
    idx = torch.randint(0, 64, (1, 8))
    changed = idx.clone()
    changed[0, 3] = (idx[0, 3] + 1) % 64
    logits = model(idx)
    logits_changed = model(changed)
    for pos in (0, 1, 2, 4, 5, 6, 7):
        assert torch.allclose(logits[0, pos], logits_changed[0, pos], atol=1e-5)
    assert not torch.allclose(logits[0, 3], logits_changed[0, 3], atol=1e-5)


def test_same_token_same_logits():
    model = _make(20)
    model.eval()
    logits = model(torch.tensor([[4, 7, 4]]))
    assert torch.allclose(logits[0, 0], logits[0, 2], atol=1e-6)


def test_gradients_nonzero():
    model = _make(64)
    torch.manual_seed(0)
    idx = torch.randint(0, 64, (2, 8))
    targets = torch.randint(0, 64, (2, 8))
    logits = model(idx)
    loss = F.cross_entropy(logits.reshape(-1, 64), targets.reshape(-1))
    loss.backward()
    grad = model.token_embedding.weight.grad
    assert grad is not None
    assert torch.count_nonzero(grad) > 0


def test_invalid_vocab_size():
    with pytest.raises(ValueError):
        BigramLanguageModel(0)
    with pytest.raises(ValueError, match="-1"):
        BigramLanguageModel(-1)


def test_out_of_range_id_raises():
    model = _make(5)
    with pytest.raises((IndexError, RuntimeError)):
        model(torch.tensor([[5]]))
