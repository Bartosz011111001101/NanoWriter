"""Testy get_masked_batch, get_batch oraz write_mask / load_mask."""

import warnings

import pytest
import torch

from nanowriter.dataset import (
    MASK_DTYPE,
    get_batch,
    get_masked_batch,
    load_mask,
    load_tokens,
    write_mask,
    write_tokens,
)


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _irregular_data_and_mask() -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.randperm(200, generator=_gen(123))
    mask = torch.randint(0, 2, (200,), generator=_gen(456)).bool()
    return data, mask


DOC_X = [[2, 3, 4, 5], [3, 4, 5, 6]]
DOC_Y = [[3, 4, 5, 6], [4, 5, 6, 7]]
DOC_M = [[False, True, False, True], [True, False, True, False]]


def test_mask_dtype_constant():
    assert MASK_DTYPE == "uint8"


def test_masked_batch_docstring_example():
    data = torch.arange(10)
    mask = data % 2 == 0
    x, y, m = get_masked_batch(data, mask, 2, 4, _gen(0))
    assert x.tolist() == DOC_X
    assert y.tolist() == DOC_Y
    assert m.tolist() == DOC_M
    assert list(x.shape) == [2, 4]
    assert list(y.shape) == [2, 4]
    assert list(m.shape) == [2, 4]
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
    assert m.dtype == torch.bool


def test_masked_batch_x_y_equal_get_batch():
    data = torch.arange(50)
    mask = data % 3 == 0
    for seed in range(5):
        x, y, _ = get_masked_batch(data, mask, 8, 5, _gen(seed))
        xb, yb = get_batch(data, 8, 5, _gen(seed))
        assert torch.equal(x, xb)
        assert torch.equal(y, yb)


def test_masked_batch_m_is_target_mask():
    data = torch.arange(10)
    mask = data % 2 == 0
    x, y, m = get_masked_batch(data, mask, 2, 4, _gen(0))
    assert torch.equal(m, y % 2 == 0)
    assert not torch.equal(m, x % 2 == 0)


def test_masked_batch_shift_y_x():
    data = torch.arange(20)
    mask = torch.ones(20, dtype=torch.bool)
    x, y, _ = get_masked_batch(data, mask, 4, 6, _gen(0))
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert torch.equal(y, x + 1)


def test_masked_batch_irregular_mask_window_origin():
    data, mask = _irregular_data_and_mask()
    x, y, m = get_masked_batch(data, mask, 16, 8, _gen(0))
    for b in range(16):
        s = int((data == x[b, 0]).nonzero().item())
        assert torch.equal(x[b], data[s : s + 8])
        assert torch.equal(y[b], data[s + 1 : s + 9])
        assert torch.equal(m[b], mask[s + 1 : s + 9])


def test_masked_batch_mask_int_and_bool_equal():
    data, mask = _irregular_data_and_mask()
    _, _, m_bool = get_masked_batch(data, mask, 16, 8, _gen(0))
    _, _, m_int = get_masked_batch(data, mask.to(torch.int64), 16, 8, _gen(0))
    assert torch.equal(m_bool, m_int)
    assert m_bool.dtype == torch.bool
    assert m_int.dtype == torch.bool


def test_masked_batch_mask_uint8_accepted():
    data = torch.arange(10)
    mask = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.uint8)
    _, _, m = get_masked_batch(data, mask, 2, 4, _gen(0))
    assert m.tolist() == [[True, False, True, False], [False, True, False, True]]
    assert m.dtype == torch.bool


def test_masked_batch_minimal_shapes():
    data = torch.arange(6)
    mask = torch.tensor([1, 0, 0, 1, 0, 1], dtype=torch.bool)
    x, y, m = get_masked_batch(data, mask, 1, 1, _gen(0))
    assert x.tolist() == [[4]]
    assert y.tolist() == [[5]]
    assert m.tolist() == [[True]]
    assert list(x.shape) == [1, 1]
    assert list(y.shape) == [1, 1]
    assert list(m.shape) == [1, 1]


def test_masked_batch_n_equals_block_size_plus_one():
    data = torch.arange(5) + 10
    mask = torch.tensor([1, 0, 1, 1, 0], dtype=torch.bool)
    x, y, m = get_masked_batch(data, mask, 3, 4, _gen(0))
    for b in range(3):
        assert x[b].tolist() == [10, 11, 12, 13]
        assert y[b].tolist() == [11, 12, 13, 14]
        assert m[b].tolist() == [False, True, True, False]
        assert torch.equal(m[b], mask[1:])


def test_masked_batch_all_ones_and_all_zeros_mask():
    data = torch.arange(30)
    _, _, m_ones = get_masked_batch(data, torch.ones(30, dtype=torch.bool), 4, 6, _gen(0))
    _, _, m_zeros = get_masked_batch(data, torch.zeros(30, dtype=torch.bool), 4, 6, _gen(0))
    assert list(m_ones.shape) == [4, 6]
    assert list(m_zeros.shape) == [4, 6]
    assert bool(m_ones.all())
    assert not bool(m_zeros.any())


def test_masked_batch_deterministic():
    data = torch.arange(100)
    mask = data % 5 == 0
    x1, y1, m1 = get_masked_batch(data, mask, 4, 8, _gen(7))
    x2, y2, m2 = get_masked_batch(data, mask, 4, 8, _gen(7))
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)
    assert torch.equal(m1, m2)


def test_masked_batch_consumes_generator_like_get_batch():
    data = torch.arange(100)
    mask = data % 5 == 0
    gen_a = _gen(3)
    gen_b = _gen(3)
    get_masked_batch(data, mask, 4, 8, gen_a)
    xa, ya = get_batch(data, 4, 8, gen_a)
    get_batch(data, 4, 8, gen_b)
    xb, yb = get_batch(data, 4, 8, gen_b)
    assert torch.equal(xa, xb)
    assert torch.equal(ya, yb)


def test_masked_batch_device_cpu():
    data = torch.arange(10)
    mask = data % 2 == 0
    x, y, m = get_masked_batch(data, mask, 2, 4, _gen(0), device="cpu")
    assert x.device.type == "cpu"
    assert y.device.type == "cpu"
    assert m.device.type == "cpu"
    x, y, m = get_masked_batch(data, mask, 2, 4, _gen(0))
    assert x.device.type == "cpu"
    assert y.device.type == "cpu"
    assert m.device.type == "cpu"


def test_masked_batch_integer_data_dtype():
    data = torch.arange(10, dtype=torch.int32)
    mask = data % 2 == 0
    x, y, m = get_masked_batch(data, mask, 2, 4, _gen(0))
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
    assert m.dtype == torch.bool
    assert x.tolist() == DOC_X
    assert y.tolist() == DOC_Y
    assert m.tolist() == DOC_M


def test_masked_batch_mask_shape_mismatch():
    data = torch.arange(6)
    with pytest.raises(ValueError) as excinfo:
        get_masked_batch(data, torch.zeros(5, dtype=torch.bool), 1, 2, _gen(0))
    assert "(5,)" in str(excinfo.value)
    assert "(6,)" in str(excinfo.value)
    with pytest.raises(ValueError):
        get_masked_batch(data, torch.zeros(7, dtype=torch.bool), 1, 2, _gen(0))
    with pytest.raises(ValueError):
        get_masked_batch(data, torch.zeros(6, 1, dtype=torch.bool), 1, 2, _gen(0))


@pytest.mark.parametrize(
    ("batch_size", "block_size", "two_d"),
    [
        (1, 2, True),
        (1, 10, False),
        (1, 11, False),
        (0, 4, False),
        (-1, 4, False),
        (2, 0, False),
        (2, -1, False),
    ],
)
def test_masked_batch_value_errors(batch_size, block_size, two_d):
    if two_d:
        data = torch.arange(9).view(3, 3)
        mask = data % 2 == 0
    else:
        data = torch.arange(10)
        mask = data % 2 == 0
    with pytest.raises(ValueError):
        get_masked_batch(data, mask, batch_size, block_size, _gen(0))


def test_masked_batch_works_with_data_from_loaded_files(tmp_path):
    ids = list(range(50))
    mask_list = [1 if (i * 7) % 5 < 2 else 0 for i in range(50)]
    write_tokens(ids, tmp_path / "t.bin")
    write_mask(mask_list, tmp_path / "m.bin")
    data = load_tokens(tmp_path / "t.bin")
    mask = load_mask(tmp_path / "m.bin")
    x, _, m = get_masked_batch(data, mask, 6, 7, _gen(0))
    for b in range(6):
        s = int(x[b, 0])
        for t in range(7):
            assert m[b, t].item() == bool(mask_list[s + 1 + t])
    assert m.dtype == torch.bool


def test_get_batch_exact_values_seed_0():
    x, y = get_batch(torch.arange(10), 2, 4, _gen(0))
    assert x.tolist() == DOC_X
    assert y.tolist() == DOC_Y


def test_get_batch_exact_values_seed_1():
    x, y = get_batch(torch.arange(31), 3, 5, _gen(1))
    assert x.tolist() == [
        [19, 20, 21, 22, 23],
        [9, 10, 11, 12, 13],
        [16, 17, 18, 19, 20],
    ]
    assert torch.equal(y, x + 1)


def test_get_batch_invariants():
    x, y = get_batch(torch.arange(20), 4, 6, _gen(0))
    assert torch.equal(y, x + 1)
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
    assert list(x.shape) == [4, 6]
    assert list(y.shape) == [4, 6]
    x, y = get_batch(torch.arange(5), 2, 4, _gen(0))
    for b in range(2):
        assert x[b].tolist() == [0, 1, 2, 3]
        assert y[b].tolist() == [1, 2, 3, 4]


def test_get_batch_integer_dtype_to_int64():
    x, y = get_batch(torch.arange(10, dtype=torch.int32), 2, 4, _gen(0))
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
    assert x.tolist() == DOC_X
    assert y.tolist() == DOC_Y


def test_get_batch_value_errors_and_messages():
    with pytest.raises(ValueError, match="jednowymiarowy"):
        get_batch(torch.arange(9).view(3, 3), 1, 2, _gen(0))
    with pytest.raises(ValueError, match="batch_size i block_size"):
        get_batch(torch.arange(10), 0, 4, _gen(0))
    with pytest.raises(ValueError, match="batch_size i block_size"):
        get_batch(torch.arange(10), 2, 0, _gen(0))
    with pytest.raises(ValueError, match="za krótkie data"):
        get_batch(torch.arange(4), 1, 4, _gen(0))
    with pytest.raises(ValueError, match="za krótkie data"):
        get_batch(torch.arange(3), 1, 4, _gen(0))


def test_write_mask_docstring_example(tmp_path):
    p = tmp_path / "m.bin"
    assert write_mask([0, 1, 1], p) == 3
    assert p.read_bytes() == b"\x00\x01\x01"


def test_write_mask_empty(tmp_path):
    p = tmp_path / "m.bin"
    assert write_mask([], p) == 0
    assert p.exists()
    assert p.stat().st_size == 0


def test_write_mask_one_byte_per_value(tmp_path):
    p = tmp_path / "m.bin"
    values = [(i * 13) % 2 for i in range(1000)]
    assert write_mask(values, p) == 1000
    raw = p.read_bytes()
    assert len(raw) == 1000
    assert list(raw) == values


@pytest.mark.parametrize("values", [[0, 2], [-1], [1, 0, 255]])
def test_write_mask_invalid_values(tmp_path, values):
    p = tmp_path / "m.bin"
    with pytest.raises(ValueError):
        write_mask(values, p)
    assert not p.exists()


def test_write_mask_invalid_does_not_create_dirs(tmp_path):
    p = tmp_path / "a" / "b" / "m.bin"
    with pytest.raises(ValueError):
        write_mask([0, 2], p)
    assert not (tmp_path / "a").exists()


def test_write_mask_creates_parent_dirs(tmp_path):
    p = tmp_path / "x" / "y" / "m.bin"
    assert write_mask([1, 0], p) == 2
    assert p.exists()
    assert p.read_bytes() == b"\x01\x00"


def test_write_mask_overwrites(tmp_path):
    p = tmp_path / "m.bin"
    write_mask([1, 1, 1, 1], p)
    write_mask([0, 1], p)
    assert p.read_bytes() == b"\x00\x01"


def test_load_mask_roundtrip(tmp_path):
    p = tmp_path / "m.bin"
    write_mask([0, 1, 1], p)
    t = load_mask(p)
    assert t.tolist() == [False, True, True]
    assert t.dtype == torch.bool
    assert t.dim() == 1
    assert t.shape == (3,)


def test_load_mask_empty_file(tmp_path):
    p = tmp_path / "m.bin"
    write_mask([], p)
    t = load_mask(p)
    assert t.shape == (0,)
    assert t.dtype == torch.bool
    q = tmp_path / "empty.bin"
    q.write_bytes(b"")
    t = load_mask(q)
    assert t.shape == (0,)
    assert t.dtype == torch.bool


def test_load_mask_roundtrip_long(tmp_path):
    p = tmp_path / "m.bin"
    values = [1 if (i * i + 3 * i) % 7 < 3 else 0 for i in range(5000)]
    assert write_mask(values, p) == 5000
    t = load_mask(p)
    assert t.shape == (5000,)
    assert t.tolist() == [bool(v) for v in values]


def test_load_mask_invalid_bytes(tmp_path):
    p = tmp_path / "m.bin"
    for raw in (b"\x00\x01\x02", b"\x02", b"\xff"):
        p.write_bytes(raw)
        with pytest.raises(ValueError):
            load_mask(p)
    p.write_bytes(b"\x01\x00\x01")
    assert load_mask(p).tolist() == [True, False, True]


def test_load_mask_returns_writable_copy(tmp_path):
    p = tmp_path / "m.bin"
    write_mask([0, 1, 1], p)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = load_mask(p)
        t[0] = True
    assert t[0].item() is True
    assert load_mask(p).tolist() == [False, True, True]


def test_load_mask_device_cpu(tmp_path):
    p = tmp_path / "m.bin"
    write_mask([0, 1], p)
    assert load_mask(p).device.type == "cpu"


def test_new_functions_print_nothing(tmp_path, capsys):
    data = torch.arange(10)
    mask = data % 2 == 0
    p = tmp_path / "m.bin"
    get_masked_batch(data, mask, 2, 4, _gen(0))
    write_mask([0, 1], p)
    load_mask(p)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
