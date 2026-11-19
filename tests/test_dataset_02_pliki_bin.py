"""Testy plików .bin z tokenami, encode_file i metadanych (dataset)."""

import dataclasses
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from nanowriter.dataset import (
    MAX_TOKEN_ID,
    TOKEN_DTYPE,
    TokenMeta,
    encode_file,
    get_batch,
    load_meta,
    load_tokens,
    save_meta,
    write_tokens,
)
from nanowriter.tokenizer import CharTokenizer

META = TokenMeta(1024, "uint16", 10, 25, 2, 5)
META_TEXT = (
    '{\n  "vocab_size": 1024,\n  "dtype": "uint16",\n  "train_tokens": 10,\n'
    '  "train_chars": 25,\n  "val_tokens": 2,\n  "val_chars": 5\n}\n'
)
META_FIELDS = ["vocab_size", "dtype", "train_tokens", "train_chars", "val_tokens", "val_chars"]


class Atrapa:
    """Atrapa tokenizera: zapamiętuje wywołania encode i zwraca stałą listę."""

    def __init__(self, vocab_size: int, ids: list[int]) -> None:
        self.vocab_size = vocab_size
        self.ids = ids
        self.calls: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.calls.append(text)
        return list(self.ids)


def _text_file(tmp_path: Path, data: bytes, name: str = "t.txt") -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


# Stałe


def test_constants():
    assert TOKEN_DTYPE == "uint16"
    assert MAX_TOKEN_ID == 65535
    assert np.dtype(TOKEN_DTYPE).itemsize == 2


# write_tokens / load_tokens


def test_write_tokens_docstring_example(tmp_path):
    p = tmp_path / "x.bin"
    assert write_tokens([1, 2, 1023], p) == 3
    assert p.read_bytes() == b"\x01\x00\x02\x00\xff\x03"
    t = load_tokens(p)
    assert t.tolist() == [1, 2, 1023]
    assert t.dtype == torch.int64


def test_write_tokens_empty(tmp_path):
    p = tmp_path / "x.bin"
    assert write_tokens([], p) == 0
    assert p.exists()
    assert p.stat().st_size == 0
    t = load_tokens(p)
    assert t.shape == (0,)
    assert t.dtype == torch.int64


def test_write_tokens_bounds(tmp_path):
    p = tmp_path / "x.bin"
    assert write_tokens([0, 65535], p) == 2
    assert p.read_bytes() == b"\x00\x00\xff\xff"
    assert load_tokens(p).tolist() == [0, 65535]


@pytest.mark.parametrize("ids", [[65536], [-1], [1, 70000, 2], [5, -3]])
def test_write_tokens_out_of_range(tmp_path, ids):
    p = tmp_path / "x.bin"
    with pytest.raises(ValueError):
        write_tokens(ids, p)
    assert not p.exists()


def test_write_tokens_random_roundtrip(tmp_path):
    p = tmp_path / "x.bin"
    rng = random.Random(0)
    ids = [rng.randrange(0, 65536) for _ in range(10_000)]
    assert write_tokens(ids, p) == 10_000
    assert p.stat().st_size == 20_000
    assert load_tokens(p).tolist() == ids
    assert np.fromfile(p, dtype="<u2").tolist() == ids


def test_write_tokens_creates_dirs(tmp_path):
    p = tmp_path / "a" / "b" / "x.bin"
    assert write_tokens([7], p) == 1
    assert p.exists()
    assert p.read_bytes() == b"\x07\x00"


def test_write_tokens_overwrites(tmp_path):
    p = tmp_path / "x.bin"
    write_tokens([1, 2, 3], p)
    write_tokens([7], p)
    assert load_tokens(p).tolist() == [7]
    assert p.stat().st_size == 2


def test_load_tokens_type_and_device(tmp_path):
    p = tmp_path / "x.bin"
    write_tokens([4, 5, 6], p)
    t = load_tokens(p)
    assert isinstance(t, torch.Tensor)
    assert t.dim() == 1
    assert t.dtype == torch.int64
    assert t.device.type == "cpu"


@pytest.mark.parametrize("data", [b"\x01\x00\x02", b"\x01"])
def test_load_tokens_odd_size(tmp_path, data):
    p = tmp_path / "x.bin"
    p.write_bytes(data)
    with pytest.raises(ValueError):
        load_tokens(p)


def test_load_tokens_writable_copy(tmp_path):
    p = tmp_path / "x.bin"
    write_tokens([1, 2, 3], p)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = load_tokens(p)
        t[0] = 99
    assert t.tolist() == [99, 2, 3]
    assert p.read_bytes() == b"\x01\x00\x02\x00\x03\x00"
    assert load_tokens(p).tolist() == [1, 2, 3]


def test_get_batch_on_loaded_tokens(tmp_path):
    p = tmp_path / "x.bin"
    write_tokens(list(range(100)), p)
    data = load_tokens(p)
    x, y = get_batch(data, 4, 8, torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (4, 8)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert torch.equal(y, x + 1)


# encode_file


def test_encode_file_docstring_example(tmp_path):
    t = _text_file(tmp_path, b"cab")
    out = tmp_path / "t.bin"
    result = encode_file(CharTokenizer.from_text("abca"), t, out)
    assert result == (3, 3)
    assert isinstance(result, tuple)
    assert all(isinstance(v, int) for v in result)
    assert load_tokens(out).tolist() == [2, 0, 1]


def test_encode_file_counts_chars_not_bytes(tmp_path):
    t = _text_file(tmp_path, "zażółć\n".encode())
    out = tmp_path / "t.bin"
    tok = CharTokenizer.from_text("zażółć\n")
    assert encode_file(tok, t, out) == (7, 7)
    assert load_tokens(out).tolist() == tok.encode("zażółć\n")


def test_encode_file_uses_load_text(tmp_path):
    t = _text_file(tmp_path, b"ab\r\ncd")
    out = tmp_path / "t.bin"
    tok = CharTokenizer.from_text("ab\ncd")
    assert encode_file(tok, t, out) == (5, 5)
    assert load_tokens(out).tolist() == tok.encode("ab\ncd")


def test_encode_file_single_encode_call(tmp_path):
    t = _text_file(tmp_path, b"ab\ncd\n")
    out = tmp_path / "t.bin"
    a = Atrapa(1024, [5, 6, 1023])
    assert encode_file(a, t, out) == (3, 6)
    assert a.calls == ["ab\ncd\n"]
    assert load_tokens(out).tolist() == [5, 6, 1023]


def test_encode_file_empty(tmp_path):
    t = _text_file(tmp_path, b"")
    out = tmp_path / "t.bin"
    assert encode_file(CharTokenizer.from_text("abc"), t, out) == (0, 0)
    assert out.exists()
    assert out.stat().st_size == 0


def test_encode_file_creates_dirs(tmp_path):
    t = _text_file(tmp_path, b"cab")
    out = tmp_path / "tokens" / "sub" / "t.bin"
    assert encode_file(CharTokenizer.from_text("abc"), t, out) == (3, 3)
    assert out.exists()


@pytest.mark.parametrize("vocab_size", [70000, 65537])
def test_encode_file_vocab_too_large_existing_file(tmp_path, vocab_size):
    t = _text_file(tmp_path, b"abc")
    out = tmp_path / "t.bin"
    a = Atrapa(vocab_size, [1])
    with pytest.raises(ValueError):
        encode_file(a, t, out)
    assert not out.exists()
    assert a.calls == []


def test_encode_file_vocab_too_large_missing_text(tmp_path):
    out = tmp_path / "t.bin"
    a = Atrapa(70000, [1])
    with pytest.raises(ValueError):
        encode_file(a, tmp_path / "brak.txt", out)
    assert not out.exists()


def test_encode_file_vocab_boundary(tmp_path):
    t = _text_file(tmp_path, b"x")
    out = tmp_path / "t.bin"
    assert encode_file(Atrapa(65536, [65535]), t, out) == (1, 1)
    assert load_tokens(out).tolist() == [65535]


def test_encode_file_bad_ids_from_tokenizer(tmp_path):
    t = _text_file(tmp_path, b"x")
    out = tmp_path / "t.bin"
    with pytest.raises(ValueError):
        encode_file(Atrapa(65536, [70000]), t, out)
    assert not out.exists()


# TokenMeta, save_meta, load_meta


def test_token_meta_fields():
    assert META.train_chars == 25
    assert [f.name for f in dataclasses.fields(TokenMeta)] == META_FIELDS
    by_name = TokenMeta(
        vocab_size=1024,
        dtype="uint16",
        train_tokens=10,
        train_chars=25,
        val_tokens=2,
        val_chars=5,
    )
    assert by_name == META
    with pytest.raises(TypeError):
        TokenMeta()  # type: ignore[call-arg]  # pylint: disable=no-value-for-parameter


def test_token_meta_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        META.vocab_size = 1  # type: ignore[misc]


def test_save_meta_exact_file(tmp_path):
    p = tmp_path / "meta.json"
    save_meta(META, p)
    assert p.read_bytes() == META_TEXT.encode()


def test_save_meta_creates_dirs(tmp_path):
    p = tmp_path / "a" / "b" / "meta.json"
    save_meta(META, p)
    assert p.exists()
    assert load_meta(p) == META


def test_meta_roundtrip(tmp_path):
    p = tmp_path / "meta.json"
    m = TokenMeta(1024, "uint16", 11_000_000, 25_000_000, 0, 0)
    save_meta(m, p)
    loaded = load_meta(p)
    assert loaded == m
    assert isinstance(loaded, TokenMeta)


@pytest.mark.parametrize("key", META_FIELDS)
def test_load_meta_missing_key(tmp_path, key):
    p = tmp_path / "meta.json"
    data = {k: v for k, v in dataclasses.asdict(META).items() if k != key}
    p.write_bytes(json.dumps(data).encode())
    with pytest.raises(ValueError):
        load_meta(p)


def test_load_meta_wrong_dtype(tmp_path):
    p = tmp_path / "meta.json"
    p.write_bytes(json.dumps({**dataclasses.asdict(META), "dtype": "int32"}).encode())
    with pytest.raises(ValueError):
        load_meta(p)


def test_load_meta_extra_key_ignored(tmp_path):
    p = tmp_path / "meta.json"
    p.write_bytes(json.dumps({**dataclasses.asdict(META), "inne": 1}).encode())
    assert load_meta(p) == META


# Ogólne


def test_module_prints_nothing(tmp_path, capsys):
    p = tmp_path / "x.bin"
    t = _text_file(tmp_path, b"cab")
    out = tmp_path / "t.bin"
    m = tmp_path / "meta.json"
    write_tokens([1, 2], p)
    load_tokens(p)
    encode_file(CharTokenizer.from_text("abc"), t, out)
    save_meta(META, m)
    load_meta(m)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
