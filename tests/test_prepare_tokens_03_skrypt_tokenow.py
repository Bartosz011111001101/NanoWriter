"""Testy skryptu scripts/prepare_tokens.py i bramka etapu 3."""

import json
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from nanowriter.data import load_text
from nanowriter.dataset import TOKEN_DTYPE, TokenMeta, load_meta, load_tokens
from nanowriter.tokenizer import BPETokenizer

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_tokens.py"

P = (
    "Zażółć gęślą jaźń. „Pan Wokulski — rzekł subiekt — nie przyjdzie dziś.”\n"
    "Ala ma kota, a kot ma Alę.\n"
)
GATE_TEXTS = [
    "",
    "Zażółć gęślą jaźń",
    "\U0001F600 \U0001F1F5\U0001F1F1 \U0001F44D\U0001F3FD",
    "\n\t",
    "„Cytat” — myślnik",
    "Ꮚ漢字 ünïcödé",
]
ALA_TRAIN = "Ala ma kota, a kot ma Ale.\n" * 200
ALA_VAL = "Kot ma Ale.\n" * 20
PL_TRAIN = "Zażółć gęślą jaźń. Pan Wokulski — rzekł — nie przyjdzie.\n" * 30
PL_VAL = "Żółw \U0001F600 i \U0001F1F5\U0001F1F1 nie było w treningu.\n"
ALA_META = TokenMeta(
    vocab_size=267,
    dtype="uint16",
    train_tokens=2000,
    train_chars=5400,
    val_tokens=120,
    val_chars=240,
)


def _write(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def _make_dir(path: Path, train: str, val: str) -> Path:
    path.mkdir(parents=True)
    _write(path / "train.txt", train)
    _write(path / "val.txt", val)
    return path


@pytest.fixture(scope="module")
def g():
    return runpy.run_path(str(SCRIPT))


@pytest.fixture(scope="module")
def main(g):
    return g["main"]


@pytest.fixture(scope="module")
def parse_args(g):
    return g["parse_args"]


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(P * 20, 400)


@pytest.fixture
def d(tmp_path):
    return _make_dir(tmp_path / "dane", ALA_TRAIN, ALA_VAL)


@pytest.fixture
def out(d):
    return d / "tokens"


@pytest.fixture
def ala_args(d, out):
    return [
        "--data-dir", str(d),
        "--out-dir", str(out),
        "--vocab-size", "300",
        "--sample-mb", "1",
    ]  # fmt: skip


@pytest.fixture
def dane_pl(tmp_path):
    return _make_dir(tmp_path / "dane_pl", PL_TRAIN, PL_VAL)


def _with(args, option, value):
    args = list(args)
    args[args.index(option) + 1] = value
    return args


# ---------------------------------------------------------------- bramka etapu 3


@pytest.mark.parametrize("x", GATE_TEXTS)
def test_gate_roundtrip_trained_tokenizer(tok, x):
    assert 257 <= tok.vocab_size <= 400
    ids = tok.encode(x)
    assert tok.decode(ids) == x
    assert all(0 <= i < tok.vocab_size for i in ids)
    assert tok.encode("") == []


def test_gate_roundtrip_after_save_load(tok, tmp_path):
    tok.save(tmp_path / "tok.json")
    tok2 = BPETokenizer.load(tmp_path / "tok.json")
    assert tok2.vocab_size == tok.vocab_size
    assert tok2.merges == tok.merges
    for x in [*GATE_TEXTS, P]:
        assert tok2.encode(x) == tok.encode(x)
        assert tok2.decode(tok2.encode(x)) == x


def test_gate_bin_files_dtype(main, d, out, ala_args):
    main(ala_args)
    meta_json = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    tok = BPETokenizer.load(out / "tokenizer.json")
    assert TOKEN_DTYPE == "uint16"
    assert meta_json["dtype"] == "uint16"
    assert (out / "train.bin").stat().st_size == 2 * meta_json["train_tokens"]
    assert (out / "val.bin").stat().st_size == 2 * meta_json["val_tokens"]
    for name in ("train", "val"):
        text = load_text(d / f"{name}.txt")
        expected = tok.encode(text)
        assert np.fromfile(out / f"{name}.bin", dtype="<u2").tolist() == expected
        loaded = load_tokens(out / f"{name}.bin")
        assert loaded.dtype == torch.int64
        assert loaded.dim() == 1
        assert loaded.tolist() == expected
        assert tok.decode(loaded.tolist()) == text


def test_gate_script_polish_and_emoji(main, dane_pl):
    out = dane_pl / "tokens"
    meta = main(["--data-dir", str(dane_pl), "--out-dir", str(out), "--vocab-size", "300"])
    assert meta.train_chars == 1710
    assert meta.val_chars == 33
    tok = BPETokenizer.load(out / "tokenizer.json")
    assert tok.decode(load_tokens(out / "val.bin").tolist()) == load_text(dane_pl / "val.txt")
    assert tok.decode(load_tokens(out / "train.bin").tolist()) == load_text(
        dane_pl / "train.txt"
    )
    assert all(0 <= i < meta.vocab_size for i in load_tokens(out / "val.bin").tolist())
    for x in GATE_TEXTS:
        assert tok.decode(tok.encode(x)) == x


# ---------------------------------------------------------------- wczytanie


def test_load_script_prints_nothing(capsys):
    res = runpy.run_path(str(SCRIPT))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert callable(res["main"])
    assert callable(res["parse_args"])


# ---------------------------------------------------------------- parse_args


def test_parse_args_defaults(parse_args):
    a = parse_args([])
    assert a.data_dir == Path("data")
    assert isinstance(a.data_dir, Path)
    assert a.out_dir == Path("data/tokens")
    assert isinstance(a.out_dir, Path)
    assert a.vocab_size == 1024
    assert isinstance(a.vocab_size, int)
    assert a.sample_mb == 8.0
    assert isinstance(a.sample_mb, float)


def test_parse_args_vocab_size(parse_args):
    a = parse_args(["--vocab-size", "300"])
    assert a.vocab_size == 300
    assert isinstance(a.vocab_size, int)
    assert a.data_dir == Path("data")
    assert a.out_dir == Path("data/tokens")
    assert a.sample_mb == 8.0


def test_parse_args_all_options(parse_args):
    a = parse_args(
        ["--data-dir", "a", "--out-dir", "b/c", "--vocab-size", "512", "--sample-mb", "0.5"]
    )
    assert a.data_dir == Path("a")
    assert a.out_dir == Path("b/c")
    assert isinstance(a.data_dir, Path)
    assert isinstance(a.out_dir, Path)
    assert a.vocab_size == 512
    assert isinstance(a.vocab_size, int)
    assert a.sample_mb == 0.5
    assert isinstance(a.sample_mb, float)


def test_parse_args_argv_none(parse_args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prepare_tokens.py", "--vocab-size", "777"])
    a = parse_args()
    assert a.vocab_size == 777
    assert a.data_dir == Path("data")


def test_parse_args_bad_int(parse_args):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--vocab-size", "abc"])
    assert exc.value.code != 0


def test_parse_args_help_lists_options(parse_args, capsys):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    for option in ("--data-dir", "--out-dir", "--vocab-size", "--sample-mb"):
        assert option in text


# ---------------------------------------------------------------- main


def test_main_example(main, out, ala_args, capsys):
    result = main(ala_args)
    assert isinstance(result, TokenMeta)
    assert result == ALA_META
    assert sorted(p.name for p in out.iterdir()) == [
        "meta.json",
        "tokenizer.json",
        "train.bin",
        "val.bin",
    ]
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 6
    assert lines[0] == "probka: 5400 z 5400 znakow, trening BPE (vocab 300)"
    assert lines[1] == f"slownik: 267 tokenow -> {out / 'tokenizer.json'}"
    assert lines[2] == (
        f"train: 2000 tokenow, 5400 znakow, 2.700 znaku/token -> {out / 'train.bin'}"
    )
    assert lines[3] == f"val: 120 tokenow, 240 znakow, 2.000 znaku/token -> {out / 'val.bin'}"
    assert lines[4] == f"meta -> {out / 'meta.json'}"
    assert re.match(r"^czas: \d+\.\d s$", lines[5])


def test_main_meta_file_matches_result(main, out, ala_args):
    m = main(ala_args)
    assert load_meta(out / "meta.json") == m
    data = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert list(data) == [
        "vocab_size",
        "dtype",
        "train_tokens",
        "train_chars",
        "val_tokens",
        "val_chars",
    ]


def test_main_tokenizer_file(main, out, ala_args):
    m = main(ala_args)
    tok = BPETokenizer.load(out / "tokenizer.json")
    assert tok.vocab_size == m.vocab_size == 267
    assert len(tok.merges) == 11
    data = json.loads((out / "tokenizer.json").read_text(encoding="utf-8"))
    assert data["type"] == "bpe-bytes"


def test_main_smaller_sample(main, d, out, ala_args, capsys):
    m = main(_with(ala_args, "--sample-mb", "0.001"))
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "probka: 917 z 5400 znakow, trening BPE (vocab 300)"
    assert m.train_chars == 5400
    assert m.val_chars == 240
    tok = BPETokenizer.load(out / "tokenizer.json")
    assert m.train_tokens == len(tok.encode(load_text(d / "train.txt")))
    assert m.train_tokens == (out / "train.bin").stat().st_size // 2


def test_main_vocab_256_no_merges(main, ala_args, capsys):
    m = main(_with(ala_args, "--vocab-size", "256"))
    assert m == TokenMeta(
        vocab_size=256,
        dtype="uint16",
        train_tokens=5400,
        train_chars=5400,
        val_tokens=240,
        val_chars=240,
    )
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].startswith("slownik: 256 tokenow -> ")
    train_line = next(ln for ln in lines if ln.startswith("train:"))
    assert "5400 tokenow, 5400 znakow, 1.000 znaku/token" in train_line


def test_main_empty_files(main, tmp_path, capsys):
    e = _make_dir(tmp_path / "puste", "", "")
    out = e / "tokens"
    m = main(["--data-dir", str(e), "--out-dir", str(out), "--vocab-size", "300"])
    assert m == TokenMeta(
        vocab_size=256,
        dtype="uint16",
        train_tokens=0,
        train_chars=0,
        val_tokens=0,
        val_chars=0,
    )
    assert (out / "train.bin").stat().st_size == 0
    assert (out / "val.bin").stat().st_size == 0
    lines = capsys.readouterr().out.splitlines()
    assert "probka: 0 z 0 znakow, trening BPE (vocab 300)" in lines
    assert any(ln.startswith("train: 0 tokenow, 0 znakow, 0.000 znaku/token -> ") for ln in lines)
    assert any(ln.startswith("val: 0 tokenow, 0 znakow, 0.000 znaku/token -> ") for ln in lines)


def test_main_creates_nested_out_dir(main, d, ala_args, tmp_path):
    nested = tmp_path / "x" / "y" / "tokens"
    main(_with(ala_args, "--out-dir", str(nested)))
    assert sorted(p.name for p in nested.iterdir()) == [
        "meta.json",
        "tokenizer.json",
        "train.bin",
        "val.bin",
    ]


def test_main_existing_out_dir(main, out, ala_args):
    main(ala_args)
    second = main(_with(ala_args, "--vocab-size", "256"))
    assert second.vocab_size == 256
    assert load_meta(out / "meta.json").vocab_size == 256
    assert BPETokenizer.load(out / "tokenizer.json").vocab_size == 256
    assert (out / "train.bin").stat().st_size == 2 * 5400


def test_main_reproducible(main, d, ala_args, tmp_path):
    o1 = tmp_path / "o1"
    o2 = tmp_path / "o2"
    main(_with(ala_args, "--out-dir", str(o1)))
    main(_with(ala_args, "--out-dir", str(o2)))
    for name in ("tokenizer.json", "train.bin", "val.bin", "meta.json"):
        assert (o1 / name).read_bytes() == (o2 / name).read_bytes()


def test_main_argv_none(main, ala_args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prepare_tokens.py", *ala_args])
    assert main() == ALA_META


def test_main_missing_train(main, tmp_path):
    d = tmp_path / "bez_train"
    d.mkdir()
    _write(d / "val.txt", ALA_VAL)
    with pytest.raises(FileNotFoundError):
        main(["--data-dir", str(d), "--out-dir", str(d / "tokens"), "--vocab-size", "300"])
    assert not (d / "tokens").exists()


def test_main_missing_val(main, tmp_path):
    d = tmp_path / "bez_val"
    d.mkdir()
    _write(d / "train.txt", ALA_TRAIN)
    with pytest.raises(FileNotFoundError):
        main(
            [
                "--data-dir", str(d),
                "--out-dir", str(d / "tokens"),
                "--vocab-size", "300",
                "--sample-mb", "1",
            ]  # fmt: skip
        )


def test_main_vocab_too_small(main, out, ala_args):
    with pytest.raises(ValueError):
        main(_with(ala_args, "--vocab-size", "100"))
    assert not out.exists()


def test_main_sample_mb_zero(main, ala_args):
    with pytest.raises(ValueError):
        main(_with(ala_args, "--sample-mb", "0"))


def test_script_utf8_with_redirected_stdout(tmp_path):
    katalog = _make_dir(tmp_path / "dane_żółć", PL_TRAIN, PL_VAL)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
    env["PYTHONIOENCODING"] = "ascii"
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--data-dir", str(katalog),
            "--out-dir", str(katalog / "tokens"),
            "--vocab-size", "300",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=180,
        check=False,
    )  # fmt: skip
    stderr = proc.stderr.decode("utf-8", errors="replace")
    assert proc.returncode == 0, stderr
    lines = proc.stdout.decode("utf-8").splitlines()
    assert "probka: 1710 z 1710 znakow, trening BPE (vocab 300)" in lines
    assert any(ln.startswith("slownik: ") and "dane_żółć" in ln for ln in lines)
    assert (katalog / "tokens" / "meta.json").exists()
