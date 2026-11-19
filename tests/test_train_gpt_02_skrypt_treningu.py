"""Testy skryptu scripts/train_gpt.py oraz bramki etapu 5 (wznowienie z checkpointu)."""

import json
import math
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from nanowriter.dataset import TokenMeta, load_tokens, save_meta, write_tokens
from nanowriter.model import GPT, GPTConfig
from nanowriter.train import (
    TrainConfig,
    bits_per_char,
    estimate_loss,
    load_checkpoint,
    load_gpt,
    train,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_gpt.py"

LN32 = math.log(32)
RATIO = 600 / (1200 * math.log(2))

STEP_RE = re.compile(r"^krok +(\d+)  strata (\d+\.\d{4})$")
VAL_RE = re.compile(r"^krok +(\d+)  val (\d+\.\d{4})  bity/znak (\d+\.\d{4})$")
FINAL_RE = re.compile(r"^val \(calosc\): strata (\d+\.\d{4}), bity/znak (\d+\.\d{4})$")
TIME_RE = re.compile(r"^czas: \d+\.\d s$")

DATA_LINE = "dane: train 3000 tokenow, val 600 tokenow, slownik 32"
MODEL_LINE = "model: 4080 parametrow, urzadzenie cpu"

ALL_OPTIONS = [
    "--data-dir",
    "--out-dir",
    "--steps",
    "--batch-size",
    "--block-size",
    "--n-layer",
    "--n-head",
    "--d-model",
    "--lr",
    "--min-lr",
    "--warmup",
    "--weight-decay",
    "--grad-clip",
    "--eval-every",
    "--eval-batches",
    "--checkpoint-every",
    "--log-every",
    "--seed",
    "--device",
    "--no-bf16",
    "--resume",
    "--until-step",
]


@pytest.fixture(scope="module")
def script_funcs():
    g = runpy.run_path(str(SCRIPT))
    return g["parse_args"], g["main"]


@pytest.fixture
def parse_args(script_funcs):
    return script_funcs[0]


@pytest.fixture
def main(script_funcs):
    return script_funcs[1]


@pytest.fixture
def d(tmp_path):
    d = tmp_path / "tokens"
    write_tokens([i % 32 for i in range(3000)], d / "train.bin")
    write_tokens([i % 32 for i in range(600)], d / "val.bin")
    save_meta(TokenMeta(32, "uint16", 3000, 6000, 600, 1200), d / "meta.json")
    return d


def bazowe(d, out):
    return [
        "--data-dir", str(d),
        "--out-dir", str(out),
        "--device", "cpu",
        "--steps", "20",
        "--batch-size", "4",
        "--block-size", "16",
        "--n-layer", "1",
        "--n-head", "2",
        "--d-model", "16",
        "--eval-every", "10",
        "--eval-batches", "2",
        "--log-every", "10",
        "--warmup", "2",
        "--checkpoint-every", "0",
    ]  # fmt: skip


def read_log(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def step_lines(out):
    """Pary (krok, tekst straty) z linii straty."""
    res = []
    for line in out.splitlines():
        m = STEP_RE.match(line)
        if m:
            res.append((int(m.group(1)), m.group(2)))
    return res


def val_lines(out):
    """Trójki (krok, val, bity) z linii walidacji."""
    res = []
    for line in out.splitlines():
        m = VAL_RE.match(line)
        if m:
            res.append((int(m.group(1)), m.group(2), m.group(3)))
    return res


def loss_and_val_lines(out):
    return [
        line for line in out.splitlines() if STEP_RE.match(line) or VAL_RE.match(line)
    ]


# ---------------------------------------------------------------------------
# Bramka etapu 5 na modułach (bez skryptu)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 3, 6, 11])
def test_gate_resume_equals_uninterrupted(tmp_path, k):
    g = torch.Generator().manual_seed(1)
    data = torch.randint(0, 32, (3000,), generator=g)
    val = torch.randint(0, 32, (600,), generator=g)
    gcfg = GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, d_model=16)
    cfg = TrainConfig(
        batch_size=4,
        block_size=16,
        max_steps=12,
        learning_rate=2e-3,
        seed=0,
        log_every=3,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        warmup_steps=3,
        lr_schedule="cosine",
        min_lr=2e-4,
        grad_clip=1.0,
        eval_every=4,
        eval_batches=2,
    )
    idx = data[:16].unsqueeze(0)

    torch.manual_seed(0)
    model_a = GPT(gcfg)
    evals_a = []
    losses_a = train(
        model_a,
        data,
        cfg,
        "cpu",
        val_data=val,
        on_eval=lambda step, v: evals_a.append((step, v)),
        log_path=tmp_path / "a.jsonl",
    )

    p = tmp_path / "ckpt.pt"
    torch.manual_seed(0)
    model_b = GPT(gcfg)
    evals_b = []
    l1 = train(
        model_b,
        data,
        cfg,
        "cpu",
        val_data=val,
        on_eval=lambda step, v: evals_b.append((step, v)),
        checkpoint_path=p,
        until_step=k,
        log_path=tmp_path / "b.jsonl",
    )
    assert load_checkpoint(p)["step"] == k

    torch.manual_seed(123)
    model_c = GPT(gcfg)
    l2 = train(
        model_c,
        data,
        cfg,
        "cpu",
        val_data=val,
        on_eval=lambda step, v: evals_b.append((step, v)),
        checkpoint_path=p,
        resume_from=p,
        log_path=tmp_path / "b.jsonl",
    )

    assert len(losses_a) == 12
    assert len(l1) == k
    assert len(l2) == 12 - k
    assert l1 + l2 == losses_a
    assert evals_b == evals_a
    assert read_log(tmp_path / "b.jsonl") == read_log(tmp_path / "a.jsonl")

    model_a.eval()
    model_c.eval()
    with torch.no_grad():
        la = model_a(idx)
        lc = model_c(idx)
    if isinstance(la, tuple):
        la, lc = la[0], lc[0]
    assert torch.allclose(la, lc, atol=1e-5)


# ---------------------------------------------------------------------------
# Bramka etapu 5 przez skrypt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("until", "ckpt_every"), [("10", "0"), ("7", "5"), ("1", "0")]
)
def test_gate_script_resume_equals_uninterrupted(
    main, d, tmp_path, capsys, until, ckpt_every
):
    a = tmp_path / "a"
    b = tmp_path / "b"
    main(bazowe(d, a) + ["--checkpoint-every", ckpt_every, "--until-step", until])
    out1 = capsys.readouterr().out
    r2 = main(bazowe(d, a) + ["--checkpoint-every", ckpt_every, "--resume"])
    out2 = capsys.readouterr().out
    rb = main(bazowe(d, b) + ["--checkpoint-every", ckpt_every])
    out3 = capsys.readouterr().out

    assert r2 == rb
    assert read_log(a / "train_log.jsonl") == read_log(b / "train_log.jsonl")
    ca = load_checkpoint(a / "gpt.pt")
    cb = load_checkpoint(b / "gpt.pt")
    assert ca["step"] == 20
    for key, tensor in ca["model"].items():
        assert torch.equal(tensor, cb["model"][key]), key
    assert loss_and_val_lines(out1) + loss_and_val_lines(out2) == loss_and_val_lines(
        out3
    )

    if until == "10" and ckpt_every == "0":
        assert [s for s, _ in step_lines(out1)] == [0]
        assert [s for s, _, _ in val_lines(out1)] == [9]
        assert [s for s, _ in step_lines(out2)] == [10, 19]
        assert [s for s, _, _ in val_lines(out2)] == [19]


# ---------------------------------------------------------------------------
# Wczytanie skryptu
# ---------------------------------------------------------------------------


def test_load_script_prints_nothing(capsys):
    g = runpy.run_path(str(SCRIPT))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert callable(g["main"])
    assert callable(g["parse_args"])


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

FLOAT_FIELDS = ["lr", "min_lr", "weight_decay", "grad_clip"]
INT_FIELDS = [
    "steps",
    "batch_size",
    "block_size",
    "n_layer",
    "n_head",
    "d_model",
    "warmup",
    "eval_every",
    "eval_batches",
    "checkpoint_every",
    "log_every",
    "seed",
]


def check_types(args):
    for f in FLOAT_FIELDS:
        assert type(getattr(args, f)) is float, f
    for f in INT_FIELDS:
        assert type(getattr(args, f)) is int, f
    assert isinstance(args.data_dir, Path)
    assert isinstance(args.out_dir, Path)


def test_parse_args_defaults(parse_args):
    a = parse_args([])
    assert a.data_dir == Path("data/tokens")
    assert a.out_dir == Path("data/checkpoints")
    assert a.steps == 5000
    assert a.batch_size == 64
    assert a.block_size == 256
    assert a.n_layer == 4
    assert a.n_head == 4
    assert a.d_model == 128
    assert a.lr == 0.002
    assert a.min_lr == 0.0002
    assert a.warmup == 200
    assert a.weight_decay == 0.1
    assert a.grad_clip == 1.0
    assert a.eval_every == 250
    assert a.eval_batches == 20
    assert a.checkpoint_every == 500
    assert a.log_every == 100
    assert a.seed == 0
    assert a.device == "cuda"
    assert a.no_bf16 is False
    assert a.resume is False
    assert a.until_step is None
    check_types(a)


def test_parse_args_device_and_resume(parse_args):
    a = parse_args(["--device", "cpu", "--resume"])
    assert a.device == "cpu"
    assert a.resume is True
    assert a.steps == 5000
    assert a.no_bf16 is False
    assert a.until_step is None


def test_parse_args_all_options(parse_args):
    a = parse_args(
        [
            "--data-dir", "d",
            "--out-dir", "o",
            "--steps", "7",
            "--batch-size", "3",
            "--block-size", "5",
            "--n-layer", "2",
            "--n-head", "8",
            "--d-model", "64",
            "--lr", "0.5",
            "--min-lr", "0.05",
            "--warmup", "11",
            "--weight-decay", "0.2",
            "--grad-clip", "0.7",
            "--eval-every", "13",
            "--eval-batches", "6",
            "--checkpoint-every", "17",
            "--log-every", "19",
            "--seed", "23",
            "--device", "cpu",
            "--no-bf16",
            "--resume",
            "--until-step", "29",
        ]
    )  # fmt: skip
    assert a.data_dir == Path("d")
    assert a.out_dir == Path("o")
    assert a.steps == 7
    assert a.batch_size == 3
    assert a.block_size == 5
    assert a.n_layer == 2
    assert a.n_head == 8
    assert a.d_model == 64
    assert a.lr == 0.5
    assert a.min_lr == 0.05
    assert a.warmup == 11
    assert a.weight_decay == 0.2
    assert a.grad_clip == 0.7
    assert a.eval_every == 13
    assert a.eval_batches == 6
    assert a.checkpoint_every == 17
    assert a.log_every == 19
    assert a.seed == 23
    assert a.device == "cpu"
    assert a.no_bf16 is True
    assert a.resume is True
    assert a.until_step == 29
    assert type(a.until_step) is int
    check_types(a)


@pytest.mark.parametrize("option", ["--steps", "--until-step"])
def test_parse_args_bad_int(parse_args, option):
    with pytest.raises(SystemExit) as exc:
        parse_args([option, "abc"])
    assert exc.value.code != 0


def test_parse_args_help_lists_options(parse_args, capsys):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for option in ALL_OPTIONS:
        assert option in out, option


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_example_output(main, d, tmp_path, capsys):
    main(bazowe(d, tmp_path / "out"))
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 10
    assert lines[0] == DATA_LINE
    assert lines[1] == MODEL_LINE
    assert STEP_RE.match(lines[2]) and lines[2].startswith("krok     0  strata ")
    assert VAL_RE.match(lines[3]) and lines[3].startswith("krok     9  val ")
    assert STEP_RE.match(lines[4]) and lines[4].startswith("krok    10  strata ")
    assert STEP_RE.match(lines[5]) and lines[5].startswith("krok    19  strata ")
    assert VAL_RE.match(lines[6]) and lines[6].startswith("krok    19  val ")
    assert FINAL_RE.match(lines[7])
    assert lines[8] == f"checkpoint -> {tmp_path / 'out' / 'gpt.pt'}"
    assert TIME_RE.match(lines[9])
    assert int(STEP_RE.match(lines[2]).group(1)) == 0
    assert int(VAL_RE.match(lines[3]).group(1)) == 9
    assert int(STEP_RE.match(lines[4]).group(1)) == 10
    assert int(STEP_RE.match(lines[5]).group(1)) == 19
    assert int(VAL_RE.match(lines[6]).group(1)) == 19


def test_main_example_return_and_files(main, d, tmp_path, capsys):
    out_dir = tmp_path / "out"
    f, b = main(bazowe(d, out_dir))
    out = capsys.readouterr().out
    assert type(f) is float
    assert type(b) is float
    assert b / f == pytest.approx(RATIO, rel=1e-9)
    assert b == pytest.approx(bits_per_char(f, 600, 1200), rel=1e-12)
    assert sorted(p.name for p in out_dir.iterdir()) == ["gpt.pt", "train_log.jsonl"]
    assert out.splitlines()[-3] == f"val (calosc): strata {f:.4f}, bity/znak {b:.4f}"


def test_main_log_matches_output(main, d, tmp_path, capsys):
    out_dir = tmp_path / "out"
    main(bazowe(d, out_dir))
    out = capsys.readouterr().out
    records = read_log(out_dir / "train_log.jsonl")
    train_recs = [r for r in records if "train_loss" in r]
    val_recs = [r for r in records if "val_loss" in r]
    assert [r["step"] for r in train_recs] == [0, 10, 19]
    assert all(set(r) == {"step", "train_loss", "lr"} for r in train_recs)
    assert [r["step"] for r in val_recs] == [9, 19]
    assert all(set(r) == {"step", "val_loss"} for r in val_recs)
    assert len(records) == 5

    by_train = {r["step"]: r for r in train_recs}
    by_val = {r["step"]: r for r in val_recs}
    for step, text in step_lines(out):
        assert text == f"{by_train[step]['train_loss']:.4f}"
    for step, vtext, btext in val_lines(out):
        vl = by_val[step]["val_loss"]
        assert vtext == f"{vl:.4f}"
        assert btext == f"{bits_per_char(vl, 600, 1200):.4f}"

    assert by_train[0]["lr"] == pytest.approx(0.001, rel=1e-6)
    assert by_train[10]["lr"] == pytest.approx(0.0012562833599, rel=1e-6)
    assert by_train[19]["lr"] == pytest.approx(0.00021367302229, rel=1e-6)


def test_main_checkpoint_configs(main, d, tmp_path):
    main(bazowe(d, tmp_path / "out"))
    ck = load_checkpoint(tmp_path / "out" / "gpt.pt")
    assert ck["step"] == 20
    assert ck["model_config"] == {
        "vocab_size": 32,
        "block_size": 16,
        "n_layer": 1,
        "n_head": 2,
        "d_model": 16,
    }
    assert ck["config"] == {
        "batch_size": 4,
        "block_size": 16,
        "max_steps": 20,
        "learning_rate": 0.002,
        "seed": 0,
        "log_every": 10,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "warmup_steps": 2,
        "lr_schedule": "cosine",
        "min_lr": 0.0002,
        "grad_clip": 1.0,
        "bf16": True,
        "eval_every": 10,
        "eval_batches": 2,
        "checkpoint_every": 0,
    }


def test_main_checkpoint_configs_custom(main, d, tmp_path):
    main(
        bazowe(d, tmp_path / "out")
        + [
            "--lr", "0.01",
            "--min-lr", "0.001",
            "--weight-decay", "0.05",
            "--grad-clip", "0.5",
            "--seed", "3",
            "--eval-batches", "3",
            "--checkpoint-every", "7",
            "--n-layer", "2",
            "--n-head", "4",
            "--no-bf16",
        ]
    )  # fmt: skip
    ck = load_checkpoint(tmp_path / "out" / "gpt.pt")
    c = ck["config"]
    assert c["learning_rate"] == 0.01
    assert c["min_lr"] == 0.001
    assert c["weight_decay"] == 0.05
    assert c["grad_clip"] == 0.5
    assert c["seed"] == 3
    assert c["eval_batches"] == 3
    assert c["checkpoint_every"] == 7
    assert c["bf16"] is False
    assert c["lr_schedule"] == "cosine"
    assert c["beta1"] == 0.9
    assert c["beta2"] == 0.95
    assert ck["model_config"]["n_layer"] == 2
    assert ck["model_config"]["n_head"] == 4


def test_main_returned_loss_is_full_val(main, d, tmp_path):
    f, _ = main(bazowe(d, tmp_path / "out"))
    m = load_gpt(tmp_path / "out" / "gpt.pt", "cpu")
    e = estimate_loss(m, load_tokens(d / "val.bin"), 4, 16, "cpu")
    assert f == pytest.approx(e, abs=1e-6)


def test_main_initial_loss_near_ln_vocab(main, d, tmp_path, capsys):
    main(bazowe(d, tmp_path / "out"))
    steps = step_lines(capsys.readouterr().out)
    first = float(dict(steps)[0])
    assert 0.9 * LN32 <= first <= 1.1 * LN32


def test_main_loss_decreases(main, d, tmp_path, capsys):
    f, _ = main(
        bazowe(d, tmp_path / "out")
        + ["--steps", "100", "--lr", "0.01", "--min-lr", "0.001",
           "--log-every", "50", "--eval-every", "50"]
    )  # fmt: skip
    steps = step_lines(capsys.readouterr().out)
    assert [s for s, _ in steps] == [0, 50, 99]
    by = dict(steps)
    assert float(by[99]) < float(by[0])
    assert f < 1.0


def test_main_reproducible(main, d, tmp_path, capsys):
    r1 = main(bazowe(d, tmp_path / "a"))
    out1 = capsys.readouterr().out
    r2 = main(bazowe(d, tmp_path / "b"))
    out2 = capsys.readouterr().out
    assert r1 == r2
    assert (tmp_path / "a" / "train_log.jsonl").read_text(encoding="utf-8") == (
        tmp_path / "b" / "train_log.jsonl"
    ).read_text(encoding="utf-8")
    assert loss_and_val_lines(out1) == loss_and_val_lines(out2)


def test_main_steps_zero(main, d, tmp_path, capsys):
    f, b = main(bazowe(d, tmp_path / "out") + ["--steps", "0"])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 5
    assert lines[0] == DATA_LINE
    assert lines[1] == MODEL_LINE
    assert FINAL_RE.match(lines[2])
    assert lines[3] == f"checkpoint -> {tmp_path / 'out' / 'gpt.pt'}"
    assert TIME_RE.match(lines[4])
    assert 0.9 * LN32 <= f <= 1.1 * LN32
    assert b == pytest.approx(f * RATIO, rel=1e-9)
    assert not (tmp_path / "out" / "gpt.pt").exists()


def test_main_seed_sets_initial_weights(main, d, tmp_path):
    r0a = main(bazowe(d, tmp_path / "a") + ["--steps", "0", "--seed", "0"])
    r0b = main(bazowe(d, tmp_path / "b") + ["--steps", "0", "--seed", "0"])
    r1 = main(bazowe(d, tmp_path / "c") + ["--steps", "0", "--seed", "1"])
    assert r0a == r0b
    assert r1[0] != r0a[0]


def test_main_seed_changes_result(main, d, tmp_path):
    r0 = main(bazowe(d, tmp_path / "a"))
    r1 = main(bazowe(d, tmp_path / "b") + ["--seed", "1"])
    assert r0[0] != r1[0]


def test_main_until_step_stops(main, d, tmp_path, capsys):
    main(bazowe(d, tmp_path / "out") + ["--until-step", "10"])
    out = capsys.readouterr().out
    assert [s for s, _ in step_lines(out)] == [0]
    assert [s for s, _, _ in val_lines(out)] == [9]
    ck = load_checkpoint(tmp_path / "out" / "gpt.pt")
    assert ck["step"] == 10
    assert ck["config"]["max_steps"] == 20
    lines = out.splitlines()
    assert any(FINAL_RE.match(line) for line in lines)
    assert any(TIME_RE.match(line) for line in lines)


def test_main_until_step_beyond_steps(main, d, tmp_path, capsys):
    r = main(bazowe(d, tmp_path / "a") + ["--until-step", "50"])
    out = capsys.readouterr().out
    rb = main(bazowe(d, tmp_path / "b"))
    assert [s for s, _ in step_lines(out)] == [0, 10, 19]
    assert load_checkpoint(tmp_path / "a" / "gpt.pt")["step"] == 20
    assert r == rb


def test_main_polish_out_dir(main, d, tmp_path, capsys):
    out_dir = tmp_path / "wyjście żółw"
    main(bazowe(d, out_dir))
    lines = capsys.readouterr().out.splitlines()
    assert lines[8] == f"checkpoint -> {out_dir / 'gpt.pt'}"
    assert (out_dir / "gpt.pt").exists()


# ---------------------------------------------------------------------------
# Błędy
# ---------------------------------------------------------------------------


def test_main_resume_without_checkpoint(main, d, tmp_path):
    with pytest.raises(FileNotFoundError):
        main(bazowe(d, tmp_path / "out") + ["--resume"])


def test_main_missing_meta(main, tmp_path):
    pusty = tmp_path / "pusty"
    pusty.mkdir()
    with pytest.raises(FileNotFoundError):
        main(bazowe(pusty, tmp_path / "out"))


def test_main_missing_val_bin(main, d, tmp_path):
    (d / "val.bin").unlink()
    with pytest.raises(FileNotFoundError):
        main(bazowe(d, tmp_path / "out"))


@pytest.mark.parametrize(
    "para",
    [
        ["--batch-size", "0"],
        ["--block-size", "0"],
        ["--n-layer", "0"],
        ["--d-model", "0"],
        ["--n-head", "3"],
        ["--lr", "0"],
        ["--min-lr", "0.01"],
        ["--warmup", "-1"],
        ["--weight-decay", "-0.1"],
        ["--grad-clip", "-1"],
        ["--eval-every", "-1"],
        ["--eval-batches", "0"],
        ["--checkpoint-every", "-1"],
        ["--log-every", "0"],
        ["--steps", "-1"],
        ["--until-step", "-1"],
    ],
    ids=lambda p: " ".join(p),
)
def test_main_bad_values(main, d, tmp_path, para):
    with pytest.raises(ValueError):
        main(bazowe(d, tmp_path / "out") + para)


# ---------------------------------------------------------------------------
# Uruchomienie jako program
# ---------------------------------------------------------------------------


def test_script_utf8_with_redirected_stdout(d, tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONUTF8", None)
    env.pop("PYTHONIOENCODING", None)
    env["PYTHONIOENCODING"] = "ascii"
    out = tmp_path / "wyjście"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)]
        + bazowe(d, out)
        + ["--steps", "4", "--log-every", "2", "--eval-every", "2"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=180,
    )
    stderr = proc.stderr.decode("utf-8", errors="replace")
    assert proc.returncode == 0, stderr
    lines = proc.stdout.decode("utf-8").splitlines()
    assert f"checkpoint -> {out / 'gpt.pt'}" in lines, stderr
    assert TIME_RE.match(lines[-1]), stderr


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="brak GPU")
@pytest.mark.parametrize("opcje", [[], ["--no-bf16"]], ids=["bf16", "no-bf16"])
def test_main_cuda(main, d, tmp_path, capsys, opcje):
    f, b = main(
        bazowe(d, tmp_path / "out")
        + ["--device", "cuda", "--steps", "10", "--eval-every", "5"]
        + opcje
    )
    out = capsys.readouterr().out
    assert math.isfinite(f)
    assert math.isfinite(b)
    assert "model: 4080 parametrow, urzadzenie cuda" in out.splitlines()
    m = load_gpt(tmp_path / "out" / "gpt.pt", "cpu")
    assert m.config.vocab_size == 32
