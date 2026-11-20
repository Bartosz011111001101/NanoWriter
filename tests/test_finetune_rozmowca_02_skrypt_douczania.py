"""Testy skryptu scripts/finetune_rozmowca.py oraz bramki zadania 09 (douczanie rozmówcy)."""

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

from nanowriter.dataset import (
    TokenMeta,
    load_mask,
    load_tokens,
    save_meta,
    write_mask,
    write_tokens,
)
from nanowriter.model import GPT, GPTConfig
from nanowriter.train import estimate_loss, load_checkpoint, load_gpt

REPO = Path(__file__).resolve().parents[1]
FINETUNE = REPO / "scripts" / "finetune_rozmowca.py"
PRETRAIN = REPO / "scripts" / "train_gpt.py"

LN8 = math.log(8)
DATA_LINE = "dane: train 3000 tokenow (maska 1500), val 600 tokenow (maska 300), slownik 8"
MODEL_RE = re.compile(r"^model: (\d+) parametrow z (.+), urzadzenie cpu, kontekst 16$")
BEFORE_RE = re.compile(r"^val przed douczaniem: strata (\d+\.\d{4})$")
STEP_RE = re.compile(r"^krok +(\d+)  strata (\d+\.\d{4})$")
VAL_RE = re.compile(r"^krok +(\d+)  val (\d+\.\d{4})  perplexity (\d+\.\d{2})$")
AFTER_RE = re.compile(r"^val po douczaniu: strata (\d+\.\d{4}), perplexity (\d+\.\d{2})$")
TIME_RE = re.compile(r"^czas: \d+\.\d s$")

ALL_OPTIONS = [
    "--init-from",
    "--data-dir",
    "--out-dir",
    "--steps",
    "--batch-size",
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

PRETRAIN_CONFIG = {"vocab_size": 8, "block_size": 16, "n_layer": 2, "n_head": 2, "d_model": 16}


def build_dialogs(dlg, vocab=8):
    """Buduje katalog sztucznych rozmów ze słownikiem vocab."""
    write_tokens([i % 8 for i in range(3000)], dlg / "train.bin")
    write_mask([int(i % 8 >= 4) for i in range(3000)], dlg / "train_mask.bin")
    write_tokens([i % 8 for i in range(600)], dlg / "val.bin")
    write_mask([int(i % 8 >= 4) for i in range(600)], dlg / "val_mask.bin")
    save_meta(TokenMeta(vocab, "uint16", 3000, 3000, 600, 600), dlg / "meta.json")
    return dlg


def build_pretrain(pre):
    """Buduje katalog sztucznego treningu wstępnego (losowe tokeny)."""
    g = torch.Generator().manual_seed(1)
    write_tokens(torch.randint(0, 8, (3000,), generator=g).tolist(), pre / "train.bin")
    write_tokens(torch.randint(0, 8, (600,), generator=g).tolist(), pre / "val.bin")
    save_meta(TokenMeta(8, "uint16", 3000, 6000, 600, 1200), pre / "meta.json")
    return pre


def bazowe(init, dlg, out):
    return [
        "--init-from", str(init),
        "--data-dir", str(dlg),
        "--out-dir", str(out),
        "--device", "cpu",
        "--steps", "60",
        "--batch-size", "8",
        "--lr", "0.01",
        "--min-lr", "0.001",
        "--warmup", "5",
        "--eval-every", "30",
        "--eval-batches", "2",
        "--log-every", "30",
        "--checkpoint-every", "0",
    ]  # fmt: skip


def read_log(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def loss_lines(out):
    """Linie straty i walidacji (`krok ...`) w kolejności wypisania."""
    return [ln for ln in out.splitlines() if STEP_RE.match(ln) or VAL_RE.match(ln)]


def step_numbers(out):
    return [int(STEP_RE.match(ln).group(1)) for ln in out.splitlines() if STEP_RE.match(ln)]


def val_numbers(out):
    return [int(VAL_RE.match(ln).group(1)) for ln in out.splitlines() if VAL_RE.match(ln)]


def tree(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def snapshot_data():
    snap = {}
    for sub in ("checkpoints", "tokens", "rozmowca"):
        base = REPO / "data" / sub
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file():
                snap[p.relative_to(REPO).as_posix()] = p.stat().st_size
    return snap


def same_state(path_a, path_b):
    a = load_checkpoint(path_a)["model"]
    b = load_checkpoint(path_b)["model"]
    assert set(a) == set(b)
    return all(torch.equal(a[k], b[k]) for k in a)


@pytest.fixture(scope="module")
def finetune_funcs():
    g = runpy.run_path(str(FINETUNE))
    return g["parse_args"], g["main"]


@pytest.fixture(scope="module")
def pretrain_main():
    return runpy.run_path(str(PRETRAIN))["main"]


@pytest.fixture
def parse_args(finetune_funcs):
    return finetune_funcs[0]


@pytest.fixture
def main(finetune_funcs):
    return finetune_funcs[1]


@pytest.fixture(autouse=True)
def cwd(tmp_path, monkeypatch):
    d = tmp_path / "cwd"
    d.mkdir()
    monkeypatch.chdir(d)
    return d


@pytest.fixture(scope="module")
def pre_root(tmp_path_factory, pretrain_main):
    root = tmp_path_factory.mktemp("wstepny")
    pre = build_pretrain(root / "pre")
    work = root / "cwd"
    work.mkdir()
    out_wstepny = root / "out_wstepny"
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(work)
        pretrain_main(
            [
                "--data-dir", str(pre),
                "--out-dir", str(out_wstepny),
                "--device", "cpu",
                "--steps", "20",
                "--batch-size", "4",
                "--block-size", "16",
                "--n-layer", "2",
                "--n-head", "2",
                "--d-model", "16",
                "--lr", "0.002",
                "--min-lr", "0.0002",
                "--warmup", "2",
                "--eval-every", "10",
                "--eval-batches", "2",
                "--log-every", "10",
                "--checkpoint-every", "0",
            ]  # fmt: skip
        )
    return root


@pytest.fixture(scope="module")
def INIT(pre_root):
    return pre_root / "out_wstepny" / "gpt.pt"


@pytest.fixture
def dlg(tmp_path):
    return build_dialogs(tmp_path / "dlg")


@pytest.fixture
def out(tmp_path):
    return tmp_path / "out_rozmowy"


# ---------------------------------------------------------------------------
# Bramka zadania 09
# ---------------------------------------------------------------------------


def test_gate_target_config_10m():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=8192, block_size=256, n_layer=6, n_head=5, d_model=320))
    assert sum(p.numel() for p in model.parameters()) == 10_101_760
    assert model.lm_head.weight is model.tok_emb.weight
    model.eval()
    with torch.no_grad():
        assert model(torch.randint(0, 8192, (1, 256))).shape == (1, 256, 8192)


def test_gate_pretrain_checkpoint(INIT):
    assert INIT.exists()
    ck = load_checkpoint(INIT)
    assert ck["step"] == 20
    assert ck["model_config"] == PRETRAIN_CONFIG


def test_gate_finetune_from_checkpoint(main, INIT, dlg, out):
    h = INIT.read_bytes()
    pliki = sorted(p.name for p in INIT.parent.iterdir())
    assert pliki == ["gpt.pt", "train_log.jsonl"]

    przed, po = main(bazowe(INIT, dlg, out))

    assert INIT.read_bytes() == h
    assert sorted(p.name for p in INIT.parent.iterdir()) == pliki
    assert (out / "gpt.pt").exists()
    assert sorted(p.name for p in out.iterdir()) == ["gpt.pt", "train_log.jsonl"]
    ck = load_checkpoint(out / "gpt.pt")
    assert ck["step"] == 60
    assert ck["model_config"] == load_checkpoint(INIT)["model_config"]
    model = load_gpt(out / "gpt.pt", "cpu")
    assert model.config.vocab_size == 8
    assert model.config.block_size == 16
    assert isinstance(przed, float) and isinstance(po, float)
    assert math.isfinite(przed) and math.isfinite(po)
    assert po < przed
    start = load_checkpoint(INIT)["model"]
    assert any(not torch.equal(v, start[k]) for k, v in ck["model"].items())


def test_gate_loss_masked_only(main, INIT, dlg, out):
    przed, po = main(bazowe(INIT, dlg, out))
    m0 = load_gpt(INIT, "cpu")
    val = load_tokens(dlg / "val.bin")
    vm = load_mask(dlg / "val_mask.bin")

    assert przed == pytest.approx(estimate_loss(m0, val, 8, 16, "cpu", mask=vm), abs=1e-6)
    assert abs(przed - estimate_loss(m0, val, 8, 16, "cpu")) > 1e-6
    assert abs(przed - estimate_loss(m0, val, 8, 16, "cpu", mask=~vm)) > 1e-6
    po_ref = estimate_loss(load_gpt(out / "gpt.pt", "cpu"), val, 8, 16, "cpu", mask=vm)
    assert po == pytest.approx(po_ref, abs=1e-6)


@pytest.mark.parametrize(
    ("until", "ckpt_every"), [("30", "0"), ("25", "10"), ("1", "0")]
)
def test_gate_resume_equals_uninterrupted(main, INIT, dlg, tmp_path, capsys, until, ckpt_every):
    steps = "20" if until == "1" else "60"
    opcje = [
        "--steps", steps,
        "--warmup", "3",
        "--eval-every", "10",
        "--log-every", "5",
        "--checkpoint-every", ckpt_every,
    ]  # fmt: skip
    a = tmp_path / "a"
    b = tmp_path / "b"
    h = INIT.read_bytes()

    main(bazowe(INIT, dlg, a) + opcje + ["--until-step", until])
    out1 = capsys.readouterr().out
    r2 = main(bazowe(INIT, dlg, a) + opcje + ["--resume"])
    out2 = capsys.readouterr().out
    rb = main(bazowe(INIT, dlg, b) + opcje)
    out3 = capsys.readouterr().out

    assert r2 == rb
    assert read_log(a / "train_log.jsonl") == read_log(b / "train_log.jsonl")
    assert load_checkpoint(a / "gpt.pt")["step"] == int(steps)
    assert same_state(a / "gpt.pt", b / "gpt.pt")
    assert loss_lines(out1) + loss_lines(out2) == loss_lines(out3)
    assert INIT.read_bytes() == h


def test_gate_separate_folders(main, INIT, dlg, out, tmp_path):
    snap = snapshot_data()
    drzewo = tree(tmp_path)

    main(bazowe(INIT, dlg, out))

    assert snapshot_data() == snap
    assert tree(tmp_path) == sorted(
        drzewo + ["out_rozmowy", "out_rozmowy/gpt.pt", "out_rozmowy/train_log.jsonl"]
    )
    assert list((tmp_path / "cwd").iterdir()) == []


# ---------------------------------------------------------------------------
# Wczytanie skryptu
# ---------------------------------------------------------------------------


def test_load_script_prints_nothing(capsys):
    g = runpy.run_path(str(FINETUNE))
    cap = capsys.readouterr()
    assert cap.out == ""
    assert cap.err == ""
    assert callable(g["main"])
    assert callable(g["parse_args"])


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_defaults(parse_args):
    a = parse_args([])
    assert a.init_from == Path("data/rozmowca/ckpt-wstepny/gpt.pt")
    assert a.data_dir == Path("data/rozmowca/tokeny-rozmowy")
    assert a.out_dir == Path("data/rozmowca/ckpt-rozmowy")
    for p in (a.init_from, a.data_dir, a.out_dir):
        assert isinstance(p, Path)
    assert a.steps == 300
    assert a.batch_size == 16
    assert a.lr == 0.0003
    assert a.min_lr == 0.00003
    assert a.warmup == 20
    assert a.weight_decay == 0.1
    assert a.grad_clip == 1.0
    assert a.eval_every == 25
    assert a.eval_batches == 20
    assert a.checkpoint_every == 50
    assert a.log_every == 10
    assert a.seed == 0
    assert a.device == "cuda"
    assert a.no_bf16 is False
    assert a.resume is False
    assert a.until_step is None
    for f in (a.lr, a.min_lr, a.weight_decay, a.grad_clip):
        assert type(f) is float
    for i in (a.steps, a.batch_size, a.warmup, a.eval_every, a.eval_batches,
              a.checkpoint_every, a.log_every, a.seed):  # fmt: skip
        assert type(i) is int


def test_parse_args_device_and_resume(parse_args):
    a = parse_args(["--device", "cpu", "--resume"])
    assert a.device == "cpu"
    assert a.resume is True
    assert a.steps == 300
    assert a.no_bf16 is False
    assert a.until_step is None


def test_parse_args_all_options(parse_args):
    a = parse_args(
        [
            "--init-from", "i.pt", "--data-dir", "d", "--out-dir", "o",
            "--steps", "7", "--batch-size", "3", "--lr", "0.5", "--min-lr", "0.05",
            "--warmup", "11", "--weight-decay", "0.2", "--grad-clip", "0.7",
            "--eval-every", "13", "--eval-batches", "6", "--checkpoint-every", "17",
            "--log-every", "19", "--seed", "23", "--device", "cpu", "--no-bf16",
            "--resume", "--until-step", "29",
        ]  # fmt: skip
    )
    assert a.init_from == Path("i.pt")
    assert a.data_dir == Path("d")
    assert a.out_dir == Path("o")
    assert a.steps == 7
    assert a.batch_size == 3
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
    for p in (a.init_from, a.data_dir, a.out_dir):
        assert isinstance(p, Path)
    for f in (a.lr, a.min_lr, a.weight_decay, a.grad_clip):
        assert type(f) is float
    for i in (a.steps, a.batch_size, a.warmup, a.eval_every, a.eval_batches,
              a.checkpoint_every, a.log_every, a.seed, a.until_step):  # fmt: skip
        assert type(i) is int


@pytest.mark.parametrize("opcja", ["--steps", "--until-step"])
def test_parse_args_bad_int(parse_args, opcja):
    with pytest.raises(SystemExit) as exc:
        parse_args([opcja, "abc"])
    assert exc.value.code != 0


def test_parse_args_help_lists_options(parse_args, capsys):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    for opcja in ALL_OPTIONS:
        assert opcja in text


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_example_output(main, INIT, dlg, out, capsys):
    przed, po = main(bazowe(INIT, dlg, out))
    lines = capsys.readouterr().out.splitlines()

    assert len(lines) == 11
    assert lines[0] == DATA_LINE
    m = MODEL_RE.match(lines[1])
    assert m, lines[1]
    assert int(m.group(1)) == sum(p.numel() for p in load_gpt(INIT, "cpu").parameters())
    assert m.group(2) == str(INIT)
    m = BEFORE_RE.match(lines[2])
    assert m, lines[2]
    assert m.group(1) == f"{przed:.4f}"
    assert STEP_RE.match(lines[3]) and lines[3].startswith("krok     0  strata ")
    assert VAL_RE.match(lines[4]) and lines[4].startswith("krok    29  val ")
    assert STEP_RE.match(lines[5]) and int(STEP_RE.match(lines[5]).group(1)) == 30
    assert STEP_RE.match(lines[6]) and int(STEP_RE.match(lines[6]).group(1)) == 59
    assert VAL_RE.match(lines[7]) and int(VAL_RE.match(lines[7]).group(1)) == 59
    m = AFTER_RE.match(lines[8])
    assert m, lines[8]
    assert m.group(1) == f"{po:.4f}"
    assert m.group(2) == f"{math.exp(po):.2f}"
    assert lines[9] == f"checkpoint -> {out / 'gpt.pt'}"
    assert TIME_RE.match(lines[10]), lines[10]


def test_main_return_and_files(main, INIT, dlg, out):
    przed, po = main(bazowe(INIT, dlg, out))
    assert isinstance(przed, float) and isinstance(po, float)
    assert po < przed
    assert 0.5 * LN8 <= przed <= 1.5 * LN8
    assert po < 0.5


def test_main_log_matches_output(main, INIT, dlg, out, capsys):
    main(bazowe(INIT, dlg, out))
    text = capsys.readouterr().out
    records = read_log(out / "train_log.jsonl")
    train_recs = [r for r in records if "train_loss" in r]
    val_recs = [r for r in records if "val_loss" in r]

    assert [r["step"] for r in train_recs] == [0, 30, 59]
    assert all(set(r) == {"step", "train_loss", "lr"} for r in train_recs)
    assert [r["step"] for r in val_recs] == [29, 59]
    assert all(set(r) == {"step", "val_loss"} for r in val_recs)
    assert len(records) == 5

    by_step = {r["step"]: r for r in train_recs}
    for ln in text.splitlines():
        m = STEP_RE.match(ln)
        if m:
            assert m.group(2) == f"{by_step[int(m.group(1))]['train_loss']:.4f}"
    by_val = {r["step"]: r for r in val_recs}
    for ln in text.splitlines():
        m = VAL_RE.match(ln)
        if m:
            v = by_val[int(m.group(1))]["val_loss"]
            assert m.group(2) == f"{v:.4f}"
            assert m.group(3) == f"{math.exp(v):.2f}"

    assert by_step[0]["lr"] == pytest.approx(0.002)
    lr30 = 0.001 + 0.5 * (1 + math.cos(math.pi * 25 / 55)) * 0.009
    lr59 = 0.001 + 0.5 * (1 + math.cos(math.pi * 54 / 55)) * 0.009
    assert by_step[30]["lr"] == pytest.approx(lr30, rel=1e-6)
    assert by_step[59]["lr"] == pytest.approx(lr59, rel=1e-6)


def test_main_checkpoint_config(main, INIT, dlg, out):
    main(bazowe(INIT, dlg, out))
    ck = load_checkpoint(out / "gpt.pt")
    assert ck["step"] == 60
    assert ck["config"] == {
        "batch_size": 8,
        "block_size": 16,
        "max_steps": 60,
        "learning_rate": 0.01,
        "seed": 0,
        "log_every": 30,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "warmup_steps": 5,
        "lr_schedule": "cosine",
        "min_lr": 0.001,
        "grad_clip": 1.0,
        "bf16": True,
        "eval_every": 30,
        "eval_batches": 2,
        "checkpoint_every": 0,
    }


def test_main_checkpoint_config_custom(main, INIT, dlg, out):
    main(
        bazowe(INIT, dlg, out)
        + ["--weight-decay", "0.05", "--grad-clip", "0.5", "--seed", "3",
           "--eval-batches", "3", "--checkpoint-every", "7", "--no-bf16"]  # fmt: skip
    )
    cfg = load_checkpoint(out / "gpt.pt")["config"]
    assert cfg["weight_decay"] == 0.05
    assert cfg["grad_clip"] == 0.5
    assert cfg["seed"] == 3
    assert cfg["eval_batches"] == 3
    assert cfg["checkpoint_every"] == 7
    assert cfg["bf16"] is False
    assert cfg["lr_schedule"] == "cosine"
    assert cfg["beta1"] == 0.9
    assert cfg["beta2"] == 0.95
    assert cfg["block_size"] == 16


def test_main_reproducible(main, INIT, dlg, tmp_path, capsys):
    a = tmp_path / "a"
    b = tmp_path / "b"
    r1 = main(bazowe(INIT, dlg, a))
    out1 = capsys.readouterr().out
    r2 = main(bazowe(INIT, dlg, b))
    out2 = capsys.readouterr().out

    assert r1 == r2
    assert (a / "train_log.jsonl").read_text(encoding="utf-8") == (
        b / "train_log.jsonl"
    ).read_text(encoding="utf-8")
    assert loss_lines(out1) == loss_lines(out2)
    assert same_state(a / "gpt.pt", b / "gpt.pt")


def test_main_seed_changes_result(main, INIT, dlg, tmp_path):
    r0 = main(bazowe(INIT, dlg, tmp_path / "a"))
    r1 = main(bazowe(INIT, dlg, tmp_path / "b") + ["--seed", "1"])
    assert r0[0] == r1[0]
    assert r0[1] != r1[1]


def test_main_steps_zero(main, INIT, dlg, out, capsys):
    przed, po = main(bazowe(INIT, dlg, out) + ["--steps", "0"])
    lines = capsys.readouterr().out.splitlines()

    assert len(lines) == 6
    assert lines[0] == DATA_LINE
    assert MODEL_RE.match(lines[1]), lines[1]
    assert BEFORE_RE.match(lines[2]), lines[2]
    assert AFTER_RE.match(lines[3]), lines[3]
    assert lines[4] == f"checkpoint -> {out / 'gpt.pt'}"
    assert TIME_RE.match(lines[5]), lines[5]
    assert po == przed
    assert not (out / "gpt.pt").exists()


def test_main_until_step_stops(main, INIT, dlg, out, capsys):
    main(bazowe(INIT, dlg, out) + ["--until-step", "30"])
    text = capsys.readouterr().out
    lines = text.splitlines()

    assert step_numbers(text) == [0]
    assert val_numbers(text) == [29]
    ck = load_checkpoint(out / "gpt.pt")
    assert ck["step"] == 30
    assert ck["config"]["max_steps"] == 60
    assert any(AFTER_RE.match(ln) for ln in lines)
    assert f"checkpoint -> {out / 'gpt.pt'}" in lines
    assert any(TIME_RE.match(ln) for ln in lines)


def test_main_polish_out_dir(main, INIT, dlg, tmp_path, capsys):
    polish = tmp_path / "wyjście żółw"
    main(bazowe(INIT, dlg, polish))
    lines = capsys.readouterr().out.splitlines()
    assert lines[-2] == f"checkpoint -> {polish / 'gpt.pt'}"
    assert (polish / "gpt.pt").exists()


# ---------------------------------------------------------------------------
# Błędy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wariant", ["ta_sama_sciezka", "inna_sciezka"])
def test_main_out_dir_same_as_init(main, INIT, dlg, capsys, wariant):
    h = INIT.read_bytes()
    pliki = sorted(p.name for p in INIT.parent.iterdir())
    log = (INIT.parent / "train_log.jsonl").read_text(encoding="utf-8")
    if wariant == "ta_sama_sciezka":
        out_dir = INIT.parent
    else:
        out_dir = INIT.parent / ".." / INIT.parent.name

    with pytest.raises(ValueError):
        main(bazowe(INIT, dlg, out_dir))

    assert INIT.read_bytes() == h
    assert sorted(p.name for p in INIT.parent.iterdir()) == pliki
    assert (INIT.parent / "train_log.jsonl").read_text(encoding="utf-8") == log
    assert "dane:" not in capsys.readouterr().out


def test_main_vocab_mismatch(main, INIT, tmp_path, out):
    dlg16 = build_dialogs(tmp_path / "dlg16", vocab=16)
    with pytest.raises(ValueError) as exc:
        main(bazowe(INIT, dlg16, out))
    msg = str(exc.value)
    assert "8" in msg
    assert "16" in msg
    assert not out.exists()


def test_main_resume_without_checkpoint(main, INIT, dlg, out):
    assert not out.exists()
    with pytest.raises(FileNotFoundError):
        main(bazowe(INIT, dlg, out) + ["--resume"])


@pytest.mark.parametrize(
    "brak",
    ["init", "meta.json", "train.bin", "train_mask.bin", "val.bin", "val_mask.bin"],
)
def test_main_missing_inputs(main, INIT, dlg, out, tmp_path, brak):
    h = INIT.read_bytes()
    init = INIT
    if brak == "init":
        init = tmp_path / "nie_ma.pt"
    else:
        (dlg / brak).unlink()
    with pytest.raises(FileNotFoundError):
        main(bazowe(init, dlg, out))
    assert INIT.read_bytes() == h


@pytest.mark.parametrize(
    "para",
    [
        ["--batch-size", "0"],
        ["--lr", "0"],
        ["--min-lr", "0.5"],
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
def test_main_bad_values(main, INIT, dlg, out, para):
    h = INIT.read_bytes()
    with pytest.raises(ValueError):
        main(bazowe(INIT, dlg, out) + para)
    assert INIT.read_bytes() == h


# ---------------------------------------------------------------------------
# Uruchomienie jako program
# ---------------------------------------------------------------------------


def test_script_utf8_with_redirected_stdout(INIT, dlg, tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
    env["PYTHONIOENCODING"] = "ascii"
    out_dir = tmp_path / "wyjście"
    res = subprocess.run(
        [sys.executable, str(FINETUNE)]
        + bazowe(INIT, dlg, out_dir)
        + ["--steps", "4", "--log-every", "2", "--eval-every", "2"],
        cwd=tmp_path / "cwd",
        env=env,
        capture_output=True,
        timeout=180,
    )
    err = res.stderr.decode("utf-8", errors="replace")
    assert res.returncode == 0, err
    lines = res.stdout.decode("utf-8").splitlines()
    assert f"checkpoint -> {out_dir / 'gpt.pt'}" in lines, err
    assert TIME_RE.match(lines[-1]), err


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="brak GPU")
@pytest.mark.parametrize("opcje", [[], ["--no-bf16"]], ids=["bf16", "fp32"])
def test_main_cuda(main, INIT, dlg, out, capsys, opcje):
    przed, po = main(
        bazowe(INIT, dlg, out)
        + ["--device", "cuda", "--steps", "10", "--eval-every", "5", "--log-every", "5"]
        + opcje
    )
    assert math.isfinite(przed)
    assert math.isfinite(po)
    assert "urzadzenie cuda, kontekst 16" in capsys.readouterr().out
    assert load_gpt(out / "gpt.pt", "cpu").config.vocab_size == 8
