"""Testy skryptu scripts/train_bigram.py oraz bramki etapu 1."""

import math
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from nanowriter.model import BigramLanguageModel
from nanowriter.sample import generate
from nanowriter.tokenizer import CharTokenizer
from nanowriter.train import TrainConfig, train

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_bigram.py"

ALA_TEXT = "ala ma kota a kot ma ale " * 40
PL_TEXT = "żółw żółty źle żuje. " * 40
ALA_CHARS = {" ", "a", "e", "k", "l", "m", "o", "t"}
ALA_STATS = "znaki: 1000, slownik: 8, parametry: 64"
STEP_RE = re.compile(r"^krok +(\d+)  strata (\d+\.\d{4})$")


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
def plik_ala(tmp_path):
    p = tmp_path / "ala.txt"
    p.write_text(ALA_TEXT, encoding="utf-8")
    return p


@pytest.fixture
def plik_polski(tmp_path):
    p = tmp_path / "pl.txt"
    p.write_text(PL_TEXT, encoding="utf-8")
    return p


@pytest.fixture
def base_args(plik_ala):
    return [
        "--text", str(plik_ala), "--steps", "50", "--log-every", "25",
        "--prompt", "ala", "--length", "30",
    ]  # fmt: skip


def step_lines(out):
    """Zwraca listę (krok, strata) z linii zaczynających się od `krok`."""
    result = []
    for line in out.splitlines():
        if line.startswith("krok"):
            m = STEP_RE.match(line)
            assert m is not None, f"zły format linii: {line!r}"
            result.append((int(m.group(1)), float(m.group(2))))
    return result


def line_after_marker(out):
    lines = out.splitlines()
    i = lines.index("--- tekst ---")
    return lines[i + 1]


# --- Bramka etapu 1 ---


@pytest.fixture(scope="module")
def gate():
    text = "Zażółć gęślą jaźń. Ala ma kota, a kot ma Alę.\n" * 30
    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.int64)
    torch.manual_seed(0)
    model = BigramLanguageModel(tokenizer.vocab_size)
    losses = train(model, data, TrainConfig(max_steps=200, log_every=50), "cpu")
    model.eval()
    idx = torch.tensor([tokenizer.encode("Ala")], dtype=torch.int64)
    return tokenizer, model, losses, idx


def test_gate_loss_decreases(gate):
    torch.manual_seed(0)
    _, _, losses, _ = gate
    assert len(losses) == 200
    assert sum(losses[-20:]) / 20 < sum(losses[:20]) / 20
    assert 0.9 * math.log(23) <= losses[0] <= 1.1 * math.log(23)


def test_gate_generate_length(gate):
    torch.manual_seed(0)
    tokenizer, model, _, idx = gate
    out = generate(model, idx, 50, generator=torch.Generator().manual_seed(0))
    s = tokenizer.decode(out[0].tolist())
    assert out.shape == (1, 53)
    assert len(s) == 53
    assert s.startswith("Ala")
    assert all(c in tokenizer.chars for c in s)


@pytest.mark.parametrize("max_new_tokens", [0, 1, 100])
def test_gate_generate_length_various(gate, max_new_tokens):
    torch.manual_seed(0)
    tokenizer, model, _, idx = gate
    out = generate(model, idx, max_new_tokens, generator=torch.Generator().manual_seed(0))
    s = tokenizer.decode(out[0].tolist())
    assert len(s) == 3 + max_new_tokens
    assert s.startswith("Ala")


# --- Wczytanie skryptu ---


def test_load_script_prints_nothing(capsys):
    g = runpy.run_path(str(SCRIPT))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert callable(g["main"])
    assert callable(g["parse_args"])


# --- parse_args ---


def test_parse_args_defaults(parse_args):
    a = parse_args([])
    assert a.slug == "lalka-tom-pierwszy"
    assert a.text is None
    assert a.data_dir == Path("data/raw")
    assert isinstance(a.data_dir, Path)
    assert a.steps == 5000
    assert a.batch_size == 32
    assert a.block_size == 8
    assert a.lr == 0.01
    assert isinstance(a.lr, float)
    assert a.seed == 0
    assert a.log_every == 500
    assert a.prompt == "Wokulski"
    assert a.length == 500
    assert a.device == "cpu"


def test_parse_args_text_and_steps(parse_args):
    a = parse_args(["--text", "a.txt", "--steps", "50"])
    assert a.text == Path("a.txt")
    assert isinstance(a.text, Path)
    assert a.steps == 50
    assert isinstance(a.steps, int)
    assert a.batch_size == 32
    assert a.prompt == "Wokulski"


def test_parse_args_all_options(parse_args):
    a = parse_args([
        "--slug", "abc", "--text", "t.txt", "--data-dir", "d", "--steps", "7",
        "--batch-size", "4", "--block-size", "3", "--lr", "0.5", "--seed", "9",
        "--log-every", "2", "--prompt", "xy", "--length", "11", "--device", "cuda",
    ])  # fmt: skip
    assert a.slug == "abc"
    assert a.text == Path("t.txt")
    assert a.data_dir == Path("d")
    assert a.steps == 7
    assert a.batch_size == 4
    assert a.block_size == 3
    assert a.lr == 0.5
    assert a.seed == 9
    assert a.log_every == 2
    assert a.prompt == "xy"
    assert a.length == 11
    assert a.device == "cuda"
    for name in ("steps", "batch_size", "block_size", "seed", "log_every", "length"):
        assert isinstance(getattr(a, name), int)
    assert isinstance(a.lr, float)
    assert isinstance(a.text, Path)
    assert isinstance(a.data_dir, Path)


def test_parse_args_bad_int(parse_args):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--steps", "abc"])
    assert exc.value.code != 0


def test_parse_args_help_lists_options(parse_args, capsys):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for opt in (
        "--slug", "--text", "--data-dir", "--steps", "--batch-size", "--block-size",
        "--lr", "--seed", "--log-every", "--prompt", "--length", "--device",
    ):  # fmt: skip
        assert opt in out


# --- main ---


def test_main_example(main, base_args, capsys):
    result = main(base_args)
    out = capsys.readouterr().out
    assert isinstance(result, str)
    assert len(result) == 33
    assert result.startswith("ala")
    assert set(result) <= ALA_CHARS
    lines = out.splitlines()
    assert ALA_STATS in lines
    assert [s for s, _ in step_lines(out)] == [0, 25, 49]
    assert "--- tekst ---" in lines
    assert line_after_marker(out) == result
    first_step = next(i for i, ln in enumerate(lines) if ln.startswith("krok"))
    last_step = max(i for i, ln in enumerate(lines) if ln.startswith("krok"))
    assert lines.index(ALA_STATS) < first_step
    assert last_step < lines.index("--- tekst ---")


def test_main_initial_loss_near_ln_vocab(main, base_args, capsys):
    main(base_args)
    steps = step_lines(capsys.readouterr().out)
    assert steps[0][0] == 0
    assert 0.9 * math.log(8) <= steps[0][1] <= 1.1 * math.log(8)


def test_main_loss_decreases(main, plik_ala, capsys):
    main([
        "--text", str(plik_ala), "--steps", "200", "--log-every", "50",
        "--prompt", "ala", "--length", "10",
    ])  # fmt: skip
    steps = step_lines(capsys.readouterr().out)
    assert [s for s, _ in steps] == [0, 50, 100, 150, 199]
    assert steps[-1][1] < steps[0][1]


def test_main_reproducible(main, base_args, capsys):
    r1 = main(base_args)
    out1 = capsys.readouterr().out
    r2 = main(base_args)
    out2 = capsys.readouterr().out
    assert r1 == r2
    assert step_lines(out1) == step_lines(out2)


def test_main_polish_text(main, plik_polski, capsys):
    result = main([
        "--text", str(plik_polski), "--steps", "50", "--log-every", "25",
        "--prompt", "żó", "--length", "20",
    ])  # fmt: skip
    out = capsys.readouterr().out
    assert len(result) == 22
    assert result.startswith("żó")
    assert set(result) <= set(PL_TEXT)
    assert line_after_marker(out) == result


def test_main_steps_zero(main, plik_ala, capsys):
    result = main(["--text", str(plik_ala), "--steps", "0", "--prompt", "ala", "--length", "30"])
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert len(result) == 33
    assert result.startswith("ala")
    assert not any(ln.startswith("krok") for ln in lines)
    assert ALA_STATS in lines
    assert "--- tekst ---" in lines


def test_main_length_zero(main, plik_ala):
    result = main(["--text", str(plik_ala), "--steps", "10", "--prompt", "ala", "--length", "0"])
    assert result == "ala"


def test_main_uses_text_without_download(main, base_args, tmp_path):
    brak = tmp_path / "brak"
    result = main([*base_args, "--slug", "Zly Slug", "--data-dir", str(brak)])
    assert len(result) == 33
    assert not brak.exists()


def test_main_existing_book_default_slug(main, tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "lalka-tom-pierwszy.txt").write_text(ALA_TEXT, encoding="utf-8")
    result = main([
        "--data-dir", str(raw), "--steps", "20", "--log-every", "10",
        "--prompt", "ala", "--length", "5",
    ])  # fmt: skip
    out = capsys.readouterr().out
    assert len(result) == 8
    assert result.startswith("ala")
    assert ALA_STATS in out.splitlines()
    assert [p.name for p in raw.iterdir()] == ["lalka-tom-pierwszy.txt"]


def test_main_existing_book_custom_slug(main, tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "moja-ksiazka.txt").write_text("abcabc" * 20, encoding="utf-8")
    result = main([
        "--slug", "moja-ksiazka", "--data-dir", str(raw), "--steps", "20",
        "--log-every", "10", "--prompt", "ab", "--length", "4",
    ])  # fmt: skip
    out = capsys.readouterr().out
    assert len(result) == 6
    assert result.startswith("ab")
    assert "znaki: 120, slownik: 3, parametry: 9" in out.splitlines()


def test_main_prompt_out_of_vocab(main, plik_ala):
    with pytest.raises(ValueError):
        main(["--text", str(plik_ala), "--steps", "5", "--prompt", "xyz", "--length", "5"])


def test_main_empty_prompt(main, plik_ala):
    with pytest.raises(ValueError):
        main(["--text", str(plik_ala), "--steps", "5", "--prompt", "", "--length", "5"])


def test_main_text_too_short(main, tmp_path):
    krotki = tmp_path / "krotki.txt"
    krotki.write_text("abc", encoding="utf-8")
    with pytest.raises(ValueError):
        main(["--text", str(krotki), "--steps", "1", "--prompt", "a", "--length", "5"])


@pytest.mark.parametrize(
    "extra",
    [
        ["--batch-size", "0"],
        ["--block-size", "0"],
        ["--lr", "0"],
        ["--log-every", "0"],
        ["--steps", "-1"],
        ["--length", "-1"],
    ],
)
def test_main_bad_values(main, base_args, extra):
    with pytest.raises(ValueError):
        main([*base_args, *extra])


def test_script_utf8_with_redirected_stdout(plik_polski, tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
    env["PYTHONIOENCODING"] = "ascii"
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--text", str(plik_polski), "--steps", "20",
            "--log-every", "10", "--prompt", "żó", "--length", "20",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=180,
        check=False,
    )  # fmt: skip
    stderr = proc.stderr.decode("utf-8", errors="replace")
    assert proc.returncode == 0, stderr
    out = proc.stdout.decode("utf-8")
    assert "--- tekst ---" in out.splitlines(), stderr
    line = line_after_marker(out)
    assert len(line) == 22
    assert line.startswith("żó")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="brak GPU")
def test_main_cuda(main, base_args):
    result = main([*base_args, "--device", "cuda"])
    assert len(result) == 33
    assert result.startswith("ala")
