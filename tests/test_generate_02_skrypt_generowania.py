"""Testy skryptu scripts/generate.py i bramka etapu 6 (KV cache == bez cache)."""

import os
import re
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from nanowriter.model import GPT, GPTConfig
from nanowriter.sample import generate
from nanowriter.tokenizer import BPETokenizer
from nanowriter.train import TrainConfig, load_gpt, make_optimizer, save_checkpoint

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "generate.py"
TRAIN_TEXT = "ala ma kota a kot ma ale " * 50
STATS_RE = r"tokeny: (\d+), czas: \d+\.\d{2} s, cache: (tak|nie)"
PREFIX = "--- tekst ---\n"


@pytest.fixture(scope="module")
def script():
    return runpy.run_path(str(SCRIPT))


@pytest.fixture
def parse_args(script):
    return script["parse_args"]


@pytest.fixture
def main(script):
    return script["main"]


@pytest.fixture
def pliki(tmp_path):
    tok = BPETokenizer.train(TRAIN_TEXT, 270)
    tok.save(tmp_path / "tok.json")
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=2, n_head=2, d_model=32))
    cfg = TrainConfig()
    save_checkpoint(tmp_path / "gpt.pt", m, make_optimizer(m, cfg), 0, torch.Generator(), cfg)
    return tmp_path / "gpt.pt", tmp_path / "tok.json"


def base(c, t):
    return ["--checkpoint", str(c), "--tokenizer", str(t), "--device", "cpu"]


def make_model():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, d_model=32)).eval()


def gen(s):
    return torch.Generator().manual_seed(s)


def stats_line(rest):
    assert rest.endswith("\n")
    line = rest[:-1]
    assert "\n" not in line
    m = re.fullmatch(STATS_RE, line)
    assert m is not None, line
    return m.groups()


@pytest.mark.parametrize("s", [0, 1, 2])
@pytest.mark.parametrize(
    "ustawienia",
    [
        {"temperature": 1.0, "top_k": None, "top_p": None},
        {"temperature": 0.8, "top_k": 10, "top_p": 0.9},
        {"temperature": 0.0},
    ],
    ids=["domyslne", "filtry", "argmax"],
)
@pytest.mark.parametrize("B,T", [(1, 1), (2, 5), (1, 16)])
def test_bramka_generate_cache_rowny_bez_cache(s, ustawienia, B, T):
    model = make_model()
    torch.manual_seed(100)
    idx = torch.randint(0, 64, (B, T))
    a = generate(model, idx, 40, generator=gen(s), block_size=16, **ustawienia, use_cache=True)
    b = generate(model, idx, 40, generator=gen(s), block_size=16, **ustawienia, use_cache=False)
    assert a.shape == (B, T + 40)
    assert b.shape == (B, T + 40)
    assert a.dtype == torch.int64
    assert b.dtype == torch.int64
    assert torch.equal(a, b)
    assert torch.equal(a[:, :T], idx)


@pytest.mark.parametrize("s", [0, 1])
def test_bramka_generate_cache_bez_block_size(s):
    model = make_model()
    torch.manual_seed(100)
    idx = torch.randint(0, 64, (2, 3))
    a = generate(model, idx, 13, generator=gen(s), block_size=None, use_cache=True)
    b = generate(model, idx, 13, generator=gen(s), block_size=None, use_cache=False)
    assert a.shape == (2, 16)
    assert torch.equal(a, b)


@pytest.mark.parametrize("s", [0, 1, 2])
def test_bramka_skrypt_cache_rowny_bez_cache(main, pliki, s):
    b0 = base(*pliki)
    opcje = ["--prompt", "ala", "--length", "40", "--seed", str(s)]
    a = main(b0 + opcje)
    b = main(b0 + opcje + ["--no-cache"])
    assert isinstance(a, str)
    assert a == b
    assert a.startswith("ala")


def test_bramka_skrypt_bez_filtrow(main, pliki):
    opcje = ["--prompt", "ala", "--length", "40", "--seed", "5", "--temperature", "1.0",
             "--top-k", "0", "--top-p", "1.0"]
    assert main(base(*pliki) + opcje) == main(base(*pliki) + opcje + ["--no-cache"])


def test_skrypt_prompt_dluzszy_niz_block_size(main, pliki):
    prompt = "ala ma kota a kot ma ale " * 6
    opcje = ["--prompt", prompt, "--length", "20"]
    a = main(base(*pliki) + opcje)
    b = main(base(*pliki) + opcje + ["--no-cache"])
    assert a == b
    assert a.startswith(prompt)


def test_parse_args_domyslne(parse_args):
    a = parse_args([])
    assert a.checkpoint == Path("data/checkpoints/gpt.pt")
    assert isinstance(a.checkpoint, Path)
    assert a.tokenizer == Path("data/tokens/tokenizer.json")
    assert isinstance(a.tokenizer, Path)
    assert a.prompt == "Wieczorem"
    assert a.length == 500 and isinstance(a.length, int)
    assert a.temperature == 0.8 and isinstance(a.temperature, float)
    assert a.top_k == 50 and isinstance(a.top_k, int)
    assert a.top_p == 0.95 and isinstance(a.top_p, float)
    assert a.seed == 0 and isinstance(a.seed, int)
    assert a.device == "cuda"
    assert a.no_cache is False


def test_parse_args_wylaczenie_filtrow(parse_args):
    a = parse_args(["--top-k", "0", "--no-cache"])
    assert a.top_k == 0
    assert a.no_cache is True
    b = parse_args(["--top-p", "1.0"])
    assert b.top_p == 1.0
    assert b.no_cache is False


def test_parse_args_wszystkie_opcje(parse_args):
    a = parse_args([
        "--checkpoint", "a.pt", "--tokenizer", "t.json", "--prompt", "Ala", "--length", "7",
        "--temperature", "0.5", "--top-k", "3", "--top-p", "0.5", "--seed", "42",
        "--device", "cpu", "--no-cache",
    ])
    assert a.checkpoint == Path("a.pt") and isinstance(a.checkpoint, Path)
    assert a.tokenizer == Path("t.json") and isinstance(a.tokenizer, Path)
    assert a.prompt == "Ala"
    assert a.length == 7 and isinstance(a.length, int)
    assert a.temperature == 0.5 and isinstance(a.temperature, float)
    assert a.top_k == 3 and isinstance(a.top_k, int)
    assert a.top_p == 0.5 and isinstance(a.top_p, float)
    assert a.seed == 42 and isinstance(a.seed, int)
    assert a.device == "cpu"
    assert a.no_cache is True


def test_parse_args_argv_none(parse_args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["generate.py", "--length", "3", "--prompt", "Ola"])
    for a in (parse_args(), parse_args(None)):
        assert a.length == 3
        assert a.prompt == "Ola"
        assert a.top_k == 50


def test_parse_args_help(parse_args, capsys):
    with pytest.raises(SystemExit) as e:
        parse_args(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for opcja in ["--checkpoint", "--tokenizer", "--prompt", "--length", "--temperature",
                  "--top-k", "--top-p", "--seed", "--device", "--no-cache"]:
        assert opcja in out


def test_main_przyklad_wyjscie(main, pliki, capsys):
    b0 = base(*pliki)
    text = main(b0 + ["--prompt", "ala", "--length", "40"])
    out = capsys.readouterr().out
    assert isinstance(text, str)
    assert text.startswith("ala")
    przedrostek = PREFIX + text + "\n"
    assert out.startswith(przedrostek)
    assert stats_line(out[len(przedrostek):]) == ("40", "tak")

    text2 = main(b0 + ["--prompt", "ala", "--length", "40", "--no-cache"])
    out2 = capsys.readouterr().out
    assert text2 == text
    assert out2.startswith(przedrostek)
    assert stats_line(out2[len(przedrostek):]) == ("40", "nie")


@pytest.mark.parametrize(
    "opcje,temperature,top_k,top_p,s",
    [
        (["--seed", "3", "--temperature", "1.0", "--top-k", "0", "--top-p", "1.0"],
         1.0, None, None, 3),
        (["--seed", "4", "--temperature", "0.7", "--top-k", "5", "--top-p", "0.9"],
         0.7, 5, 0.9, 4),
        (["--seed", "0"], 0.8, 50, 0.95, 0),
    ],
)
def test_main_zgodny_z_modulami(main, pliki, opcje, temperature, top_k, top_p, s):
    c, t = pliki
    tok = BPETokenizer.load(t)
    model = load_gpt(c, "cpu")
    idx = torch.tensor([tok.encode("ala")], dtype=torch.int64)
    oczekiwany = tok.decode(
        generate(
            model, idx, 30, generator=gen(s), block_size=16, temperature=temperature,
            top_k=top_k, top_p=top_p, use_cache=True,
        )[0].tolist()
    )
    assert main(base(c, t) + ["--prompt", "ala", "--length", "30"] + opcje) == oczekiwany


def test_main_length_zero(main, pliki, capsys):
    text = main(base(*pliki) + ["--prompt", "ala", "--length", "0"])
    out = capsys.readouterr().out
    assert text == "ala"
    przedrostek = PREFIX + text + "\n"
    assert out.startswith(przedrostek)
    assert stats_line(out[len(przedrostek):]) == ("0", "tak")


def test_main_temperatura_zero_niezalezna_od_seeda(main, pliki):
    opcje = ["--prompt", "ala", "--length", "40", "--temperature", "0"]
    wyniki = [main(base(*pliki) + opcje + ["--seed", s]) for s in ("0", "1", "7")]
    assert wyniki[0] == wyniki[1] == wyniki[2]


def test_main_powtarzalnosc_seed(main, pliki):
    opcje = ["--prompt", "ala", "--length", "40", "--temperature", "1.0",
             "--top-k", "0", "--top-p", "1.0"]
    a = main(base(*pliki) + opcje + ["--seed", "0"])
    b = main(base(*pliki) + opcje + ["--seed", "0"])
    c = main(base(*pliki) + opcje + ["--seed", "1"])
    assert a == b
    assert a != c


def test_main_polski_prompt(main, pliki, capsys):
    text = main(base(*pliki) + ["--prompt", "Zażółć", "--length", "20"])
    assert text.startswith("Zażółć")
    assert "Zażółć" in capsys.readouterr().out


def test_main_pusty_prompt(main, pliki):
    with pytest.raises(ValueError):
        main(base(*pliki) + ["--prompt", "", "--length", "5"])


def test_main_brak_checkpointu(main, pliki, tmp_path):
    _, t = pliki
    with pytest.raises(FileNotFoundError):
        main(["--checkpoint", str(tmp_path / "brak.pt"), "--tokenizer", str(t),
              "--device", "cpu", "--prompt", "ala", "--length", "5"])


def test_main_brak_slownika(main, pliki, tmp_path):
    c, _ = pliki
    with pytest.raises(FileNotFoundError):
        main(["--checkpoint", str(c), "--tokenizer", str(tmp_path / "brak.json"),
              "--device", "cpu", "--prompt", "ala", "--length", "5"])


@pytest.mark.parametrize(
    "zle",
    [["--temperature", "-1"], ["--top-k", "-1"], ["--top-p", "0"], ["--top-p", "1.5"]],
)
def test_main_zle_parametry_probkowania(main, pliki, zle):
    with pytest.raises(ValueError):
        main(base(*pliki) + ["--prompt", "ala", "--length", "5"] + zle)


def test_import_skryptu_nie_uruchamia_main(capsys):
    capsys.readouterr()
    g = runpy.run_path(str(SCRIPT))
    assert capsys.readouterr().out == ""
    assert callable(g["parse_args"])
    assert callable(g["main"])


def test_skrypt_jako_proces_emoji(pliki):
    c, t = pliki
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--checkpoint", str(c), "--tokenizer", str(t),
         "--device", "cpu", "--prompt", "Zażółć 🙂", "--length", "10"],
        cwd=REPO, env=env, capture_output=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    out = r.stdout.decode("utf-8")
    assert "--- tekst ---" in out
    assert "Zażółć 🙂" in out
    assert re.search(STATS_RE, out).groups() == ("10", "tak")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="brak GPU")
def test_skrypt_gpu_cache_rowny_bez_cache(main, pliki):
    c, t = pliki
    opcje = ["--checkpoint", str(c), "--tokenizer", str(t), "--device", "cuda",
             "--prompt", "ala", "--length", "40", "--seed", "0"]
    a = main(opcje)
    assert a == main(opcje + ["--no-cache"])
    assert a.startswith("ala")
