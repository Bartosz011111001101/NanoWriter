"""Testy skryptu scripts/prepare_tokens_rozmowca.py i bramka zadania 08."""

import contextlib
import io
import json
import os
import re
import runpy
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from nanowriter.dataset import (
    TokenMeta,
    get_masked_batch,
    load_mask,
    load_meta,
    load_tokens,
)
from nanowriter.tokenizer import BPETokenizer

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_tokens_rozmowca.py"

CORPUS_TRAIN = (
    "Pan Wokulski — rzekł subiekt — nie przyjdzie dziś.\n"
    "Zażółć gęślą jaźń, bo „cytat” kończy się tak.\n"
    "Ala ma kota \U0001F600 i psa \U0001F1F5\U0001F1F1.\n"
    "Napisy: to jest tylko test, nic więcej!\n"
) * 40
CORPUS_VAL = ("Kot ma Alę.\n" "Żółw \U0001F422 idzie wolno.\n") * 5

D_LINES = [
    "A: Cześć, jak się masz?",
    "B: Dziękuję, mam się świetnie.",
    "A: To dobrze.",
    "B: A ty? \U0001F600",
    "",
    "A: Zażółć gęślą jaźń.",
    "B: Zażółć gęślą jaźń.",
    "",
    "A: Co robisz wieczorem?",
    "B: Czytam książkę, a potem idę spać!",
    "A: Miłej lektury \U0001F4DA",
    "B: Dziękuję \U0001F600",
    "",
]
V_LINES = [
    "A: Żółw?",
    "B: Żółw \U0001F422 nie biega.",
    "",
    "A: Pa.",
    "B: Do zobaczenia.",
    "",
]
DIALOGS_TRAIN = "".join(ln + "\n" for ln in D_LINES) * 12
DIALOGS_VAL = "".join(ln + "\n" for ln in V_LINES) * 3

EDGE_TEXTS = [
    "",
    "Zażółć gęślą jaźń",
    "\U0001F600 \U0001F1F5\U0001F1F1 \U0001F44D\U0001F3FD",
    "\n\t",
    "A: hej.\nB: ok \U0001F600\n",
]

EX_CORPUS_TRAIN = "ala ma kota\n" * 200
EX_CORPUS_VAL = "kot ma ale\n" * 20
EX_DIALOGS_TRAIN = "A: hej\nB: ok tak\n\n" * 50
EX_DIALOGS_VAL = "A: hej\nB: ok tak\n\n" * 5

OUT_FILES = [
    "tokeny-rozmowy/meta.json",
    "tokeny-rozmowy/train.bin",
    "tokeny-rozmowy/train_mask.bin",
    "tokeny-rozmowy/val.bin",
    "tokeny-rozmowy/val_mask.bin",
    "tokeny-wstepne/meta.json",
    "tokeny-wstepne/tokenizer.json",
    "tokeny-wstepne/train.bin",
    "tokeny-wstepne/val.bin",
]


def lines_b(text: str) -> list[str]:
    """Wypowiedzi B: linie zaczynające się od 'B: ' bez 'B:', z '\\n' na końcu."""
    return [ln[2:] + "\n" for ln in text.split("\n") if ln.startswith("B: ")]


def runs(mask: list[int]) -> list[tuple[int, int]]:
    """Ciągłe serie jedynek jako pary (początek, koniec wyłącznie)."""
    result = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            result.append((start, i))
            start = None
    if start is not None:
        result.append((start, len(mask)))
    return result


def _write(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def _setup(base: Path, corpus: tuple[str, str], dialogs: tuple[str, str]):
    corpus_dir = base / "corpus"
    dialogs_dir = base / "dialogs"
    corpus_dir.mkdir(parents=True)
    dialogs_dir.mkdir(parents=True)
    _write(corpus_dir / "train.txt", corpus[0])
    _write(corpus_dir / "val.txt", corpus[1])
    _write(dialogs_dir / "train.txt", dialogs[0])
    _write(dialogs_dir / "val.txt", dialogs[1])
    return corpus_dir, dialogs_dir, base / "out"


def _args(corpus_dir, dialogs_dir, out, vocab="300", extra=()):
    return [
        "--corpus-dir", str(corpus_dir),
        "--dialogs-dir", str(dialogs_dir),
        "--out-dir", str(out),
        "--vocab-size", vocab,
        "--sample-mb", "1",
        "--dialog-sample-mb", "1",
        *extra,
    ]  # fmt: skip


def _run(main, args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = main(args)
    return result, buf.getvalue()


def _files(out: Path) -> list[str]:
    return sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())


@dataclass
class Gate:
    corpus: Path
    dialogs: Path
    out: Path
    corpus_meta: TokenMeta
    dialogs_meta: TokenMeta
    output: str
    tokenizer: BPETokenizer


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
def gate_run(main, tmp_path_factory) -> Gate:
    base = tmp_path_factory.mktemp("gate")
    corpus, dialogs, out = _setup(
        base, (CORPUS_TRAIN, CORPUS_VAL), (DIALOGS_TRAIN, DIALOGS_VAL)
    )
    (corpus_meta, dialogs_meta), output = _run(main, _args(corpus, dialogs, out))
    tokenizer = BPETokenizer.load(out / "tokeny-wstepne" / "tokenizer.json")
    return Gate(corpus, dialogs, out, corpus_meta, dialogs_meta, output, tokenizer)


# ---------------------------------------------------------------- argumenty


def test_parse_args_defaults(parse_args):
    a = parse_args([])
    assert a.corpus_dir == Path("data/pretrain")
    assert a.dialogs_dir == Path("data/rozmowy")
    assert a.out_dir == Path("data/rozmowca")
    assert a.vocab_size == 8192
    assert a.sample_mb == 8.0
    assert a.dialog_sample_mb == 4.0
    assert a.block_mb == 4.0
    for name in ("corpus_dir", "dialogs_dir", "out_dir"):
        assert isinstance(getattr(a, name), Path)
    assert isinstance(a.vocab_size, int)
    for name in ("sample_mb", "dialog_sample_mb", "block_mb"):
        assert isinstance(getattr(a, name), float)


def test_parse_args_overrides(parse_args):
    a = parse_args(["--vocab-size", "300"])
    assert a.vocab_size == 300
    assert a.corpus_dir == Path("data/pretrain")
    assert a.dialogs_dir == Path("data/rozmowy")
    assert a.out_dir == Path("data/rozmowca")
    assert a.sample_mb == 8.0
    assert a.dialog_sample_mb == 4.0
    assert a.block_mb == 4.0
    b = parse_args(
        [
            "--corpus-dir", "a",
            "--dialogs-dir", "b",
            "--out-dir", "c",
            "--vocab-size", "512",
            "--sample-mb", "1.5",
            "--dialog-sample-mb", "0.5",
            "--block-mb", "2",
        ]
    )  # fmt: skip
    assert b.corpus_dir == Path("a")
    assert b.dialogs_dir == Path("b")
    assert b.out_dir == Path("c")
    assert b.vocab_size == 512
    assert b.sample_mb == 1.5
    assert b.dialog_sample_mb == 0.5
    assert b.block_mb == 2.0


# ---------------------------------------------------------------- przykład z „Gotowe, gdy”


def test_example_from_prompt(main, tmp_path):
    corpus, dialogs, out = _setup(
        tmp_path, (EX_CORPUS_TRAIN, EX_CORPUS_VAL), (EX_DIALOGS_TRAIN, EX_DIALOGS_VAL)
    )
    (corpus_meta, dialogs_meta), output = _run(main, _args(corpus, dialogs, out, "256"))
    assert corpus_meta == TokenMeta(256, "uint16", 2400, 2400, 220, 220)
    assert dialogs_meta == TokenMeta(256, "uint16", 900, 900, 90, 90)

    lines = output.splitlines()
    assert len(lines) == 7
    assert lines[0] == (
        "probka: 2400 znakow z korpusu + 900 znakow z rozmow, trening BPE (vocab 256)"
    )
    assert lines[1].startswith("slownik: 256 tokenow -> ")
    assert lines[1].endswith("tokenizer.json")
    assert lines[2] == "korpus train: 2400 tokenow, 2400 znakow, 1.000 znaku/token"
    assert lines[3] == "korpus val: 220 tokenow, 220 znakow, 1.000 znaku/token"
    assert lines[4] == "rozmowy train: 900 tokenow, z maska 400 (44.4%), 900 znakow"
    assert lines[5] == "rozmowy val: 90 tokenow, z maska 40 (44.4%), 90 znakow"
    assert re.match(r"^czas: \d+\.\d s$", lines[6])

    d_out = out / "tokeny-rozmowy"
    ids = load_tokens(d_out / "train.bin").tolist()
    assert ids == list(EX_DIALOGS_TRAIN.encode("utf-8"))
    train_mask = load_mask(d_out / "train_mask.bin")
    val_mask = load_mask(d_out / "val_mask.bin")
    assert len(train_mask) == 900
    assert int(train_mask.sum()) == 400
    assert len(val_mask) == 90
    assert int(val_mask.sum()) == 40
    for a, b in runs(train_mask.tolist()):
        assert b - a == 8
        assert bytes(ids[a:b]) == b" ok tak\n"


# ---------------------------------------------------------------- bramka zadania 08


def test_gate_files_and_formats(gate_run):
    out = gate_run.out
    assert _files(out) == OUT_FILES
    tokenizer = gate_run.tokenizer
    assert 256 < tokenizer.vocab_size <= 300
    cm, dm = gate_run.corpus_meta, gate_run.dialogs_meta
    w, r = out / "tokeny-wstepne", out / "tokeny-rozmowy"
    assert (w / "train.bin").stat().st_size == 2 * cm.train_tokens
    assert (w / "val.bin").stat().st_size == 2 * cm.val_tokens
    assert (r / "train.bin").stat().st_size == 2 * dm.train_tokens
    assert (r / "val.bin").stat().st_size == 2 * dm.val_tokens
    assert (r / "train_mask.bin").stat().st_size == dm.train_tokens
    assert (r / "val_mask.bin").stat().st_size == dm.val_tokens
    for name in ("train_mask.bin", "val_mask.bin"):
        assert set((r / name).read_bytes()) <= {0, 1}
    for meta_path in (w / "meta.json", r / "meta.json"):
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert data["dtype"] == "uint16"
        assert data["vocab_size"] == tokenizer.vocab_size
    for meta in (cm, dm):
        assert meta.vocab_size == tokenizer.vocab_size
        assert meta.dtype == "uint16"
    assert dm.train_tokens < len(DIALOGS_TRAIN.encode("utf-8"))


def test_gate_meta_equals_files(gate_run):
    out = gate_run.out
    cm, dm = gate_run.corpus_meta, gate_run.dialogs_meta
    w, r = out / "tokeny-wstepne", out / "tokeny-rozmowy"
    assert load_meta(w / "meta.json") == cm
    assert load_meta(r / "meta.json") == dm
    assert cm.train_tokens == len(load_tokens(w / "train.bin"))
    assert cm.val_tokens == len(load_tokens(w / "val.bin"))
    assert dm.train_tokens == len(load_tokens(r / "train.bin"))
    assert dm.val_tokens == len(load_tokens(r / "val.bin"))
    assert cm.train_chars == len(CORPUS_TRAIN)
    assert cm.val_chars == len(CORPUS_VAL)
    assert dm.train_chars == len(DIALOGS_TRAIN)
    assert dm.val_chars == len(DIALOGS_VAL)
    assert dm.train_chars != len(DIALOGS_TRAIN.encode("utf-8"))


def test_gate_roundtrip(gate_run):
    out, tokenizer = gate_run.out, gate_run.tokenizer
    cases = [
        (out / "tokeny-wstepne" / "train.bin", CORPUS_TRAIN),
        (out / "tokeny-wstepne" / "val.bin", CORPUS_VAL),
        (out / "tokeny-rozmowy" / "train.bin", DIALOGS_TRAIN),
        (out / "tokeny-rozmowy" / "val.bin", DIALOGS_VAL),
    ]
    for path, text in cases:
        ids = load_tokens(path).tolist()
        assert tokenizer.decode(ids) == text
        assert ids == tokenizer.encode(text)


def test_gate_tokenizer_edge_texts(gate_run):
    tokenizer = gate_run.tokenizer
    for t in EDGE_TEXTS:
        ids = tokenizer.encode(t)
        assert tokenizer.decode(ids) == t
        assert all(0 <= i < tokenizer.vocab_size for i in ids)
    saved = load_tokens(gate_run.out / "tokeny-rozmowy" / "train.bin").tolist()
    assert tokenizer.encode(DIALOGS_TRAIN) == saved


def test_gate_mask_length_and_runs(gate_run):
    tokenizer = gate_run.tokenizer
    r = gate_run.out / "tokeny-rozmowy"
    ids = load_tokens(r / "train.bin").tolist()
    mask = load_mask(r / "train_mask.bin").tolist()
    assert len(mask) == len(ids)
    series = runs(mask)
    assert len(series) == 60
    decoded = [tokenizer.decode(ids[a:b]) for a, b in series]
    assert decoded == lines_b(DIALOGS_TRAIN)
    assert decoded[0] == " Dziękuję, mam się świetnie.\n"
    assert decoded[1] == " A ty? \U0001F600\n"
    assert decoded[2] == " Zażółć gęślą jaźń.\n"

    val_ids = load_tokens(r / "val.bin").tolist()
    val_mask = load_mask(r / "val_mask.bin").tolist()
    assert len(val_mask) == len(val_ids)
    val_series = runs(val_mask)
    assert len(val_series) == 6
    val_decoded = [tokenizer.decode(val_ids[a:b]) for a, b in val_series]
    assert val_decoded == [" Żółw \U0001F422 nie biega.\n", " Do zobaczenia.\n"] * 3


def test_gate_mask_zero_part(gate_run):
    tokenizer = gate_run.tokenizer
    r = gate_run.out / "tokeny-rozmowy"
    ids = load_tokens(r / "train.bin").tolist()
    mask = load_mask(r / "train_mask.bin").tolist()
    zero_ids = [i for i, m in zip(ids, mask, strict=True) if m == 0]
    block = (
        "A: Cześć, jak się masz?\nB:A: To dobrze.\nB:\n"
        "A: Zażółć gęślą jaźń.\nB:\n"
        "A: Co robisz wieczorem?\nB:A: Miłej lektury \U0001F4DA\nB:\n"
    )
    assert tokenizer.decode(zero_ids) == block * 12


def test_gate_mask_run_boundaries(gate_run):
    tokenizer = gate_run.tokenizer
    r = gate_run.out / "tokeny-rozmowy"
    ids = load_tokens(r / "train.bin").tolist()
    mask = load_mask(r / "train_mask.bin").tolist()
    newline = tokenizer.encode("\n")[0]
    bc = tokenizer.encode("B:")
    assert bc == [66, 58]
    for a, b in runs(mask):
        assert a >= 2
        assert ids[a - 2 : a] == [66, 58]
        assert tokenizer.decode([ids[b - 1]]) == "\n"
        assert ids[b - 1] == newline
        assert ids[a] != newline


def test_gate_newline_boundaries(main, tmp_path):
    corpus, dialogs, out = _setup(
        tmp_path,
        ("Ala ma kota.\nKot ma Alę.\n" * 50, "Ala ma kota.\n" * 5),
        (
            "A: Hej.\nB: Dobrze.\n\nA: Co tam?\nB: Super \U0001F600\n\nA: Pa!\nB: Pa pa.\n\n" * 30,
            "A: Hej.\nB: Dobrze.\n\nA: Co tam?\nB: Super \U0001F600\n\nA: Pa!\nB: Pa pa.\n\n" * 2,
        ),
    )
    _run(main, _args(corpus, dialogs, out))
    tokenizer = BPETokenizer.load(out / "tokeny-wstepne" / "tokenizer.json")
    assert tokenizer.encode("\n") == [10]
    for i in range(tokenizer.vocab_size):
        assert ("\n" in tokenizer.decode([i])) == (i == 10)
    r = out / "tokeny-rozmowy"
    ids = load_tokens(r / "train.bin").tolist()
    series = runs(load_mask(r / "train_mask.bin").tolist())
    assert len(series) == 90
    assert all(ids[b - 1] == 10 for _, b in series)
    assert tokenizer.decode([10]) == "\n"
    decoded = [tokenizer.decode(ids[a:b]) for a, b in series]
    assert decoded == [" Dobrze.\n", " Super \U0001F600\n", " Pa pa.\n"] * 30


def test_bpe_newline_never_merged_module():
    words = [
        "ala", "kot", "pies", "dom", "las", "rzeka", "morze", "góra", "szkoła", "książka",
        "okno", "drzwi", "stół", "krzesło", "ogród", "kwiat", "drzewo", "chmura", "słońce",
        "księżyc", "wiatr", "deszcz", "śnieg", "lato", "zima", "wiosna", "jesień", "miasto",
        "wieś", "droga", "most", "rynek", "kościół", "zamek", "park", "plac", "ulica",
        "sklep", "praca", "wakacje",
    ]  # fmt: skip
    text = "".join(
        " ".join(words[(7 * i + 3 * j) % 40] for j in range(6)) + ".\n" for i in range(300)
    )
    tokenizer = BPETokenizer.train(text, 1000)
    assert tokenizer.vocab_size > 300
    for i in range(256, tokenizer.vocab_size):
        assert "\n" not in tokenizer.decode([i])
    assert tokenizer.encode("\n") == [10]
    assert tokenizer.encode("Ala ma kota." + "\n")[-1] == 10
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_gate_masked_batch_on_files(gate_run):
    r = gate_run.out / "tokeny-rozmowy"
    data = load_tokens(r / "train.bin")
    mask = load_mask(r / "train_mask.bin")
    n = len(data)
    for seed in (0, 1, 2):
        x, y, m = get_masked_batch(data, mask, 16, 32, torch.Generator().manual_seed(seed))
        x_ar, _, _ = get_masked_batch(
            torch.arange(n), mask, 16, 32, torch.Generator().manual_seed(seed)
        )
        assert x.shape == y.shape == m.shape == (16, 32)
        assert x.dtype == torch.int64
        assert y.dtype == torch.int64
        assert m.dtype == torch.bool
        for b in range(16):
            s = int(x_ar[b, 0])
            assert torch.equal(x[b], data[s : s + 32])
            assert torch.equal(y[b], data[s + 1 : s + 33])
            assert torch.equal(m[b], mask[s + 1 : s + 33].to(torch.bool))


def test_gate_masked_batch_mask_fraction(gate_run):
    r = gate_run.out / "tokeny-rozmowy"
    data = load_tokens(r / "train.bin")
    mask = load_mask(r / "train_mask.bin")
    total = torch.cat(
        [
            get_masked_batch(data, mask, 64, 32, torch.Generator().manual_seed(s))[2].flatten()
            for s in (0, 1, 2, 3)
        ]
    )
    fraction = total.float().mean().item()
    assert 0.30 < fraction < 0.65


# ---------------------------------------------------------------- brzegi i niezmienniki


def test_empty_dialogs_val(main, tmp_path):
    corpus, dialogs, out = _setup(
        tmp_path, (CORPUS_TRAIN, CORPUS_VAL), (DIALOGS_TRAIN, "")
    )
    (_, dialogs_meta), _ = _run(main, _args(corpus, dialogs, out))
    r = out / "tokeny-rozmowy"
    assert (r / "val.bin").stat().st_size == 0
    assert (r / "val_mask.bin").stat().st_size == 0
    assert dialogs_meta.val_tokens == 0
    assert dialogs_meta.val_chars == 0
    assert len(load_tokens(r / "val.bin")) == 0
    assert len(load_mask(r / "val_mask.bin")) == 0
    assert dialogs_meta.train_tokens > 0
    assert load_meta(r / "meta.json") == dialogs_meta


def test_block_size_does_not_change_output(main, tmp_path):
    corpus, dialogs, out1 = _setup(
        tmp_path / "a", (CORPUS_TRAIN, CORPUS_VAL), (DIALOGS_TRAIN, DIALOGS_VAL)
    )
    out2 = tmp_path / "out2"
    metas1, _ = _run(main, _args(corpus, dialogs, out1))
    metas2, _ = _run(main, _args(corpus, dialogs, out2, extra=["--block-mb", "0.001"]))
    assert len(CORPUS_TRAIN) > 4000
    assert metas1 == metas2
    assert _files(out1) == OUT_FILES
    assert _files(out2) == OUT_FILES
    for name in OUT_FILES:
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes()


def test_reproducible(main, tmp_path):
    corpus, dialogs, out1 = _setup(
        tmp_path / "a", (CORPUS_TRAIN, CORPUS_VAL), (DIALOGS_TRAIN, DIALOGS_VAL)
    )
    out2 = tmp_path / "out2"
    metas1, _ = _run(main, _args(corpus, dialogs, out1))
    metas2, _ = _run(main, _args(corpus, dialogs, out2))
    assert metas1 == metas2
    for name in OUT_FILES:
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes()


def test_no_files_outside_out_dir(main, tmp_path, monkeypatch):
    base = tmp_path
    cwd = base / "cwd"
    cwd.mkdir()
    corpus, dialogs, out = _setup(
        base, (CORPUS_TRAIN, CORPUS_VAL), (DIALOGS_TRAIN, DIALOGS_VAL)
    )
    monkeypatch.chdir(cwd)
    _run(main, _args(corpus, dialogs, out))
    assert list(cwd.iterdir()) == []
    assert sorted(p.name for p in corpus.iterdir()) == ["train.txt", "val.txt"]
    assert sorted(p.name for p in dialogs.iterdir()) == ["train.txt", "val.txt"]
    assert (corpus / "train.txt").read_bytes() == CORPUS_TRAIN.encode("utf-8")
    assert (corpus / "val.txt").read_bytes() == CORPUS_VAL.encode("utf-8")
    assert (dialogs / "train.txt").read_bytes() == DIALOGS_TRAIN.encode("utf-8")
    assert (dialogs / "val.txt").read_bytes() == DIALOGS_VAL.encode("utf-8")
    assert sorted(p.name for p in base.iterdir()) == ["corpus", "cwd", "dialogs", "out"]


def _snapshot(path: Path) -> list[tuple[str, int]]:
    if not path.exists():
        return []
    return sorted(
        (p.relative_to(path).as_posix(), p.stat().st_size) for p in path.rglob("*") if p.is_file()
    )


def test_old_model_untouched(main, tmp_path):
    data_dir = Path(__file__).resolve().parents[1] / "data"
    names = ["tokens", "checkpoints", "rozmowca"]
    existed = {n: (data_dir / n).exists() for n in names}
    before = {n: _snapshot(data_dir / n) for n in names}
    corpus, dialogs, out = _setup(
        tmp_path, (CORPUS_TRAIN, CORPUS_VAL), (DIALOGS_TRAIN, DIALOGS_VAL)
    )
    _run(main, _args(corpus, dialogs, out))
    for n in names:
        assert _snapshot(data_dir / n) == before[n]
        assert (data_dir / n).exists() == existed[n]


@pytest.mark.parametrize(
    "missing",
    ["corpus/train.txt", "corpus/val.txt", "dialogs/train.txt", "dialogs/val.txt"],
)
def test_missing_input_files(main, tmp_path, missing):
    corpus, dialogs, out = _setup(
        tmp_path, (CORPUS_TRAIN, CORPUS_VAL), (DIALOGS_TRAIN, DIALOGS_VAL)
    )
    (tmp_path / missing).unlink()
    with pytest.raises(FileNotFoundError):
        _run(main, _args(corpus, dialogs, out, "256"))


def test_bad_dialog_line(main, tmp_path):
    corpus, dialogs, out = _setup(
        tmp_path,
        (CORPUS_TRAIN, CORPUS_VAL),
        ("A: hej\nB: ok\n\nC: x\n", "A: hej\nB: ok\n\n"),
    )
    with pytest.raises(ValueError):
        _run(main, _args(corpus, dialogs, out))


def test_vocab_size_too_small(main, tmp_path):
    corpus, dialogs, out = _setup(
        tmp_path, (CORPUS_TRAIN, CORPUS_VAL), (DIALOGS_TRAIN, DIALOGS_VAL)
    )
    with pytest.raises(ValueError):
        _run(main, _args(corpus, dialogs, out, "100"))


def test_script_as_subprocess(tmp_path):
    corpus, dialogs, out = _setup(
        tmp_path / "dane", (EX_CORPUS_TRAIN, EX_CORPUS_VAL), (EX_DIALOGS_TRAIN, EX_DIALOGS_VAL)
    )
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *_args(corpus, dialogs, out, "256")],
        capture_output=True,
        env={**os.environ, "PYTHONUTF8": "1"},
        cwd=tmp_path,
        timeout=180,
        check=False,
    )
    stderr = proc.stderr.decode("utf-8", errors="replace")
    assert proc.returncode == 0, stderr
    lines = proc.stdout.decode("utf-8").splitlines()
    assert "rozmowy train: 900 tokenow, z maska 400 (44.4%), 900 znakow" in lines
    assert _files(out) == OUT_FILES
