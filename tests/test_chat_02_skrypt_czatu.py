"""Testy skryptu scripts/chat.py i bramka etapu 10 (rozmowa z modelem w terminalu)."""

import builtins
import os
import runpy
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

import nanowriter.train
from nanowriter.model import GPT, BigramLanguageModel, GPTConfig
from nanowriter.sample import build_context, chat_reply
from nanowriter.tokenizer import BPETokenizer
from nanowriter.train import TrainConfig, make_optimizer, save_checkpoint

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "chat.py"
WELCOME = "Rozmowa z modelem. Wpisz /koniec albo naciśnij Ctrl+C, żeby wyjść."

# Pary [poprzedni, następny] atrapy: ":" -> " ", " " -> "o", "o" -> "k", "k" -> "\n",
# "\n" -> "A", "A" -> ":". Po "B:" model pisze więc " ok\nA: ok\nA: ...".
PARY_OK = [(58, 32), (32, 111), (111, 107), (107, 10), (10, 65), (65, 58)]


class Wejscia:
    """Podmienione builtins.input: zwraca kolejne elementy listy, wyjątki rzuca."""

    def __init__(self, elementy):
        self.elementy = list(elementy)
        self.pozycja = 0

    def __call__(self, prompt=""):
        if self.pozycja >= len(self.elementy):
            raise AssertionError("input wywołane za dużo razy")
        element = self.elementy[self.pozycja]
        self.pozycja += 1
        if isinstance(element, BaseException):
            raise element
        return element


def ustaw_wejscia(monkeypatch, elementy):
    wejscia = Wejscia(elementy)
    monkeypatch.setattr(builtins, "input", wejscia)
    return wejscia


def zrob_atrape(block_size=64, pusta=False):
    """Zwraca (atrapa, dlugosci): bigram z tabelą PARY_OK i listą długości wejść modelu."""
    model = BigramLanguageModel(256)
    with torch.no_grad():
        waga = model.token_embedding.weight
        waga.zero_()
        for poprzedni, nastepny in PARY_OK:
            waga[poprzedni, nastepny] = 10.0
        if pusta:
            waga[58, 32] = 0.0
            waga[58, 10] = 10.0
    model.config = types.SimpleNamespace(block_size=block_size)
    model.eval()
    dlugosci = []
    model.register_forward_pre_hook(lambda modul, args: dlugosci.append(args[0].shape[1]))
    return model, dlugosci


def bajtowy_tokenizer(tmp_path):
    tok = BPETokenizer([])
    tok.save(tmp_path / "tok.json")
    return tok


def baza(tmp_path):
    return [
        "--checkpoint", str(tmp_path / "gpt.pt"),
        "--tokenizer", str(tmp_path / "tok.json"),
        "--device", "cpu",
        "--no-cache",
        "--temperature", "0",
        "--max-reply-tokens", "8",
    ]


def baza_gpt(tmp_path, device="cpu"):
    return [
        "--checkpoint", str(tmp_path / "gpt.pt"),
        "--tokenizer", str(tmp_path / "tok.json"),
        "--device", device,
        "--max-reply-tokens", "8",
    ]


def uruchom_z_atrapa(monkeypatch, tmp_path, model, argumenty, wejscia):
    """Uruchamia main skryptu z atrapą zamiast load_gpt; zwraca (wynik, obiekt wejść)."""
    bajtowy_tokenizer(tmp_path)
    monkeypatch.setattr(nanowriter.train, "load_gpt", lambda path, device: model)
    ns = runpy.run_path(str(SCRIPT))
    obiekt_wejsc = ustaw_wejscia(monkeypatch, wejscia)
    wynik = ns["main"](argumenty)
    return wynik, obiekt_wejsc


def linie(capsys):
    wyjscie = capsys.readouterr().out
    wyjscie = wyjscie.removesuffix("\n")
    return wyjscie.split("\n")


@pytest.fixture
def parse_args():
    return runpy.run_path(str(SCRIPT))["parse_args"]


@pytest.fixture
def pliki_gpt(tmp_path):
    torch.manual_seed(0)
    tok = BPETokenizer.train("A: ala\nB: ma kota\n" * 50, 270)
    tok.save(tmp_path / "tok.json")
    config = GPTConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=1, n_head=2, d_model=16)
    model = GPT(config)
    cfg = TrainConfig()
    optimizer = make_optimizer(model, cfg)
    save_checkpoint(tmp_path / "gpt.pt", model, optimizer, 0, torch.Generator(), cfg)
    return tmp_path


def dwa_uruchomienia(monkeypatch, capsys, argumenty_a, argumenty_b, wejscia):
    """Uruchamia main z prawdziwym modelem dwa razy; zwraca (hist_a, tekst_a, hist_b, tekst_b)."""
    ns = runpy.run_path(str(SCRIPT))
    wyniki = []
    for argumenty in (argumenty_a, argumenty_b):
        ustaw_wejscia(monkeypatch, wejscia)
        capsys.readouterr()
        historia = ns["main"](argumenty)
        wyniki.extend([historia, capsys.readouterr().out])
    return tuple(wyniki)


# --- parse_args ---


def test_parse_args_defaults(parse_args):
    args = parse_args(["--checkpoint", "m.pt", "--tokenizer", "t.json"])

    assert args.checkpoint == Path("m.pt")
    assert isinstance(args.checkpoint, Path)
    assert args.tokenizer == Path("t.json")
    assert isinstance(args.tokenizer, Path)
    assert args.max_turns == 4
    assert args.max_reply_tokens == 64
    assert args.temperature == 0.7
    assert args.top_k == 40
    assert args.top_p == 0.9
    assert args.seed == 0
    assert args.device == "cuda"
    assert args.no_cache is False


def test_parse_args_overrides(parse_args):
    args = parse_args([
        "--checkpoint", "a.pt", "--tokenizer", "b.json", "--max-turns", "2",
        "--max-reply-tokens", "16", "--temperature", "0", "--top-k", "0", "--top-p", "1.0",
        "--seed", "5", "--device", "cpu", "--no-cache",
    ])

    assert args.max_turns == 2 and isinstance(args.max_turns, int)
    assert args.max_reply_tokens == 16 and isinstance(args.max_reply_tokens, int)
    assert args.temperature == 0.0 and isinstance(args.temperature, float)
    assert args.top_k == 0 and isinstance(args.top_k, int)
    assert args.top_p == 1.0 and isinstance(args.top_p, float)
    assert args.seed == 5 and isinstance(args.seed, int)
    assert args.device == "cpu"
    assert args.no_cache is True


@pytest.mark.parametrize(
    "argv",
    [[], ["--tokenizer", "t.json"], ["--checkpoint", "m.pt"]],
)
def test_parse_args_requires_checkpoint_and_tokenizer(parse_args, argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_parse_args_reads_sys_argv(monkeypatch, parse_args):
    monkeypatch.setattr(
        sys, "argv",
        ["chat.py", "--checkpoint", "m.pt", "--tokenizer", "t.json", "--seed", "9"],
    )

    args = parse_args()

    assert args.checkpoint == Path("m.pt")
    assert args.seed == 9


# --- import skryptu ---


def test_script_import_has_no_side_effects(monkeypatch, capsys):
    def zabronione_input(prompt=""):
        raise AssertionError("input nie powinno być wywołane przy imporcie")

    monkeypatch.setattr(builtins, "input", zabronione_input)

    ns = runpy.run_path(str(SCRIPT))

    assert capsys.readouterr().out == ""
    assert callable(ns["parse_args"])
    assert callable(ns["main"])


# --- bramka etapu 10 ---


def test_gate_model_does_not_speak_for_user(tmp_path):
    model, _ = zrob_atrape()
    tok = BPETokenizer([])

    odpowiedz = chat_reply(
        model, tok, [], "cześć", device="cpu", block_size=64, max_new_tokens=8, temperature=0,
    )
    z_historia = chat_reply(
        model, tok, [("a", "b")], "Zażółć gęślą",
        device="cpu", block_size=64, max_new_tokens=8, temperature=0,
    )

    assert odpowiedz == "ok"
    assert z_historia == "ok"


def test_gate_script_one_line_per_exchange(monkeypatch, tmp_path, capsys):
    model, _ = zrob_atrape()

    historia, _ = uruchom_z_atrapa(
        monkeypatch, tmp_path, model, baza(tmp_path),
        ["cześć", "jak się masz", "co robisz", "/koniec"],
    )

    wyjscie = linie(capsys)
    assert wyjscie == [WELCOME, "B: ok", "B: ok", "B: ok"]
    assert not any(linia.startswith("A:") for linia in wyjscie)
    assert historia == [("cześć", "ok"), ("jak się masz", "ok"), ("co robisz", "ok")]


def test_gate_history_fits_in_context(monkeypatch, tmp_path):
    model, dlugosci = zrob_atrape(block_size=32)
    wejscia = []
    for n in range(1, 16):
        if n % 2 == 0:
            wejscia.append(f"to jest bardzo długa linia numer {n} " * 3)
        elif n % 4 == 1:
            wejscia.append("hej")
        else:
            wejscia.append("a")
    wejscia.append("/koniec")

    historia, _ = uruchom_z_atrapa(monkeypatch, tmp_path, model, baza(tmp_path), wejscia)

    assert len(historia) == 15
    assert len(dlugosci) == 15 * 8
    assert all(dlugosci[i] <= 24 for i in range(0, len(dlugosci), 8))
    assert all(d <= 32 for d in dlugosci)


@pytest.mark.parametrize(
    "max_turns, oczekiwane",
    [("1", [8, 20, 20]), ("4", [8, 20, 32])],
)
def test_gate_multi_turn_memory_matches_build_context(
    monkeypatch, tmp_path, max_turns, oczekiwane
):
    model, dlugosci = zrob_atrape(block_size=64)

    historia, _ = uruchom_z_atrapa(
        monkeypatch, tmp_path, model, baza(tmp_path) + ["--max-turns", max_turns],
        ["aa", "bb", "cc", "/koniec"],
    )

    assert historia == [("aa", "ok"), ("bb", "ok"), ("cc", "ok")]
    pierwsze = [dlugosci[0], dlugosci[8], dlugosci[16]]
    assert pierwsze == oczekiwane
    tok = BPETokenizer([])
    pary = [
        ([], "aa"),
        ([("aa", "ok")], "bb"),
        ([("aa", "ok"), ("bb", "ok")], "cc"),
    ]
    z_build_context = [
        len(build_context(tok, h, t, 56, max_turns=int(max_turns))) for h, t in pary
    ]
    assert pierwsze == z_build_context


def test_gate_same_seed_same_replies(monkeypatch, capsys, pliki_gpt):
    argumenty = baza_gpt(pliki_gpt) + ["--seed", "3"]

    hist_a, tekst_a, hist_b, tekst_b = dwa_uruchomienia(
        monkeypatch, capsys, argumenty, argumenty, ["ala", "ma kota", "ala ma", "/koniec"],
    )

    assert hist_a == hist_b
    assert tekst_a == tekst_b


def test_gate_different_seed_different_output(monkeypatch, capsys, pliki_gpt):
    _, tekst_0, _, tekst_1 = dwa_uruchomienia(
        monkeypatch, capsys,
        baza_gpt(pliki_gpt) + ["--seed", "0"],
        baza_gpt(pliki_gpt) + ["--seed", "1"],
        ["ala", "ma kota", "ala ma", "kot", "/koniec"],
    )

    assert tekst_0 != tekst_1


def test_gate_no_cache_equals_cache(monkeypatch, capsys, pliki_gpt):
    hist_cache, tekst_cache, hist_bez, tekst_bez = dwa_uruchomienia(
        monkeypatch, capsys,
        baza_gpt(pliki_gpt) + ["--seed", "2"],
        baza_gpt(pliki_gpt) + ["--seed", "2", "--no-cache"],
        ["ala", "ma kota", "ala ma", "kot", "/koniec"],
    )

    assert hist_cache == hist_bez
    assert tekst_cache == tekst_bez


@pytest.mark.skipif(not torch.cuda.is_available(), reason="brak GPU")
def test_gate_cuda_no_cache_equals_cache(monkeypatch, capsys, pliki_gpt):
    hist_cache, tekst_cache, hist_bez, tekst_bez = dwa_uruchomienia(
        monkeypatch, capsys,
        baza_gpt(pliki_gpt, "cuda") + ["--seed", "2"],
        baza_gpt(pliki_gpt, "cuda") + ["--seed", "2", "--no-cache"],
        ["ala", "ma kota", "ala ma", "kot", "/koniec"],
    )

    assert hist_cache == hist_bez
    assert tekst_cache == tekst_bez


# --- pętla rozmowy ---


def test_blank_lines_are_skipped(monkeypatch, tmp_path, capsys):
    model, _ = zrob_atrape()

    historia, _ = uruchom_z_atrapa(
        monkeypatch, tmp_path, model, baza(tmp_path),
        ["", "   ", "\t", "  hej  ", "/koniec"],
    )

    assert historia == [("hej", "ok")]
    assert linie(capsys) == [WELCOME, "B: ok"]


@pytest.mark.parametrize("koniec", ["/koniec", " /koniec "])
def test_koniec_stops_loop(monkeypatch, tmp_path, koniec):
    model, _ = zrob_atrape()

    historia, wejscia = uruchom_z_atrapa(
        monkeypatch, tmp_path, model, baza(tmp_path), ["a", koniec, "b"],
    )

    assert historia == [("a", "ok")]
    assert wejscia.pozycja == 2


@pytest.mark.parametrize("przerwanie", [EOFError(), KeyboardInterrupt()])
def test_eof_and_keyboard_interrupt_end_loop(monkeypatch, tmp_path, capsys, przerwanie):
    model, _ = zrob_atrape()

    historia, _ = uruchom_z_atrapa(
        monkeypatch, tmp_path, model, baza(tmp_path), ["a", przerwanie],
    )
    assert historia == [("a", "ok")]

    capsys.readouterr()
    historia_pusta, _ = uruchom_z_atrapa(
        monkeypatch, tmp_path, model, baza(tmp_path), [przerwanie],
    )
    assert historia_pusta == []
    assert linie(capsys) == [WELCOME]


def test_empty_reply_prints_no_answer_and_skips_history(monkeypatch, tmp_path, capsys):
    model, _ = zrob_atrape(pusta=True)

    historia, _ = uruchom_z_atrapa(
        monkeypatch, tmp_path, model, baza(tmp_path), ["hej", "co tam", "/koniec"],
    )

    assert linie(capsys) == [WELCOME, "B: (brak odpowiedzi)", "B: (brak odpowiedzi)"]
    assert historia == []


def test_polish_characters_input(monkeypatch, tmp_path, capsys):
    model, _ = zrob_atrape()

    historia, _ = uruchom_z_atrapa(
        monkeypatch, tmp_path, model, baza(tmp_path), ["Zażółć gęślą jaźń", "/koniec"],
    )

    assert historia == [("Zażółć gęślą jaźń", "ok")]
    assert "B: ok" in linie(capsys)


def test_main_returns_list_of_str_pairs(monkeypatch, tmp_path):
    model, _ = zrob_atrape()

    historia, _ = uruchom_z_atrapa(
        monkeypatch, tmp_path, model, baza(tmp_path), ["a", "b", "/koniec"],
    )

    assert isinstance(historia, list)
    for wymiana in historia:
        assert isinstance(wymiana, tuple)
        assert len(wymiana) == 2
        assert all(isinstance(tekst, str) for tekst in wymiana)


# --- limit odpowiedzi ---


@pytest.mark.parametrize("limit", ["70", "64"])
def test_max_reply_tokens_must_be_below_block_size(monkeypatch, tmp_path, capsys, limit):
    model, _ = zrob_atrape(block_size=64)

    with pytest.raises(SystemExit) as exc:
        uruchom_z_atrapa(
            monkeypatch, tmp_path, model, baza(tmp_path) + ["--max-reply-tokens", limit],
            [AssertionError("input nie powinno być wywołane")],
        )

    assert isinstance(exc.value.code, str)
    assert limit in exc.value.code
    assert "64" in exc.value.code
    assert capsys.readouterr().out == ""


def test_max_reply_tokens_boundary_block_size_minus_one(monkeypatch, tmp_path, capsys):
    model, _ = zrob_atrape(block_size=64)

    historia, _ = uruchom_z_atrapa(
        monkeypatch, tmp_path, model, baza(tmp_path) + ["--max-reply-tokens", "63"],
        ["/koniec"],
    )

    assert historia == []
    assert linie(capsys) == [WELCOME]


# --- brakujące pliki i złe parametry ---


def test_missing_checkpoint_raises(monkeypatch, pliki_gpt):
    ns = runpy.run_path(str(SCRIPT))
    ustaw_wejscia(monkeypatch, [AssertionError("input nie powinno być wywołane")])
    argumenty = baza_gpt(pliki_gpt)
    argumenty[1] = str(pliki_gpt / "brak.pt")

    with pytest.raises(FileNotFoundError):
        ns["main"](argumenty)


def test_missing_tokenizer_raises(monkeypatch, pliki_gpt):
    ns = runpy.run_path(str(SCRIPT))
    ustaw_wejscia(monkeypatch, [AssertionError("input nie powinno być wywołane")])
    argumenty = baza_gpt(pliki_gpt)
    argumenty[3] = str(pliki_gpt / "brak.json")

    with pytest.raises(FileNotFoundError):
        ns["main"](argumenty)


@pytest.mark.parametrize(
    "dopisek",
    [
        ["--temperature", "-1"],
        ["--top-k", "-1"],
        ["--top-p", "0"],
        ["--top-p", "1.5"],
    ],
)
def test_invalid_sampling_args_raise_value_error(monkeypatch, tmp_path, dopisek):
    model, _ = zrob_atrape()

    with pytest.raises(ValueError):
        uruchom_z_atrapa(
            monkeypatch, tmp_path, model, baza(tmp_path) + dopisek, ["hej", "/koniec"],
        )


# --- przekazywanie argumentów do chat_reply ---


def uruchom_ze_szpiegiem(monkeypatch, tmp_path, model, argumenty, wejscia):
    """Uruchamia main z szpiegiem zamiast chat_reply; zwraca (wynik, lista wywołań)."""
    bajtowy_tokenizer(tmp_path)
    monkeypatch.setattr(nanowriter.train, "load_gpt", lambda path, device: model)
    ns = runpy.run_path(str(SCRIPT))
    wywolania = []

    def szpieg(*args, **kwargs):
        nazwane = ["model", "tokenizer", "history", "user_text"]
        polozone = dict(zip(nazwane, args, strict=False))
        polozone.update(kwargs)
        wywolania.append({
            **polozone,
            "history": list(polozone["history"]),
        })
        return "x"

    monkeypatch.setitem(ns["main"].__globals__, "chat_reply", szpieg)
    ustaw_wejscia(monkeypatch, wejscia)
    wynik = ns["main"](argumenty)
    return wynik, wywolania


def test_passes_arguments_to_chat_reply(monkeypatch, tmp_path):
    model, _ = zrob_atrape(block_size=64)

    wynik, wywolania = uruchom_ze_szpiegiem(
        monkeypatch, tmp_path, model,
        baza(tmp_path) + [
            "--top-k", "0", "--top-p", "1.0", "--max-turns", "2", "--max-reply-tokens", "16",
            "--temperature", "0.5", "--seed", "7",
        ],
        ["a", "b", "/koniec"],
    )

    assert wynik == [("a", "x"), ("b", "x")]
    assert len(wywolania) == 2
    assert wywolania[0]["history"] == []
    assert wywolania[0]["user_text"] == "a"
    assert wywolania[1]["history"] == [("a", "x")]
    assert wywolania[1]["user_text"] == "b"
    for wywolanie in wywolania:
        assert wywolanie["top_k"] is None
        assert wywolanie["top_p"] is None
        assert wywolanie["use_cache"] is False
        assert wywolanie["max_turns"] == 2
        assert wywolanie["max_new_tokens"] == 16
        assert wywolanie["temperature"] == 0.5
        assert wywolanie["block_size"] == 64
        assert str(wywolanie["device"]) == "cpu"
        assert wywolanie["model"] is model
        assert isinstance(wywolanie["tokenizer"], BPETokenizer)
    generator = wywolania[0]["generator"]
    assert isinstance(generator, torch.Generator)
    assert wywolania[1]["generator"] is generator
    assert generator.initial_seed() == 7


def test_passes_default_sampling_arguments(monkeypatch, tmp_path):
    model, _ = zrob_atrape(block_size=128)

    _, wywolania = uruchom_ze_szpiegiem(
        monkeypatch, tmp_path, model,
        [
            "--checkpoint", str(tmp_path / "gpt.pt"),
            "--tokenizer", str(tmp_path / "tok.json"),
            "--device", "cpu",
        ],
        ["a", "/koniec"],
    )

    assert len(wywolania) == 1
    wywolanie = wywolania[0]
    assert wywolanie["top_k"] == 40
    assert wywolanie["top_p"] == 0.9
    assert wywolanie["temperature"] == 0.7
    assert wywolanie["max_turns"] == 4
    assert wywolanie["max_new_tokens"] == 64
    assert wywolanie["use_cache"] is True
    assert wywolanie["generator"].initial_seed() == 0
    assert wywolanie["block_size"] == 128


# --- proces ---


def test_script_as_process_utf8(pliki_gpt):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}

    wynik = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--checkpoint", str(pliki_gpt / "gpt.pt"),
            "--tokenizer", str(pliki_gpt / "tok.json"),
            "--device", "cpu", "--max-reply-tokens", "8", "--temperature", "0",
        ],
        input="Zażółć gęślą\n/koniec\n".encode(),
        cwd=REPO, env=env, capture_output=True, timeout=120,
    )

    stderr = wynik.stderr.decode("utf-8", errors="replace")
    assert wynik.returncode == 0, stderr
    stdout = wynik.stdout.decode("utf-8")
    assert WELCOME in stdout
    assert "B: " in stdout
    assert "Traceback" not in stderr
