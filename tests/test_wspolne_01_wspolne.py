import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import przewodnik
from nanowriter.model import GPT, GPTConfig
from nanowriter.sample import generate
from nanowriter.tokenizer import BPETokenizer
from przewodnik.wspolne import (
    PROMPTY_ROZMOWY,
    ROZMOWA_TEKST,
    ZDANIE,
    bajty_tokenu,
    entropia,
    etykieta_tokenu,
    probka_tekstu,
    top_k_rozklad,
    wczytaj_log,
    zakoduj_zdanie,
    zapisz_json,
)

KORZEN = Path(__file__).resolve().parents[1]

TOK_ABC = BPETokenizer([(97, 98), (256, 99)])
TOK_AB = BPETokenizer([(97, 98)])
TOK_PUSTY = BPETokenizer([])
TOK_WYTRENOWANY = BPETokenizer.train(
    "Wieczorem stary kot zasnął przy piecu. Zażółć gęślą jaźń. " * 20, 300
)


def _maly_model() -> GPT:
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=256, block_size=32, n_layer=2, n_head=2, d_model=32)
    return GPT(config).eval()


def _logity_przyklad() -> torch.Tensor:
    logity = torch.zeros(256)
    logity[97] = 2.0
    logity[98] = 1.0
    return logity


def test_pakiet_docstring_i_stale():
    assert przewodnik.__doc__ == (
        "Materiały do przewodnika po LLM: animacje, wykresy i liczby z prawdziwych modeli."
    )
    assert ZDANIE == "Wieczorem stary kot zasnął przy piecu."
    linie = ROZMOWA_TEKST.split("\n")
    assert len(linie) == 5
    assert linie[4] == ""
    assert ROZMOWA_TEKST.endswith("\n")
    assert linie[0] == "A: Cześć, jak minął ci dzień?"
    assert linie[0].startswith("A: ")
    assert linie[1].startswith("B: ")
    assert linie[2].startswith("A: ")
    assert linie[3].startswith("B: ")
    assert isinstance(PROMPTY_ROZMOWY, tuple)
    assert len(PROMPTY_ROZMOWY) == 4
    assert all(isinstance(p, str) and p for p in PROMPTY_ROZMOWY)


def test_wspolne_nie_importuje_matplotlib():
    wynik = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import przewodnik.wspolne; print('matplotlib' in sys.modules)",
        ],
        cwd=KORZEN,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert wynik.returncode == 0, wynik.stderr
    assert wynik.stdout.strip() == "False"


def test_bajty_tokenu_przyklady():
    assert bajty_tokenu(TOK_ABC, 97) == b"a"
    assert bajty_tokenu(TOK_ABC, 0) == b"\x00"
    assert bajty_tokenu(TOK_ABC, 255) == b"\xff"
    assert bajty_tokenu(TOK_ABC, 256) == b"ab"
    assert bajty_tokenu(TOK_ABC, 257) == b"abc"
    assert isinstance(bajty_tokenu(TOK_ABC, 257), bytes)
    for zle_id in (258, -1, 1000):
        with pytest.raises(ValueError):
            bajty_tokenu(TOK_ABC, zle_id)


@pytest.mark.parametrize(
    "tekst",
    ["", "Zażółć gęślą jaźń", "Wieczorem stary kot zasnął przy piecu.", "kot 🙂 kot", "\n\t"],
)
def test_bajty_tokenu_sklejone_rowne_bajtom_tekstu(tekst):
    sklejone = b"".join(
        bajty_tokenu(TOK_WYTRENOWANY, i) for i in TOK_WYTRENOWANY.encode(tekst)
    )
    assert sklejone == tekst.encode("utf-8")
    _sprawdz_wszystkie_id_niepuste()


def _sprawdz_wszystkie_id_niepuste() -> None:
    for i in range(TOK_WYTRENOWANY.vocab_size):
        bajty = bajty_tokenu(TOK_WYTRENOWANY, i)
        assert isinstance(bajty, bytes)
        assert len(bajty) >= 1
        if i >= 256:
            assert len(bajty) > 1


def test_etykieta_tokenu_przyklady():
    assert etykieta_tokenu(TOK_AB, 256) == "ab"
    assert etykieta_tokenu(TOK_AB, 97) == "a"
    assert etykieta_tokenu(TOK_AB, 197) == "\\xc5"
    assert len(etykieta_tokenu(TOK_AB, 197)) == 4
    assert etykieta_tokenu(TOK_AB, 10) == "\\n"
    assert len(etykieta_tokenu(TOK_AB, 10)) == 2
    assert etykieta_tokenu(TOK_AB, 9) == "\\t"
    assert etykieta_tokenu(TOK_AB, 1) == "\\x01"
    assert etykieta_tokenu(TOK_AB, 127) == "\\x7f"
    assert etykieta_tokenu(TOK_AB, 32) == " "
    assert etykieta_tokenu(BPETokenizer([(32, 107)]), 256) == " k"
    tok_zet = BPETokenizer([(197, 188)])
    assert etykieta_tokenu(tok_zet, 256) == "ż"
    assert etykieta_tokenu(tok_zet, 188) == "\\xbc"
    with pytest.raises(ValueError):
        etykieta_tokenu(TOK_AB, 257)


def test_etykieta_tokenu_bajty_128_255():
    for i in range(128, 256):
        assert etykieta_tokenu(TOK_PUSTY, i) == f"\\x{i:02x}"


def test_etykieta_tokenu_nigdy_nie_zwraca_znaku_zastepczego():
    for i in range(TOK_WYTRENOWANY.vocab_size):
        etykieta = etykieta_tokenu(TOK_WYTRENOWANY, i)
        assert isinstance(etykieta, str)
        assert etykieta
        assert "�" not in etykieta


def test_zakoduj_zdanie_przyklady():
    wynik = zakoduj_zdanie(TOK_PUSTY, "ab", 8)
    assert wynik.shape == (1, 2)
    assert wynik.dtype == torch.int64
    assert wynik.device.type == "cpu"
    assert wynik.tolist() == [[97, 98]]
    assert zakoduj_zdanie(TOK_AB, "ab", 8).tolist() == [[256]]
    assert zakoduj_zdanie(TOK_PUSTY, ZDANIE, 256).tolist() == [list(ZDANIE.encode("utf-8"))]
    assert zakoduj_zdanie(TOK_PUSTY, "abc", 3).tolist() == [[97, 98, 99]]
    assert zakoduj_zdanie(TOK_AB, "ab", 1).tolist() == [[256]]


def test_zakoduj_zdanie_bledy():
    with pytest.raises(ValueError):
        zakoduj_zdanie(TOK_PUSTY, "ab", 1)
    with pytest.raises(ValueError):
        zakoduj_zdanie(TOK_PUSTY, "abc", 2)
    with pytest.raises(ValueError):
        zakoduj_zdanie(TOK_PUSTY, "", 8)


def test_top_k_rozklad_przyklad_z_opisu():
    wynik = top_k_rozklad(TOK_PUSTY, _logity_przyklad(), 2)
    mianownik = 254 + math.exp(2) + math.exp(1)
    assert len(wynik) == 2
    assert set(wynik[0].keys()) == {"id", "token", "p"}
    assert wynik[0]["id"] == 97
    assert wynik[0]["token"] == "a"
    assert wynik[0]["p"] == pytest.approx(0.0280, abs=1e-4)
    assert wynik[0]["p"] == pytest.approx(math.exp(2) / mianownik, abs=1e-6)
    assert wynik[1]["id"] == 98
    assert wynik[1]["token"] == "b"
    assert wynik[1]["p"] == pytest.approx(0.0103, abs=1e-4)
    assert wynik[1]["p"] == pytest.approx(math.exp(1) / mianownik, abs=1e-6)
    assert type(wynik[0]["id"]) is int
    assert type(wynik[0]["p"]) is float
    assert type(wynik[0]["token"]) is str


def test_top_k_rozklad_dlugosc_kolejnosc_suma():
    logity = _logity_przyklad()
    wszystkie = top_k_rozklad(TOK_PUSTY, logity, 1000)
    assert len(wszystkie) == 256
    for a, b in zip(wszystkie, wszystkie[1:], strict=False):
        assert a["p"] >= b["p"]
    assert sum(e["p"] for e in wszystkie) == pytest.approx(1.0, abs=1e-5)
    ids = [e["id"] for e in wszystkie]
    assert len(set(ids)) == len(ids)

    jeden = top_k_rozklad(TOK_PUSTY, logity, 1)
    assert len(jeden) == 1
    assert jeden[0]["id"] == 97

    trzy = top_k_rozklad(TOK_PUSTY, logity, 3)
    assert len(trzy) == 3
    for a, b in zip(trzy, trzy[1:], strict=False):
        assert a["p"] >= b["p"]
    assert trzy[2]["p"] == pytest.approx(1 / (254 + math.exp(2) + math.exp(1)), abs=1e-6)

    for element in wszystkie:
        assert element["token"] == etykieta_tokenu(TOK_PUSTY, element["id"])


def test_top_k_rozklad_bledy():
    with pytest.raises(ValueError):
        top_k_rozklad(TOK_PUSTY, torch.zeros(256), 0)
    with pytest.raises(ValueError):
        top_k_rozklad(TOK_PUSTY, torch.zeros(256), -1)
    with pytest.raises(ValueError):
        top_k_rozklad(TOK_PUSTY, torch.zeros(2, 256), 1)
    with pytest.raises(ValueError):
        top_k_rozklad(TOK_PUSTY, torch.tensor(1.0), 1)


def test_entropia_przyklady():
    ln2 = entropia(torch.tensor([0.5, 0.5]))
    assert ln2.shape == ()
    assert ln2.item() == pytest.approx(0.6931, abs=1e-4)

    zero = entropia(torch.tensor([1.0, 0.0]))
    assert not torch.isnan(zero).any()
    assert zero.item() == pytest.approx(0.0, abs=1e-7)

    rowne = entropia(torch.full((3, 4), 0.25))
    assert rowne.shape == (3,)
    assert torch.allclose(rowne, torch.full((3,), 1.3863), atol=1e-4)

    mieszane = entropia(torch.tensor([[1.0, 0.0], [0.5, 0.5]]))
    assert mieszane.shape == (2,)
    assert torch.allclose(mieszane, torch.tensor([0.0, 0.6931]), atol=1e-4)

    cztery = entropia(torch.full((4,), 0.25))
    assert not torch.isnan(cztery).any()
    assert cztery.item() == pytest.approx(1.3863, abs=1e-4)


def _sprawdz_domyslne_parametry(model: GPT) -> None:
    wynik = probka_tekstu(model, TOK_PUSTY, "ala ma", 5, device="cpu", seed=3)
    idx = torch.tensor([TOK_PUSTY.encode("ala ma")])
    oczekiwane_ids = generate(
        model,
        idx,
        5,
        generator=torch.Generator(device="cpu").manual_seed(3),
        block_size=32,
        temperature=0.8,
        top_k=40,
    )[0].tolist()
    assert wynik == TOK_PUSTY.decode(oczekiwane_ids)
    assert wynik.startswith("ala ma")



@pytest.mark.parametrize("temperature, top_k", [(0.8, 40), (1.0, None)])
def test_probka_tekstu_zgodna_z_generate(temperature, top_k):
    model = _maly_model()
    _sprawdz_domyslne_parametry(model)
    wynik = probka_tekstu(
        model,
        TOK_PUSTY,
        "ala ma",
        5,
        device="cpu",
        seed=3,
        temperature=temperature,
        top_k=top_k,
    )
    idx = torch.tensor([TOK_PUSTY.encode("ala ma")])
    oczekiwane_ids = generate(
        model,
        idx,
        5,
        generator=torch.Generator(device="cpu").manual_seed(3),
        block_size=32,
        temperature=temperature,
        top_k=top_k,
    )[0].tolist()
    assert isinstance(wynik, str)
    assert wynik == TOK_PUSTY.decode(oczekiwane_ids)
    assert wynik.startswith("ala ma")


def test_probka_tekstu_powtarzalnosc_i_seed():
    model = _maly_model()
    pierwszy = probka_tekstu(model, TOK_PUSTY, "ala ma", 20, device="cpu", seed=7)
    drugi = probka_tekstu(model, TOK_PUSTY, "ala ma", 20, device="cpu", seed=7)
    assert pierwszy == drugi
    wyniki = {
        probka_tekstu(model, TOK_PUSTY, "ala ma", 20, device="cpu", seed=seed)
        for seed in range(5)
    }
    assert len(wyniki) > 1


def test_probka_tekstu_tryb_modelu_bez_zmian():
    model = _maly_model()
    assert model.training is False
    probka_tekstu(model, TOK_PUSTY, "ala ma", 3, device="cpu", seed=0)
    assert model.training is False

    model.train()
    probka_tekstu(model, TOK_PUSTY, "ala ma", 3, device="cpu", seed=0)
    assert model.training is True


def test_probka_tekstu_bledy_i_cisza(capsys):
    model = _maly_model()
    with pytest.raises(ValueError):
        probka_tekstu(model, TOK_PUSTY, "ala ma", 0, device="cpu", seed=0)
    with pytest.raises(ValueError):
        probka_tekstu(model, TOK_PUSTY, "ala ma", -1, device="cpu", seed=0)
    with pytest.raises(ValueError):
        probka_tekstu(model, TOK_PUSTY, "", 5, device="cpu", seed=0)
    capsys.readouterr()
    probka_tekstu(model, TOK_PUSTY, "ala ma", 5, device="cpu", seed=0)
    przechwycone = capsys.readouterr()
    assert przechwycone.out == ""
    assert przechwycone.err == ""


def test_zapisz_json_dokladne_bajty(tmp_path):
    sciezka = tmp_path / "x.json"
    wynik = zapisz_json({"a": float("nan"), "b": torch.tensor([1.0, 2.0])}, sciezka)
    assert wynik is sciezka
    assert sciezka.read_bytes() == b'{\n  "a": null,\n  "b": [\n    1.0,\n    2.0\n  ]\n}\n'


@pytest.mark.parametrize(
    "dane, oczekiwane",
    [
        ({"t": torch.tensor([[1, 2], [3, 4]])}, {"t": [[1, 2], [3, 4]]}),
        ({"p": Path("a") / "b"}, {"p": str(Path("a") / "b")}),
        ({"k": (1, 2, 3)}, {"k": [1, 2, 3]}),
        (
            {"x": float("inf"), "y": float("-inf"), "z": float("nan")},
            {"x": None, "y": None, "z": None},
        ),
        ({"t": torch.tensor([float("nan"), 1.0])}, {"t": [None, 1.0]}),
        ({1: "a"}, {"1": "a"}),
        (
            {"i": 3, "f": 0.5, "s": "t", "b": True, "n": None},
            {"i": 3, "f": 0.5, "s": "t", "b": True, "n": None},
        ),
        ({"a": [{"b": (float("nan"), 2)}]}, {"a": [{"b": [None, 2]}]}),
    ],
)
def test_zapisz_json_konwersje(tmp_path, dane, oczekiwane):
    sciezka = tmp_path / "x.json"
    zapisz_json(dane, sciezka)
    wczytane = json.loads(sciezka.read_text(encoding="utf-8"))
    assert wczytane == oczekiwane
    if "b" in dane:
        assert b'"b": true' in sciezka.read_bytes()


def test_zapisz_json_polskie_znaki_i_katalogi(tmp_path):
    tekst = "Zażółć gęślą jaźń 🙂"
    sciezka = tmp_path / "a" / "b" / "x.json"
    wynik = zapisz_json({"t": tekst}, sciezka)
    assert wynik == sciezka
    assert sciezka.exists()
    bajty = sciezka.read_bytes()
    assert tekst.encode("utf-8") in bajty
    assert b"\\u" not in bajty
    assert bajty.endswith(b"\n")
    assert b"\r" not in bajty

    zapisz_json({"t": "nowa treść"}, sciezka)
    assert json.loads(sciezka.read_text(encoding="utf-8")) == {"t": "nowa treść"}


def test_wczytaj_log_przyklad(tmp_path):
    sciezka = tmp_path / "log.jsonl"
    sciezka.write_text(
        '{"step": 0, "train_loss": 6.9, "lr": 1e-05}\n'
        '{"step": 100, "train_loss": 5.2, "lr": 0.001}\n'
        '{"step": 100, "val_loss": 5.5}\n',
        encoding="utf-8",
    )
    wynik = wczytaj_log(sciezka)
    assert set(wynik.keys()) == {"kroki", "train_loss", "lr", "val_kroki", "val_loss"}
    assert wynik == {
        "kroki": [0, 100],
        "train_loss": [6.9, 5.2],
        "lr": [1e-05, 0.001],
        "val_kroki": [100],
        "val_loss": [5.5],
    }


def test_wczytaj_log_brzegowe(tmp_path):
    klucze = {"kroki", "train_loss", "lr", "val_kroki", "val_loss"}

    pusty = tmp_path / "pusty.jsonl"
    pusty.write_text("", encoding="utf-8")
    wynik_pusty = wczytaj_log(pusty)
    assert set(wynik_pusty.keys()) == klucze
    assert all(wartosc == [] for wartosc in wynik_pusty.values())

    mieszany = tmp_path / "mieszany.jsonl"
    mieszany.write_text(
        "\n"
        '{"step": 0, "train_loss": 3.0, "lr": 0.1}\n'
        "   \n"
        '{"step": 5, "coś": 1}\n'
        '{"step": 5, "val_loss": 2.0}\n'
        "\n",
        encoding="utf-8",
    )
    assert wczytaj_log(mieszany) == {
        "kroki": [0],
        "train_loss": [3.0],
        "lr": [0.1],
        "val_kroki": [5],
        "val_loss": [2.0],
    }

    tylko_val = tmp_path / "val.jsonl"
    tylko_val.write_text(
        '{"step": 10, "val_loss": 4.0}\n{"step": 20, "val_loss": 3.5}\n',
        encoding="utf-8",
    )
    wynik_val = wczytaj_log(tylko_val)
    assert wynik_val["kroki"] == []
    assert wynik_val["train_loss"] == []
    assert wynik_val["lr"] == []
    assert wynik_val["val_kroki"] == [10, 20]
    assert wynik_val["val_loss"] == [4.0, 3.5]

    with pytest.raises(FileNotFoundError):
        wczytaj_log(tmp_path / "nie_ma.jsonl")
