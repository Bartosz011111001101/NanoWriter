"""Testy modułów przewodnik.konteksty i przewodnik.proba (CPU, własne drzewo plików w tmp_path)."""

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from nanowriter.dataset import TokenMeta, save_meta, write_tokens
from nanowriter.model import GPT, GPTConfig
from nanowriter.tokenizer import BPETokenizer
from nanowriter.train import TrainConfig, save_checkpoint
from przewodnik import konteksty
from przewodnik.konteksty import (
    KORZEN,
    WYNIKI_DOMYSLNE,
    BrakPlikuError,
    KontekstI,
    KontekstII,
    PrzebiegTreningu,
    uruchom_samodzielnie,
    zaladuj_kontekst_i,
    zaladuj_kontekst_ii,
)
from przewodnik.proba import maly_kontekst_i, maly_kontekst_ii
from przewodnik.wspolne import ROZMOWA_TEKST, ZDANIE

S = (
    "Wieczorem stary kot zasnął przy piecu. Cześć, jak minął ci dzień? "
    "Spokojnie, trochę pracy i długi spacer. A ja cały dzień remontowałem kuchnię. "
    "To musiało być męczące.\n"
)
TEKST = S * 40
TEKST_TOK = "Wieczorem stary kot zasnął przy piecu. Cześć, jak minął ci dzień? " * 40

PLIKI_I = [
    "data/checkpoints/gpt.pt",
    "data/tokens/tokenizer.json",
    "data/tokens/train.bin",
    "data/tokens/val.bin",
    "data/checkpoints/train_log.jsonl",
]
PLIKI_II = [
    "data/checkpoints/gpt.pt",
    "data/tokens/tokenizer.json",
    "data/checkpoints/train_log.jsonl",
    "data/tokens/meta.json",
    "data/rozmowca/ckpt-wstepny/gpt.pt",
    "data/rozmowca/ckpt-wstepny/train_log.jsonl",
    "data/rozmowca/ckpt-rozmowy/gpt.pt",
    "data/rozmowca/ckpt-rozmowy/train_log.jsonl",
    "data/rozmowca/tokeny-wstepne/tokenizer.json",
    "data/rozmowca/tokeny-wstepne/meta.json",
    "data/rozmowca/tokeny-rozmowy/meta.json",
]
WIERSZE_LOGU = [
    {"step": 0, "train_loss": 6.0, "lr": 0.001},
    {"step": 10, "train_loss": 5.5, "lr": 0.001},
    {"step": 20, "train_loss": 5.0, "lr": 0.001},
    {"step": 20, "val_loss": 5.2},
    {"step": 30, "train_loss": 4.5, "lr": 0.001},
    {"step": 40, "train_loss": 4.0, "lr": 0.001},
    {"step": 40, "val_loss": 4.2},
]


def _model(seed: int, **config: int) -> GPT:
    torch.manual_seed(seed)
    return GPT(GPTConfig(n_head=2, d_model=32, **config)).eval()


def _zapisz_checkpoint(sciezka: Path, model: GPT, krok: int, batch_size: int) -> None:
    sciezka.parent.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(
        batch_size=batch_size, block_size=model.config.block_size, max_steps=krok
    )
    save_checkpoint(
        sciezka,
        model,
        torch.optim.AdamW(model.parameters()),
        krok,
        torch.Generator().manual_seed(0),
        config,
    )


def _zapisz_log_jednowierszowy(sciezka: Path) -> None:
    sciezka.parent.mkdir(parents=True, exist_ok=True)
    wiersz = {"step": 0, "train_loss": 5.0, "lr": 0.001}
    sciezka.write_text(json.dumps(wiersz) + "\n", encoding="utf-8")


def zbuduj_drzewo(korzen: Path) -> dict[str, Any]:
    """Buduje w korzen malutkie data/ i zwraca oryginalne obiekty do porównań."""
    tok_1m = BPETokenizer.train(TEKST_TOK, 280)
    tok_10m = BPETokenizer.train(TEKST_TOK, 290)
    assert tok_1m.vocab_size == 280
    assert tok_10m.vocab_size == 290
    model_1m = _model(1, vocab_size=280, block_size=16, n_layer=1)
    model_wstepny = _model(2, vocab_size=290, block_size=24, n_layer=2)
    model_douczony = _model(3, vocab_size=290, block_size=32, n_layer=3)

    _zapisz_checkpoint(korzen / "data/checkpoints/gpt.pt", model_1m, 100, 4)
    _zapisz_log_jednowierszowy(korzen / "data/checkpoints/train_log.jsonl")

    tok_1m_sciezka = korzen / "data/tokens/tokenizer.json"
    tok_1m_sciezka.parent.mkdir(parents=True, exist_ok=True)
    tok_1m.save(tok_1m_sciezka)
    ids_train = tok_1m.encode("ala ma kota " * 5)
    ids_val = tok_1m.encode("ala ma ")
    write_tokens(ids_train, korzen / "data/tokens/train.bin")
    write_tokens(ids_val, korzen / "data/tokens/val.bin")
    save_meta(TokenMeta(280, "uint16", 1000, 3000, 100, 300), korzen / "data/tokens/meta.json")

    _zapisz_checkpoint(korzen / "data/rozmowca/ckpt-wstepny/gpt.pt", model_wstepny, 200, 8)
    _zapisz_log_jednowierszowy(korzen / "data/rozmowca/ckpt-wstepny/train_log.jsonl")
    _zapisz_checkpoint(korzen / "data/rozmowca/ckpt-rozmowy/gpt.pt", model_douczony, 50, 2)
    _zapisz_log_jednowierszowy(korzen / "data/rozmowca/ckpt-rozmowy/train_log.jsonl")

    tok_10m_sciezka = korzen / "data/rozmowca/tokeny-wstepne/tokenizer.json"
    tok_10m_sciezka.parent.mkdir(parents=True, exist_ok=True)
    tok_10m.save(tok_10m_sciezka)
    save_meta(
        TokenMeta(290, "uint16", 2000, 6000, 200, 600),
        korzen / "data/rozmowca/tokeny-wstepne/meta.json",
    )
    save_meta(
        TokenMeta(290, "uint16", 3000, 9000, 300, 900),
        korzen / "data/rozmowca/tokeny-rozmowy/meta.json",
    )
    return {
        "ids_train": ids_train,
        "ids_val": ids_val,
        "model_1m": model_1m,
        "model_wstepny": model_wstepny,
        "model_douczony": model_douczony,
        "tok_1m": tok_1m,
        "tok_10m": tok_10m,
    }


def _logity(model: GPT, idx: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(idx)


def _rowne_logity(a: GPT, b: GPT) -> bool:
    torch.manual_seed(0)
    idx = torch.randint(0, 280, (1, 8))
    return torch.allclose(_logity(a, idx), _logity(b, idx), atol=1e-5)


def _rowne_wagi(a: GPT, b: GPT) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def _liczba_parametrow(model: GPT) -> int:
    return sum(p.numel() for p in model.parameters())


def _wiersze_logu(sciezka: Path) -> list[dict[str, Any]]:
    tekst = sciezka.read_text(encoding="utf-8")
    return [json.loads(linia) for linia in tekst.splitlines() if linia.strip()]


def _sprawdz_wiersze_logu(sciezka: Path) -> None:
    wiersze = _wiersze_logu(sciezka)
    assert len(wiersze) == len(WIERSZE_LOGU)
    for wiersz, oczekiwany in zip(wiersze, WIERSZE_LOGU, strict=True):
        assert wiersz.keys() == oczekiwany.keys()
        for klucz, wartosc in oczekiwany.items():
            assert wiersz[klucz] == pytest.approx(wartosc, abs=1e-9)


def _pusty_katalog(tmp_path: Path) -> Path:
    pusty = tmp_path / "pusty"
    pusty.mkdir()
    return pusty


class _Atrapa:
    """Atrapa `zrob`: zapamiętuje kontekst i zwraca zadaną listę ścieżek."""

    def __init__(self, wynik: list[Path]) -> None:
        self.wynik = wynik
        self.ctx: Any = None
        self.wywolana = False

    def __call__(self, ctx: Any) -> list[Path]:
        self.wywolana = True
        self.ctx = ctx
        return self.wynik


# --- Testy: stałe i typy ---


def test_stale_sciezki():
    assert Path(konteksty.__file__).resolve().parents[1] == KORZEN
    assert (KORZEN / "pyproject.toml").is_file()
    assert (KORZEN / "przewodnik").is_dir()
    assert WYNIKI_DOMYSLNE == KORZEN / "przewodnik" / "wyniki"


def test_dataclassy_pola_domyslne_i_zamrozenie(tmp_path):
    assert [f.name for f in dataclasses.fields(PrzebiegTreningu)] == [
        "nazwa",
        "parametry",
        "kroki",
        "batch_size",
        "block_size",
        "log",
        "tokenow_korpusu",
        "val_tokenow",
        "val_znakow",
    ]
    pola_i = dataclasses.fields(KontekstI)
    assert [f.name for f in pola_i] == [
        "wyniki_dir",
        "device",
        "seed",
        "model",
        "tokenizer",
        "dane_train",
        "dane_val",
        "log_treningu",
        "kroki_uczenia",
        "kroki_lr",
        "kroki_bez_maski",
        "batch_size",
    ]
    domyslne = {f.name: f.default for f in pola_i}
    assert domyslne["kroki_uczenia"] == 800
    assert domyslne["kroki_lr"] == 250
    assert domyslne["kroki_bez_maski"] == 400
    assert domyslne["batch_size"] == 32
    assert [f.name for f in dataclasses.fields(KontekstII)] == [
        "wyniki_dir",
        "device",
        "seed",
        "model_1m",
        "tokenizer_1m",
        "model_wstepny",
        "model_douczony",
        "tokenizer_10m",
        "przebieg_1m",
        "przebieg_wstepny",
        "przebieg_douczania",
    ]

    ctx = maly_kontekst_i(tmp_path / "i")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.seed = 5  # type: ignore[misc]
    assert ctx == ctx  # noqa: PLR0124
    assert dataclasses.replace(ctx, seed=ctx.seed) != ctx
    hash(ctx)

    przebieg = maly_kontekst_ii(tmp_path / "ii").przebieg_1m
    with pytest.raises(dataclasses.FrozenInstanceError):
        przebieg.kroki = 1  # type: ignore[misc]
    assert issubclass(BrakPlikuError, FileNotFoundError)


# --- Testy: zaladuj_kontekst_i ---


def test_zaladuj_kontekst_i_wczytuje_dane(tmp_path):
    korzen = tmp_path / "repo"
    obiekty = zbuduj_drzewo(korzen)
    wyniki = tmp_path / "wyniki"
    ctx = zaladuj_kontekst_i("cpu", wyniki, korzen=korzen)

    assert isinstance(ctx, KontekstI)
    assert ctx.device == "cpu"
    assert ctx.seed == 0
    assert ctx.wyniki_dir == wyniki
    assert isinstance(ctx.model, GPT)
    assert ctx.model.training is False
    assert all(p.device.type == "cpu" for p in ctx.model.parameters())
    assert _rowne_logity(ctx.model, obiekty["model_1m"])
    assert ctx.tokenizer.vocab_size == 280
    assert ctx.tokenizer.encode("ala ma kota") == obiekty["tok_1m"].encode("ala ma kota")
    assert ctx.dane_train.dtype == torch.int64
    assert ctx.dane_train.dim() == 1
    assert ctx.dane_train.tolist() == obiekty["ids_train"]
    assert ctx.dane_val.dtype == torch.int64
    assert ctx.dane_val.dim() == 1
    assert ctx.dane_val.tolist() == obiekty["ids_val"]
    assert ctx.log_treningu == korzen / "data" / "checkpoints" / "train_log.jsonl"
    assert ctx.kroki_uczenia == 800
    assert ctx.kroki_lr == 250
    assert ctx.kroki_bez_maski == 400
    assert ctx.batch_size == 32


def test_zaladuj_kontekst_i_argumenty(tmp_path):
    korzen = tmp_path / "repo"
    zbuduj_drzewo(korzen)
    ctx = zaladuj_kontekst_i(
        "cpu",
        tmp_path / "wyniki",
        4,
        kroki_uczenia=7,
        kroki_lr=3,
        kroki_bez_maski=5,
        korzen=korzen,
    )
    assert ctx.seed == 4
    assert ctx.kroki_uczenia == 7
    assert ctx.kroki_lr == 3
    assert ctx.kroki_bez_maski == 5
    assert ctx.batch_size == 32


def test_zaladuj_kontekst_i_brak_wszystkich_plikow(tmp_path):
    pusty = _pusty_katalog(tmp_path)
    with pytest.raises(BrakPlikuError) as blad:
        zaladuj_kontekst_i("cpu", tmp_path / "w", korzen=pusty)
    assert isinstance(blad.value, FileNotFoundError)
    tekst = str(blad.value)
    assert "Brak plików potrzebnych do materiałów części I:" in tekst
    for sciezka in PLIKI_I:
        assert sciezka in tekst
    assert "scripts/train_gpt.py" in tekst


@pytest.mark.parametrize("brakujacy", PLIKI_I)
def test_zaladuj_kontekst_i_brak_jednego_pliku(tmp_path, brakujacy):
    korzen = tmp_path / "repo"
    zbuduj_drzewo(korzen)
    (korzen / brakujacy).unlink()
    with pytest.raises(BrakPlikuError) as blad:
        zaladuj_kontekst_i("cpu", tmp_path / "w", korzen=korzen)
    tekst = str(blad.value)
    assert brakujacy in tekst
    for inny in PLIKI_I:
        if inny != brakujacy:
            assert inny not in tekst


def test_zaladuj_kontekst_i_brak_dwoch_plikow(tmp_path):
    korzen = tmp_path / "repo"
    zbuduj_drzewo(korzen)
    (korzen / "data/tokens/val.bin").unlink()
    (korzen / "data/checkpoints/train_log.jsonl").unlink()
    with pytest.raises(BrakPlikuError) as blad:
        zaladuj_kontekst_i("cpu", tmp_path / "w", korzen=korzen)
    tekst = str(blad.value)
    assert "data/tokens/val.bin" in tekst
    assert "data/checkpoints/train_log.jsonl" in tekst
    assert "data/checkpoints/gpt.pt" not in tekst


# --- Testy: zaladuj_kontekst_ii ---


def test_zaladuj_kontekst_ii_wczytuje_modele_i_przebiegi(tmp_path):
    korzen = tmp_path / "repo"
    obiekty = zbuduj_drzewo(korzen)
    wyniki = tmp_path / "wyniki"
    ctx = zaladuj_kontekst_ii("cpu", wyniki, korzen=korzen)

    assert isinstance(ctx, KontekstII)
    assert ctx.device == "cpu"
    assert ctx.seed == 0
    assert ctx.wyniki_dir == wyniki
    for wczytany, oryginal in (
        (ctx.model_1m, obiekty["model_1m"]),
        (ctx.model_wstepny, obiekty["model_wstepny"]),
        (ctx.model_douczony, obiekty["model_douczony"]),
    ):
        assert isinstance(wczytany, GPT)
        assert wczytany.training is False
        assert _rowne_logity(wczytany, oryginal)
    assert ctx.model_1m.config.n_layer == 1
    assert ctx.model_wstepny.config.n_layer == 2
    assert ctx.model_douczony.config.n_layer == 3
    assert ctx.tokenizer_1m.vocab_size == 280
    assert ctx.tokenizer_10m.vocab_size == 290

    oczekiwane = [
        (
            ctx.przebieg_1m,
            ("1M", _liczba_parametrow(obiekty["model_1m"]), 100, 4, 16),
            korzen / "data" / "checkpoints" / "train_log.jsonl",
            (1000, 100, 300),
        ),
        (
            ctx.przebieg_wstepny,
            ("10M wstępny", _liczba_parametrow(obiekty["model_wstepny"]), 200, 8, 24),
            korzen / "data" / "rozmowca" / "ckpt-wstepny" / "train_log.jsonl",
            (2000, 200, 600),
        ),
        (
            ctx.przebieg_douczania,
            ("10M douczony", _liczba_parametrow(obiekty["model_douczony"]), 50, 2, 32),
            korzen / "data" / "rozmowca" / "ckpt-rozmowy" / "train_log.jsonl",
            (3000, 300, 900),
        ),
    ]
    for przebieg, glowne, log, liczby in oczekiwane:
        assert (
            przebieg.nazwa,
            przebieg.parametry,
            przebieg.kroki,
            przebieg.batch_size,
            przebieg.block_size,
        ) == glowne
        assert przebieg.log == log
        assert (przebieg.tokenow_korpusu, przebieg.val_tokenow, przebieg.val_znakow) == liczby


def test_zaladuj_kontekst_ii_nie_potrzebuje_tokenow_bin(tmp_path):
    korzen = tmp_path / "repo"
    zbuduj_drzewo(korzen)
    (korzen / "data/tokens/train.bin").unlink()
    (korzen / "data/tokens/val.bin").unlink()
    ctx = zaladuj_kontekst_ii("cpu", tmp_path / "wyniki", korzen=korzen)
    assert isinstance(ctx, KontekstII)


def test_zaladuj_kontekst_ii_brak_wszystkich_plikow(tmp_path):
    pusty = _pusty_katalog(tmp_path)
    with pytest.raises(BrakPlikuError) as blad:
        zaladuj_kontekst_ii("cpu", tmp_path / "w", korzen=pusty)
    tekst = str(blad.value)
    for sciezka in PLIKI_II:
        assert sciezka in tekst
    assert "data/rozmowca/ckpt-rozmowy/gpt.pt" in tekst
    assert "data/tokens/train.bin" not in tekst
    assert "data/tokens/val.bin" not in tekst
    assert "ROZMOWCA.md" in tekst


@pytest.mark.parametrize("brakujacy", PLIKI_II)
def test_zaladuj_kontekst_ii_brak_jednego_pliku(tmp_path, brakujacy):
    korzen = tmp_path / "repo"
    zbuduj_drzewo(korzen)
    (korzen / brakujacy).unlink()
    with pytest.raises(BrakPlikuError) as blad:
        zaladuj_kontekst_ii("cpu", tmp_path / "w", korzen=korzen)
    tekst = str(blad.value)
    assert brakujacy in tekst
    for inny in PLIKI_II:
        if inny != brakujacy:
            assert inny not in tekst


# --- Testy: uruchom_samodzielnie ---


def test_uruchom_samodzielnie_brak_plikow_czesc_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(konteksty, "KORZEN", _pusty_katalog(tmp_path))
    zrob = _Atrapa([])
    kod = uruchom_samodzielnie(
        zrob, 1, "opis", ["--device", "cpu", "--wyniki-dir", str(tmp_path / "w")]
    )
    assert kod == 1
    assert zrob.wywolana is False
    wyjscie = capsys.readouterr()
    assert wyjscie.out == ""
    assert "Brak plików potrzebnych do materiałów części I:" in wyjscie.err
    assert "data/checkpoints/gpt.pt" in wyjscie.err
    assert "Traceback" not in wyjscie.err


def test_uruchom_samodzielnie_brak_plikow_czesc_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(konteksty, "KORZEN", _pusty_katalog(tmp_path))
    zrob = _Atrapa([])
    kod = uruchom_samodzielnie(
        zrob, 2, "opis", ["--device", "cpu", "--wyniki-dir", str(tmp_path / "w")]
    )
    assert kod == 1
    assert zrob.wywolana is False
    wyjscie = capsys.readouterr()
    assert wyjscie.out == ""
    assert "ROZMOWCA.md" in wyjscie.err


def test_uruchom_samodzielnie_czesc_1_sukces(tmp_path, monkeypatch, capsys):
    korzen = tmp_path / "repo"
    zbuduj_drzewo(korzen)
    monkeypatch.setattr(konteksty, "KORZEN", korzen)
    zrob = _Atrapa([tmp_path / "a.png", tmp_path / "b.gif"])
    kod = uruchom_samodzielnie(
        zrob, 1, "opis", ["--device", "cpu", "--wyniki-dir", str(tmp_path / "w")]
    )
    assert kod == 0
    ctx = zrob.ctx
    assert isinstance(ctx, KontekstI)
    assert ctx.device == "cpu"
    assert ctx.seed == 0
    assert ctx.wyniki_dir == tmp_path / "w"
    assert ctx.kroki_uczenia == 800
    assert ctx.kroki_lr == 250
    assert ctx.kroki_bez_maski == 400
    assert ctx.model.training is False
    wyjscie = capsys.readouterr()
    linie = [linia for linia in wyjscie.out.splitlines() if linia.strip()]
    assert len(linie) == 3
    assert linie[0] == str(tmp_path / "a.png")
    assert linie[1] == str(tmp_path / "b.gif")
    assert linie[-1] == f"Gotowe: 2 plików w {tmp_path / 'w'}"
    assert wyjscie.err == ""


def test_uruchom_samodzielnie_argumenty_czesci_1(tmp_path, monkeypatch):
    korzen = tmp_path / "repo"
    zbuduj_drzewo(korzen)
    monkeypatch.setattr(konteksty, "KORZEN", korzen)
    zrob = _Atrapa([])
    argv = [
        "--device",
        "cpu",
        "--wyniki-dir",
        str(tmp_path / "w"),
        "--seed",
        "4",
        "--kroki-uczenia",
        "7",
        "--kroki-lr",
        "3",
        "--kroki-bez-maski",
        "5",
    ]
    assert uruchom_samodzielnie(zrob, 1, "opis", argv) == 0
    assert zrob.ctx.seed == 4
    assert zrob.ctx.kroki_uczenia == 7
    assert zrob.ctx.kroki_lr == 3
    assert zrob.ctx.kroki_bez_maski == 5


def test_uruchom_samodzielnie_czesc_2_sukces(tmp_path, monkeypatch, capsys):
    korzen = tmp_path / "repo"
    zbuduj_drzewo(korzen)
    monkeypatch.setattr(konteksty, "KORZEN", korzen)
    zrob = _Atrapa([tmp_path / "x.json"])
    argv = ["--device", "cpu", "--wyniki-dir", str(tmp_path / "w"), "--seed", "2"]
    assert uruchom_samodzielnie(zrob, 2, "opis", argv) == 0
    assert isinstance(zrob.ctx, KontekstII)
    assert zrob.ctx.seed == 2
    linie = [linia for linia in capsys.readouterr().out.splitlines() if linia.strip()]
    assert linie[-1] == f"Gotowe: 1 plików w {tmp_path / 'w'}"


def test_uruchom_samodzielnie_czesc_2_nie_przyjmuje_krokow(tmp_path, monkeypatch):
    korzen = tmp_path / "repo"
    zbuduj_drzewo(korzen)
    monkeypatch.setattr(konteksty, "KORZEN", korzen)
    zrob = _Atrapa([])
    with pytest.raises(SystemExit) as blad:
        uruchom_samodzielnie(zrob, 2, "opis", ["--device", "cpu", "--kroki-uczenia", "7"])
    assert blad.value.code == 2
    assert zrob.wywolana is False


def test_uruchom_samodzielnie_domyslne_device_i_wyniki(tmp_path, monkeypatch):
    korzen = tmp_path / "repo"
    zbuduj_drzewo(korzen)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(konteksty, "KORZEN", korzen)
    zrob = _Atrapa([])
    assert uruchom_samodzielnie(zrob, 1, "opis", []) == 0
    assert zrob.ctx.device == "cpu"
    assert zrob.ctx.wyniki_dir == WYNIKI_DOMYSLNE
    assert zrob.ctx.seed == 0


@pytest.mark.parametrize("czesc", [3, 0])
def test_uruchom_samodzielnie_zla_czesc(tmp_path, monkeypatch, czesc):
    korzen = tmp_path / "repo"
    zbuduj_drzewo(korzen)
    monkeypatch.setattr(konteksty, "KORZEN", korzen)
    with pytest.raises(ValueError):
        uruchom_samodzielnie(_Atrapa([]), czesc, "opis", ["--device", "cpu"])


def test_uruchom_samodzielnie_inne_wyjatki_nie_sa_przechwytywane(tmp_path, monkeypatch):
    korzen = tmp_path / "repo"
    zbuduj_drzewo(korzen)
    monkeypatch.setattr(konteksty, "KORZEN", korzen)

    def zrob(ctx: Any) -> list[Path]:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        uruchom_samodzielnie(zrob, 1, "opis", ["--device", "cpu"])


# --- Testy: maly_kontekst_i ---


def test_maly_kontekst_i_pola(tmp_path):
    ctx = maly_kontekst_i(tmp_path / "k")
    assert isinstance(ctx, KontekstI)
    assert ctx.device == "cpu"
    assert ctx.seed == 0
    assert ctx.kroki_uczenia == 12
    assert ctx.kroki_lr == 6
    assert ctx.kroki_bez_maski == 6
    assert ctx.batch_size == 4
    assert ctx.wyniki_dir == tmp_path / "k" / "wyniki"
    assert not ctx.wyniki_dir.exists()
    assert maly_kontekst_i(tmp_path / "k2", seed=5).seed == 5


def test_maly_kontekst_i_model_i_tokenizer(tmp_path):
    ctx = maly_kontekst_i(tmp_path / "k")
    assert ctx.tokenizer.vocab_size == 320
    config = ctx.model.config
    assert config.vocab_size == 320
    assert config.block_size == 256
    assert config.n_layer == 2
    assert config.n_head == 2
    assert config.d_model == 32
    assert ctx.model.training is False
    assert all(p.device.type == "cpu" for p in ctx.model.parameters())
    with torch.no_grad():
        wynik = ctx.model(ctx.dane_train[:16].unsqueeze(0))
    assert wynik.shape == (1, 16, 320)


def test_maly_kontekst_i_dane(tmp_path):
    ctx = maly_kontekst_i(tmp_path / "k")
    assert ctx.dane_train.dtype == torch.int64
    assert ctx.dane_train.dim() == 1
    assert ctx.dane_train.tolist() == ctx.tokenizer.encode(TEKST)
    assert int(ctx.dane_train.max()) < 320
    assert len(ctx.dane_val) == 1200
    assert ctx.dane_val.dtype == torch.int64
    assert ctx.dane_val.tolist() == ctx.dane_train[:1200].tolist()
    assert len(ctx.dane_train) > 1200
    for tekst in (ZDANIE, ROZMOWA_TEKST):
        dlugosc = len(ctx.tokenizer.encode(tekst))
        assert 1 <= dlugosc <= 256


def test_maly_kontekst_i_log(tmp_path):
    ctx = maly_kontekst_i(tmp_path / "k")
    assert ctx.log_treningu == tmp_path / "k" / "logi" / "train_log.jsonl"
    assert ctx.log_treningu.is_file()
    bajty = ctx.log_treningu.read_bytes()
    assert b"\r" not in bajty
    assert bajty.endswith(b"\n")
    _sprawdz_wiersze_logu(ctx.log_treningu)


def test_maly_kontekst_i_determinizm(tmp_path):
    a = maly_kontekst_i(tmp_path / "a", seed=0).model
    b = maly_kontekst_i(tmp_path / "b", seed=0).model
    assert _rowne_wagi(a, b)
    c = maly_kontekst_i(tmp_path / "c", seed=1).model
    assert not _rowne_wagi(a, c)


# --- Testy: maly_kontekst_ii ---


def test_maly_kontekst_ii_pola_i_modele(tmp_path):
    ctx = maly_kontekst_ii(tmp_path / "k")
    assert isinstance(ctx, KontekstII)
    assert ctx.device == "cpu"
    assert ctx.seed == 0
    assert ctx.wyniki_dir == tmp_path / "k" / "wyniki"
    assert not ctx.wyniki_dir.exists()
    assert ctx.tokenizer_1m is ctx.tokenizer_10m
    assert ctx.tokenizer_1m.vocab_size == 320
    assert ctx.model_wstepny is not ctx.model_1m
    assert ctx.model_douczony is not ctx.model_1m
    assert ctx.model_douczony is not ctx.model_wstepny
    modele = [ctx.model_1m, ctx.model_wstepny, ctx.model_douczony]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not torch.equal(modele[i].tok_emb.weight, modele[j].tok_emb.weight)
    for model in modele:
        assert model.config == ctx.model_1m.config
        assert model.config.vocab_size == 320
        assert model.config.block_size == 256
        assert model.config.n_layer == 2
        assert model.config.n_head == 2
        assert model.config.d_model == 32
        assert model.training is False
        assert all(p.device.type == "cpu" for p in model.parameters())


def test_maly_kontekst_ii_przebiegi_i_logi(tmp_path):
    ctx = maly_kontekst_ii(tmp_path / "k")
    przebiegi = [ctx.przebieg_1m, ctx.przebieg_wstepny, ctx.przebieg_douczania]
    modele = [ctx.model_1m, ctx.model_wstepny, ctx.model_douczony]
    assert [p.nazwa for p in przebiegi] == ["1M", "10M wstępny", "10M douczony"]
    assert [p.kroki for p in przebiegi] == [40, 40, 10]
    tokenow = len(ctx.tokenizer_1m.encode(TEKST))
    for przebieg, model in zip(przebiegi, modele, strict=True):
        assert przebieg.batch_size == 4
        assert przebieg.block_size == 256
        assert przebieg.val_tokenow == 1200
        assert przebieg.val_znakow == 3600
        assert przebieg.tokenow_korpusu == tokenow
        assert przebieg.parametry == _liczba_parametrow(model)
    logi = tmp_path / "k" / "logi"
    assert [p.log for p in przebiegi] == [
        logi / "1m.jsonl",
        logi / "wstepny.jsonl",
        logi / "douczanie.jsonl",
    ]
    for przebieg in przebiegi:
        assert przebieg.log.is_file()
        assert b"\r" not in przebieg.log.read_bytes()
        _sprawdz_wiersze_logu(przebieg.log)


def test_maly_kontekst_ii_seedy_modeli(tmp_path):
    assert _rowne_wagi(
        maly_kontekst_ii(tmp_path / "a", seed=0).model_wstepny,
        maly_kontekst_i(tmp_path / "b", seed=1).model,
    )
    assert _rowne_wagi(
        maly_kontekst_ii(tmp_path / "a2", seed=0).model_douczony,
        maly_kontekst_i(tmp_path / "b2", seed=2).model,
    )
    assert _rowne_wagi(
        maly_kontekst_ii(tmp_path / "a3", seed=0).model_1m,
        maly_kontekst_i(tmp_path / "b3", seed=0).model,
    )
    pierwszy = maly_kontekst_ii(tmp_path / "p", seed=3)
    drugi = maly_kontekst_ii(tmp_path / "q", seed=3)
    assert _rowne_wagi(pierwszy.model_1m, drugi.model_1m)
    assert _rowne_wagi(pierwszy.model_wstepny, drugi.model_wstepny)
    assert _rowne_wagi(pierwszy.model_douczony, drugi.model_douczony)
