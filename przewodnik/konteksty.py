"""Konteksty materiałów: wczytanie prawdziwych modeli i danych, wspólny start skryptów."""

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from nanowriter.dataset import load_meta, load_tokens
from nanowriter.model import GPT
from nanowriter.tokenizer import BPETokenizer
from nanowriter.train import load_checkpoint, load_gpt

KORZEN: Path = Path(__file__).resolve().parents[1]
WYNIKI_DOMYSLNE: Path = KORZEN / "przewodnik" / "wyniki"

_MODEL_1M = Path("data/checkpoints/gpt.pt")
_TOKENIZER_1M = Path("data/tokens/tokenizer.json")
_TRAIN_BIN = Path("data/tokens/train.bin")
_VAL_BIN = Path("data/tokens/val.bin")
_LOG_1M = Path("data/checkpoints/train_log.jsonl")
_META_1M = Path("data/tokens/meta.json")
_MODEL_WSTEPNY = Path("data/rozmowca/ckpt-wstepny/gpt.pt")
_LOG_WSTEPNY = Path("data/rozmowca/ckpt-wstepny/train_log.jsonl")
_MODEL_ROZMOWY = Path("data/rozmowca/ckpt-rozmowy/gpt.pt")
_LOG_ROZMOWY = Path("data/rozmowca/ckpt-rozmowy/train_log.jsonl")
_TOKENIZER_10M = Path("data/rozmowca/tokeny-wstepne/tokenizer.json")
_META_WSTEPNY = Path("data/rozmowca/tokeny-wstepne/meta.json")
_META_ROZMOWY = Path("data/rozmowca/tokeny-rozmowy/meta.json")


class BrakPlikuError(FileNotFoundError):
    """Brak pliku potrzebnego do materiałów; komunikat po polsku mówi czego i co zrobić."""


@dataclass(frozen=True, eq=False)
class PrzebiegTreningu:
    """Opis jednego wytrenowanego modelu: liczby do materiałów 12 i 14 oraz ścieżka logu."""

    nazwa: str
    parametry: int
    kroki: int
    batch_size: int
    block_size: int
    log: Path
    tokenow_korpusu: int
    val_tokenow: int
    val_znakow: int


@dataclass(frozen=True, eq=False)
class KontekstI:
    """Wszystko, czego potrzebują materiały części I (model 1M)."""

    wyniki_dir: Path
    device: str
    seed: int
    model: GPT
    tokenizer: BPETokenizer
    dane_train: torch.Tensor
    dane_val: torch.Tensor
    log_treningu: Path
    kroki_uczenia: int = 800
    kroki_lr: int = 250
    kroki_bez_maski: int = 400
    batch_size: int = 32


@dataclass(frozen=True, eq=False)
class KontekstII:
    """Wszystko, czego potrzebują materiały części II (1M kontra 10M)."""

    wyniki_dir: Path
    device: str
    seed: int
    model_1m: GPT
    tokenizer_1m: BPETokenizer
    model_wstepny: GPT
    model_douczony: GPT
    tokenizer_10m: BPETokenizer
    przebieg_1m: PrzebiegTreningu
    przebieg_wstepny: PrzebiegTreningu
    przebieg_douczania: PrzebiegTreningu


def _sprawdz_pliki(korzen: Path, pliki: list[Path], naglowek: str, wskazowka: str) -> None:
    """Rzuca BrakPlikuError z listą wszystkich brakujących plików (ścieżki względem korzen)."""
    brakujace = [plik for plik in pliki if not (korzen / plik).is_file()]
    if not brakujace:
        return
    linie = [naglowek, *(f"  {plik.as_posix()}" for plik in brakujace), wskazowka]
    raise BrakPlikuError("\n".join(linie))


def _przebieg(nazwa: str, model: GPT, checkpoint: Path, log: Path, meta: Path) -> PrzebiegTreningu:
    """Składa PrzebiegTreningu z modelu, checkpointu, ścieżki logu i pliku meta."""
    dane = load_checkpoint(checkpoint)
    opis = load_meta(meta)
    return PrzebiegTreningu(
        nazwa=nazwa,
        parametry=sum(p.numel() for p in model.parameters()),
        kroki=int(dane["step"]),
        batch_size=int(dane["config"]["batch_size"]),
        block_size=int(dane["config"]["block_size"]),
        log=log,
        tokenow_korpusu=opis.train_tokens,
        val_tokenow=opis.val_tokens,
        val_znakow=opis.val_chars,
    )


def zaladuj_kontekst_i(
    device: str,
    wyniki_dir: Path,
    seed: int = 0,
    *,
    kroki_uczenia: int = 800,
    kroki_lr: int = 250,
    kroki_bez_maski: int = 400,
    korzen: Path = KORZEN,
) -> KontekstI:
    """Wczytuje model 1M, tokenizer, tokeny train/val i ścieżkę logu z katalogu korzen.

    Przykład: wszystkie pliki części I istnieją -> KontekstI z modelem w eval() na device
    Przykład: brak data/checkpoints/gpt.pt -> BrakPlikuError z tą ścieżką w komunikacie
    """
    _sprawdz_pliki(
        korzen,
        [_MODEL_1M, _TOKENIZER_1M, _TRAIN_BIN, _VAL_BIN, _LOG_1M],
        "Brak plików potrzebnych do materiałów części I:",
        "Wytrenuj model 1M: uv run python scripts/train_gpt.py (zob. README.md)",
    )
    return KontekstI(
        wyniki_dir=wyniki_dir,
        device=device,
        seed=seed,
        model=load_gpt(korzen / _MODEL_1M, device),
        tokenizer=BPETokenizer.load(korzen / _TOKENIZER_1M),
        dane_train=load_tokens(korzen / _TRAIN_BIN),
        dane_val=load_tokens(korzen / _VAL_BIN),
        log_treningu=korzen / _LOG_1M,
        kroki_uczenia=kroki_uczenia,
        kroki_lr=kroki_lr,
        kroki_bez_maski=kroki_bez_maski,
    )


def zaladuj_kontekst_ii(
    device: str, wyniki_dir: Path, seed: int = 0, *, korzen: Path = KORZEN
) -> KontekstII:
    """Wczytuje modele 1M, 10M wstępny i 10M douczony, oba tokenizery i opisy przebiegów.

    Przykład: wszystkie pliki części II istnieją -> KontekstII; przebieg_wstepny.kroki == 12000
    Przykład: brak któregokolwiek pliku -> BrakPlikuError z listą wszystkich brakujących
    """
    _sprawdz_pliki(
        korzen,
        [
            _MODEL_1M,
            _TOKENIZER_1M,
            _LOG_1M,
            _META_1M,
            _MODEL_WSTEPNY,
            _LOG_WSTEPNY,
            _MODEL_ROZMOWY,
            _LOG_ROZMOWY,
            _TOKENIZER_10M,
            _META_WSTEPNY,
            _META_ROZMOWY,
        ],
        "Brak plików potrzebnych do materiałów części II:",
        "Modele 10M: zob. ROZMOWCA.md",
    )
    model_1m = load_gpt(korzen / _MODEL_1M, device)
    model_wstepny = load_gpt(korzen / _MODEL_WSTEPNY, device)
    model_douczony = load_gpt(korzen / _MODEL_ROZMOWY, device)
    return KontekstII(
        wyniki_dir=wyniki_dir,
        device=device,
        seed=seed,
        model_1m=model_1m,
        tokenizer_1m=BPETokenizer.load(korzen / _TOKENIZER_1M),
        model_wstepny=model_wstepny,
        model_douczony=model_douczony,
        tokenizer_10m=BPETokenizer.load(korzen / _TOKENIZER_10M),
        przebieg_1m=_przebieg(
            "1M", model_1m, korzen / _MODEL_1M, korzen / _LOG_1M, korzen / _META_1M
        ),
        przebieg_wstepny=_przebieg(
            "10M wstępny",
            model_wstepny,
            korzen / _MODEL_WSTEPNY,
            korzen / _LOG_WSTEPNY,
            korzen / _META_WSTEPNY,
        ),
        przebieg_douczania=_przebieg(
            "10M douczony",
            model_douczony,
            korzen / _MODEL_ROZMOWY,
            korzen / _LOG_ROZMOWY,
            korzen / _META_ROZMOWY,
        ),
    )


def uruchom_samodzielnie(
    zrob: Callable[[Any], list[Path]],
    czesc: int,
    opis: str,
    argv: list[str] | None = None,
) -> int:
    """Wspólny start skryptu materiału: argumenty, kontekst, wywołanie zrob, lista plików.

    Argumenty: --device (domyślnie cuda, gdy dostępna, inaczej cpu), --wyniki-dir
    (domyślnie WYNIKI_DOMYSLNE), --seed (0); dla czesc == 1 także --kroki-uczenia (800),
    --kroki-lr (250), --kroki-bez-maski (400). czesc == 1 -> KontekstI, czesc == 2 -> KontekstII.
    Zwraca 0 po wypisaniu ścieżek utworzonych plików; BrakPlikuError -> komunikat na stderr, 1.
    Przykład: brak plików -> wypisuje komunikat o brakujących plikach na stderr i zwraca 1
    Przykład: czesc inna niż 1 albo 2 -> ValueError
    """
    if czesc not in (1, 2):
        raise ValueError(f"czesc musi być 1 albo 2: {czesc}")
    for strumien in (sys.stdout, sys.stderr):
        reconfigure = getattr(strumien, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=opis)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cpu albo cuda"
    )
    parser.add_argument("--wyniki-dir", type=Path, default=WYNIKI_DOMYSLNE)
    parser.add_argument("--seed", type=int, default=0)
    if czesc == 1:
        parser.add_argument("--kroki-uczenia", type=int, default=800)
        parser.add_argument("--kroki-lr", type=int, default=250)
        parser.add_argument("--kroki-bez-maski", type=int, default=400)
    args = parser.parse_args(argv)

    try:
        if czesc == 1:
            kontekst: Any = zaladuj_kontekst_i(
                args.device,
                args.wyniki_dir,
                args.seed,
                kroki_uczenia=args.kroki_uczenia,
                kroki_lr=args.kroki_lr,
                kroki_bez_maski=args.kroki_bez_maski,
                korzen=KORZEN,
            )
        else:
            kontekst = zaladuj_kontekst_ii(args.device, args.wyniki_dir, args.seed, korzen=KORZEN)
    except BrakPlikuError as blad:
        print(blad, file=sys.stderr)
        return 1

    pliki = zrob(kontekst)
    for plik in pliki:
        print(plik)
    print(f"Gotowe: {len(pliki)} plików w {args.wyniki_dir}")
    return 0
