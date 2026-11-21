"""Malutkie konteksty (CPU, kilka sekund) do prób, testów i bramki zadania."""

import json
from pathlib import Path

import torch

from nanowriter.model import GPT, GPTConfig
from nanowriter.tokenizer import BPETokenizer
from przewodnik.konteksty import KontekstI, KontekstII, PrzebiegTreningu

_TEKST = (
    "Wieczorem stary kot zasnął przy piecu. Cześć, jak minął ci dzień? "
    "Spokojnie, trochę pracy i długi spacer. A ja cały dzień remontowałem kuchnię. "
    "To musiało być męczące.\n"
) * 40
_VOCAB = 320
_BLOCK = 256
_VAL_TOKENOW = 1200


def _zapisz_log(sciezka: Path) -> Path:
    """Zapisuje sztuczny log treningu (5 wierszy train_loss, 2 val_loss) i zwraca ścieżkę."""
    sciezka.parent.mkdir(parents=True, exist_ok=True)
    wiersze: list[dict[str, float | int]] = []
    for krok in (0, 10, 20, 30, 40):
        wiersze.append({"step": krok, "train_loss": round(6.0 - 0.05 * krok, 3), "lr": 0.001})
        if krok in (20, 40):
            wiersze.append({"step": krok, "val_loss": round(6.2 - 0.05 * krok, 3)})
    with open(sciezka, "w", encoding="utf-8", newline="\n") as plik:
        for wiersz in wiersze:
            plik.write(json.dumps(wiersz) + "\n")
    return sciezka


def _tokenizer() -> BPETokenizer:
    """Trenuje malutki tokenizer BPE na tekście próbnym."""
    return BPETokenizer.train(_TEKST, _VOCAB)


def _model(tokenizer: BPETokenizer, seed: int) -> GPT:
    """Buduje malutki GPT (2 warstwy, d_model=32) z danego seeda, w eval() na CPU."""
    torch.manual_seed(seed)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size, block_size=_BLOCK, n_layer=2, n_head=2, d_model=32
    )
    return GPT(config).eval()


def maly_kontekst_i(katalog: Path, seed: int = 0) -> KontekstI:
    """Buduje malutki KontekstI: GPT 2 warstwy, d_model=32, kontekst 256, tokenizer 320 tokenów.

    Dane i log treningu powstają w katalog / "logi"; wyniki trafią do katalog / "wyniki".
    Przykład: maly_kontekst_i(tmp_path) -> device "cpu", kroki_uczenia 12, batch_size 4
    """
    tokenizer = _tokenizer()
    tokeny = torch.tensor(tokenizer.encode(_TEKST), dtype=torch.int64)
    log = _zapisz_log(katalog / "logi" / "train_log.jsonl")
    return KontekstI(
        wyniki_dir=katalog / "wyniki",
        device="cpu",
        seed=seed,
        model=_model(tokenizer, seed),
        tokenizer=tokenizer,
        dane_train=tokeny,
        dane_val=tokeny[:_VAL_TOKENOW],
        log_treningu=log,
        kroki_uczenia=12,
        kroki_lr=6,
        kroki_bez_maski=6,
        batch_size=4,
    )


def maly_kontekst_ii(katalog: Path, seed: int = 0) -> KontekstII:
    """Buduje malutki KontekstII: trzy różne modele tej samej wielkości i jeden tokenizer.

    Przykład: maly_kontekst_ii(tmp_path).model_wstepny is maly_kontekst_ii(tmp_path).model_1m
        -> False (różne obiekty z różnymi wagami)
    """
    tokenizer = _tokenizer()
    liczba_tokenow = len(tokenizer.encode(_TEKST))
    model_1m = _model(tokenizer, seed)
    model_wstepny = _model(tokenizer, seed + 1)
    model_douczony = _model(tokenizer, seed + 2)
    logi = katalog / "logi"

    def przebieg(nazwa: str, model: GPT, kroki: int, log: str) -> PrzebiegTreningu:
        return PrzebiegTreningu(
            nazwa=nazwa,
            parametry=sum(p.numel() for p in model.parameters()),
            kroki=kroki,
            batch_size=4,
            block_size=_BLOCK,
            log=_zapisz_log(logi / log),
            tokenow_korpusu=liczba_tokenow,
            val_tokenow=_VAL_TOKENOW,
            val_znakow=3600,
        )

    return KontekstII(
        wyniki_dir=katalog / "wyniki",
        device="cpu",
        seed=seed,
        model_1m=model_1m,
        tokenizer_1m=tokenizer,
        model_wstepny=model_wstepny,
        model_douczony=model_douczony,
        tokenizer_10m=tokenizer,
        przebieg_1m=przebieg("1M", model_1m, 40, "1m.jsonl"),
        przebieg_wstepny=przebieg("10M wstępny", model_wstepny, 40, "wstepny.jsonl"),
        przebieg_douczania=przebieg("10M douczony", model_douczony, 10, "douczanie.jsonl"),
    )
