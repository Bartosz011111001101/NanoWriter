"""Materiał 16: krzywe treningu wstępnego i douczania oraz odpowiedzi obu modeli."""

import math
import textwrap
from pathlib import Path

import torch

from nanowriter.sample import chat_reply
from przewodnik.konteksty import KontekstII, uruchom_samodzielnie
from przewodnik.rysowanie import KOLORY, nowa_figura, zapisz_png
from przewodnik.wspolne import PROMPTY_ROZMOWY, wczytaj_log, zapisz_json

MAX_NOWYCH_TOKENOW: int = 40


def _odpowiedz(ctx: KontekstII, model: torch.nn.Module, block_size: int, prompt: str) -> str:
    """Zwraca odpowiedź modelu na jedno zdanie bez historii; pusta -> "(brak odpowiedzi)".

    Każde wywołanie dostaje świeży generator z seedem ctx.seed.
    Przykład: _odpowiedz(ctx, ctx.model_wstepny, 256, "cześć") -> napis niepusty
    """
    generator = torch.Generator(device=ctx.device)
    generator.manual_seed(ctx.seed)
    tekst = chat_reply(
        model,
        ctx.tokenizer_10m,
        [],
        prompt,
        device=ctx.device,
        block_size=block_size,
        max_new_tokens=MAX_NOWYCH_TOKENOW,
        generator=generator,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
    )
    return tekst or "(brak odpowiedzi)"


def _krzywe_do_json(log: dict[str, list[float]]) -> dict[str, list[float]]:
    """Wybiera z wczytanego logu cztery listy potrzebne do JSON (bez lr).

    Przykład: _krzywe_do_json({"kroki": [0], "train_loss": [6.9], "lr": [1e-5],
        "val_kroki": [], "val_loss": []}) -> {"kroki": [0], "train_loss": [6.9],
        "val_kroki": [], "val_loss": []}
    """
    return {
        "kroki": log["kroki"],
        "train_loss": log["train_loss"],
        "val_kroki": log["val_kroki"],
        "val_loss": log["val_loss"],
    }


def _rysuj_krzywe(
    pretrening: dict[str, list[float]],
    douczanie: dict[str, list[float]],
    vocab: int,
    sciezka: Path,
) -> Path:
    """Rysuje dwa panele (trening wstępny, douczanie) i zapisuje PNG; zwraca sciezka.

    Przykład: _rysuj_krzywe(log_a, log_b, 1024, Path("w/k.png")) -> Path("w/k.png")
    """
    fig, osie = nowa_figura(12, 4.8, 1, 2)
    panele = (
        (
            osie[0, 0],
            pretrening,
            "Trening wstępny (napisy i książki)",
            True,
        ),
        (
            osie[0, 1],
            douczanie,
            "Douczanie na rozmowach (strata tylko na wypowiedziach B)",
            False,
        ),
    )
    for ax, log, tytul, z_losowym in panele:
        ax.plot(
            log["kroki"], log["train_loss"], color=KOLORY["glowny"], label="strata treningowa"
        )
        if log["val_kroki"]:
            ax.plot(
                log["val_kroki"],
                log["val_loss"],
                color=KOLORY["drugi"],
                marker="o",
                label="strata walidacyjna",
            )
        if z_losowym:
            ax.axhline(
                math.log(vocab),
                color=KOLORY["szary"],
                linestyle="--",
                label="losowe zgadywanie = ln(V)",
            )
        ax.set_title(tytul)
        ax.set_xlabel("krok treningu")
        ax.set_ylabel("strata")
        ax.legend()
    return zapisz_png(fig, sciezka)


def _rysuj_odpowiedzi(odpowiedzi: list[dict[str, str]], sciezka: Path) -> Path:
    """Rysuje tabelę: zdanie, odpowiedź modelu wstępnego i douczonego; zwraca sciezka.

    Każdy wiersz ma linię pod sobą; nagłówek to pierwszy wiersz tabeli.
    Przykład: _rysuj_odpowiedzi([{"prompt": "a", "wstepny": "b", "douczony": "c"}],
        Path("w/o.png")) -> Path("w/o.png")
    """
    wiersz_h = 1.0 / (len(odpowiedzi) + 1)
    fig, osie = nowa_figura(13, 1.5 * len(odpowiedzi) + 1.6)
    ax = osie[0, 0]
    kolumny = (0.0, 0.24, 0.62)
    naglowki = ("Ty", "Model wstępny", "Model douczony")
    for x, naglowek in zip(kolumny, naglowki, strict=True):
        ax.text(x, 1.0, naglowek, ha="left", va="top", fontsize=10, fontweight="bold")
    for r, wiersz in enumerate(odpowiedzi, start=1):
        gora = 1.0 - r * wiersz_h
        teksty = (
            textwrap.fill(wiersz["prompt"], width=26),
            textwrap.fill(wiersz["wstepny"], width=42),
            textwrap.fill(wiersz["douczony"], width=42),
        )
        for x, tekst in zip(kolumny, teksty, strict=True):
            ax.text(x, gora - 0.15 * wiersz_h, tekst, ha="left", va="top", fontsize=9)
    for r in range(len(odpowiedzi) + 1):
        dol = 1.0 - (r + 1) * wiersz_h
        ax.plot([0.0, 1.0], [dol, dol], color=KOLORY["szary"], linewidth=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle("Te same zdania, ten sam seed (temperatura 0.7, top-k 40, top-p 0.9)")
    return zapisz_png(fig, sciezka)


def zrob(ctx: KontekstII) -> list[Path]:
    """Tworzy krzywe_pretrening_douczanie.png, odpowiedzi.png i krzywe_i_odpowiedzi.json.

    Krzywe z logów przebieg_wstepny.log i przebieg_douczania.log. Odpowiedzi: dla każdego
    zdania z PROMPTY_ROZMOWY odpowiedź modelu wstępnego i douczonego (chat_reply, bez
    historii, temperatura 0.7, top-k 40, top-p 0.9, generator z seedem ctx.seed, osobny
    świeży generator dla każdej odpowiedzi). Pusta odpowiedź to "(brak odpowiedzi)".
    Zwraca [png_krzywe, png_odpowiedzi, json] (w tej kolejności).
    Przykład: zrob(maly_kontekst_ii(tmp)) -> 3 pliki; "odpowiedzi" ma 4 elementy
    """
    pretrening = wczytaj_log(ctx.przebieg_wstepny.log)
    douczanie = wczytaj_log(ctx.przebieg_douczania.log)
    if not pretrening["train_loss"]:
        raise ValueError(
            f"log treningu wstępnego bez wierszy train_loss: {ctx.przebieg_wstepny.log}"
        )
    if not douczanie["train_loss"]:
        raise ValueError(f"log douczania bez wierszy train_loss: {ctx.przebieg_douczania.log}")

    blok_wstepny = ctx.model_wstepny.config.block_size
    blok_douczony = ctx.model_douczony.config.block_size
    odpowiedzi = [
        {
            "prompt": prompt,
            "wstepny": _odpowiedz(ctx, ctx.model_wstepny, blok_wstepny, prompt),
            "douczony": _odpowiedz(ctx, ctx.model_douczony, blok_douczony, prompt),
        }
        for prompt in PROMPTY_ROZMOWY
    ]

    png_krzywe = _rysuj_krzywe(
        pretrening,
        douczanie,
        ctx.model_wstepny.config.vocab_size,
        ctx.wyniki_dir / "krzywe_pretrening_douczanie.png",
    )
    png_odpowiedzi = _rysuj_odpowiedzi(odpowiedzi, ctx.wyniki_dir / "odpowiedzi.png")
    json_plik = zapisz_json(
        {
            "material": "krzywe_i_odpowiedzi",
            "seed": ctx.seed,
            "pretrening": _krzywe_do_json(pretrening),
            "douczanie": _krzywe_do_json(douczanie),
            "odpowiedzi": odpowiedzi,
        },
        ctx.wyniki_dir / "krzywe_i_odpowiedzi.json",
    )
    return [png_krzywe, png_odpowiedzi, json_plik]


def main(argv: list[str] | None = None) -> int:
    """Uruchamia materiał samodzielnie: uv run python -m przewodnik.krzywe_i_odpowiedzi.

    Przykład: main(["--device", "cpu"]) -> 0 (albo 1 z komunikatem, gdy brak plików w data/)
    """
    return uruchom_samodzielnie(zrob, 2, "Materiał 16: krzywe i odpowiedzi modeli", argv)


if __name__ == "__main__":
    raise SystemExit(main())
