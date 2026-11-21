"""Materiał 12: rozmiary modeli. Z czego składają się parametry, bity na znak, próbki tekstu."""

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nanowriter.train import bits_per_char
from przewodnik.konteksty import KontekstII, uruchom_samodzielnie
from przewodnik.rysowanie import KOLORY, nowa_figura, zapisz_png
from przewodnik.wspolne import probka_tekstu, wczytaj_log, zapisz_json

if TYPE_CHECKING:
    from torch import nn

    from nanowriter.model import GPT

PROMPT: str = "Wieczorem stary kot"
LICZBA_TOKENOW: int = 60
OPISY_KORPUSOW: tuple[str, str] = (
    "książki z Wolnych Lektur, słownik 1024",
    "napisy filmowe i książki, słownik 8192",
)
# Części podziału parametrów: (klucz w podziale, etykieta na wykresie, klucz w KOLORY).
_CZESCI: tuple[tuple[str, str, str], ...] = (
    ("embedding_tokenow", "embedding tokenów", "glowny"),
    ("embedding_pozycji", "embedding pozycji", "drugi"),
    ("uwaga", "uwaga", "dobry"),
    ("mlp", "MLP", "zly"),
    ("layernorm", "layernorm", "szary"),
)


def _liczba(modul: "nn.Module") -> int:
    """Zwraca sumę numel() po parametrach modułu (wagi w obrębie modułu liczone raz).

    Przykład: _liczba(model.ln_f) dla GPTConfig(d_model=128) -> 256 (waga i bias LayerNorm)
    """
    return sum(p.numel() for p in modul.parameters())


def _podzial(model: "GPT") -> dict[str, int]:
    """Zwraca liczby parametrów: embedding_tokenow, embedding_pozycji, uwaga, mlp, layernorm.

    uwaga i mlp to sumy po wszystkich blokach, layernorm to ln1 i ln2 wszystkich bloków plus
    ln_f. Suma wartości == sum(p.numel() for p in model.parameters()), inaczej RuntimeError.
    Przykład: model 1M (GPTConfig()) -> {"embedding_tokenow": 131072, "embedding_pozycji": 32768,
        "uwaga": 264192, "mlp": 526848, "layernorm": 2304}
    """
    podzial = {
        "embedding_tokenow": _liczba(model.tok_emb),
        "embedding_pozycji": _liczba(model.pos_emb),
        "uwaga": sum(_liczba(blok.attn) for blok in model.blocks),
        "mlp": sum(_liczba(blok.mlp) for blok in model.blocks),
        "layernorm": sum(_liczba(blok.ln1) + _liczba(blok.ln2) for blok in model.blocks)
        + _liczba(model.ln_f),
    }
    if sum(podzial.values()) != sum(p.numel() for p in model.parameters()):
        raise RuntimeError(
            f"podział parametrów nie sumuje się do liczby parametrów modelu: {podzial}"
        )
    return podzial


def _ostatnia_strata(log: Path) -> float:
    """Zwraca ostatnią stratę walidacyjną z logu treningu.

    Przykład: log z val_loss [5.0, 4.2] -> 4.2
    Przykład: log bez wierszy val_loss -> ValueError
    """
    straty = wczytaj_log(log)["val_loss"]
    if not straty:
        raise ValueError(f"w logu treningu brak wierszy val_loss: {log}")
    return float(straty[-1])


def _rysuj_rozmiary(modele: list[dict[str, Any]], sciezka: Path) -> Path:
    """Rysuje rozkład parametrów (słupki złożone) i bity na znak obok siebie; zapisuje PNG.

    Przykład: dwa modele w modele -> PNG z dwiema kolumnami wykresów, zwraca sciezka
    """
    fig, osie = nowa_figura(11, 5, 1, 2)
    ax_czesci, ax_bity = osie[0, 0], osie[0, 1]
    pozycje = list(range(len(modele)))

    dol = [0.0] * len(modele)
    for klucz, etykieta, kolor in _CZESCI:
        wysokosci = [m["podzial"][klucz] / 1e6 for m in modele]
        ax_czesci.bar(
            pozycje, wysokosci, width=0.6, bottom=dol, color=KOLORY[kolor], label=etykieta
        )
        for i, m in enumerate(modele):
            procent = 100 * m["podzial"][klucz] / m["parametry"]
            if procent > 4:
                ax_czesci.text(
                    i,
                    dol[i] + wysokosci[i] / 2,
                    f"{procent:.0f}%",
                    ha="center",
                    va="center",
                    color="white",
                )
        dol = [d + h for d, h in zip(dol, wysokosci, strict=True)]
    ax_czesci.set_xticks(
        pozycje, [f"{m['nazwa']}\n{m['parametry'] / 1e6:.2f} mln" for m in modele]
    )
    ax_czesci.set_ylabel("mln parametrów")
    ax_czesci.set_title("Z czego składają się parametry")
    ax_czesci.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

    bity = [m["bity_na_znak"] for m in modele]
    ax_bity.bar(pozycje, bity, width=0.6, color=[KOLORY["glowny"], KOLORY["drugi"]])
    for i, wartosc in enumerate(bity):
        ax_bity.text(i, wartosc, f"{wartosc:.2f}", ha="center", va="bottom")
    ax_bity.set_xticks(pozycje, [m["nazwa"] for m in modele])
    ax_bity.set_ylim(0, max(bity) * 1.15)
    ax_bity.set_ylabel("bity na znak")
    ax_bity.set_title("Bity na znak na własnym zbiorze walidacyjnym (mniej = lepiej)", wrap=True)

    return zapisz_png(fig, sciezka)


def _rysuj_probki(modele: list[dict[str, Any]], sciezka: Path) -> Path:
    """Rysuje obok siebie teksty próbek obu modeli z tym samym początkiem; zapisuje PNG.

    Przykład: dwa modele w modele -> PNG z dwiema osiami bez ramek, zwraca sciezka
    """
    fig, osie = nowa_figura(12, 5, 1, 2)
    for ax, m in zip(osie[0], modele, strict=True):
        ax.axis("off")
        ax.set_title(f"{m['nazwa']} ({m['parametry'] / 1e6:.1f} mln parametrów)")
        tekst = textwrap.fill(m["probka"].replace("\n", " / "), width=50)
        linie = tekst.split("\n")[:14]
        ax.text(
            0.0,
            1.0,
            "\n".join(linie),
            transform=ax.transAxes,
            ha="left",
            va="top",
            family="monospace",
            fontsize=10,
        )
    fig.suptitle(f"Ten sam początek „{PROMPT}” i ten sam seed")
    return zapisz_png(fig, sciezka)


def zrob(ctx: KontekstII) -> list[Path]:
    """Tworzy rozmiary.png, rozmiary_probki.png i rozmiary.json w ctx.wyniki_dir.

    Dwa modele: 1M (przebieg_1m, model_1m, tokenizer_1m) i 10M wstępny (przebieg_wstepny,
    model_wstepny, tokenizer_10m). Bity na znak z ostatniej straty walidacyjnej z logu
    przebiegu (bits_per_char z val_tokenow i val_znakow przebiegu).
    Zwraca [png, png_probki, json] (w tej kolejności).
    Przykład: zrob(maly_kontekst_ii(tmp)) -> 3 pliki; "modele" ma 2 elementy
    """
    wejscia = (
        (ctx.przebieg_1m, ctx.model_1m, ctx.tokenizer_1m, OPISY_KORPUSOW[0]),
        (ctx.przebieg_wstepny, ctx.model_wstepny, ctx.tokenizer_10m, OPISY_KORPUSOW[1]),
    )
    modele: list[dict[str, Any]] = []
    for przebieg, model, tokenizer, opis_korpusu in wejscia:
        val_loss = _ostatnia_strata(przebieg.log)
        modele.append(
            {
                "nazwa": przebieg.nazwa,
                "parametry": sum(p.numel() for p in model.parameters()),
                "podzial": _podzial(model),
                "val_loss": val_loss,
                "bity_na_znak": bits_per_char(
                    val_loss, przebieg.val_tokenow, przebieg.val_znakow
                ),
                "uwaga_korpus": (
                    "Strata z logu treningu na własnym zbiorze walidacyjnym tego modelu "
                    f"({opis_korpusu}); zbiory są różne, więc porównanie bitów na znak "
                    "jest przybliżone"
                ),
                "probka": probka_tekstu(
                    model, tokenizer, PROMPT, LICZBA_TOKENOW, device=ctx.device, seed=ctx.seed
                ),
            }
        )

    png = _rysuj_rozmiary(modele, ctx.wyniki_dir / "rozmiary.png")
    png_probki = _rysuj_probki(modele, ctx.wyniki_dir / "rozmiary_probki.png")
    json_plik = zapisz_json(
        {"material": "rozmiary", "prompt": PROMPT, "modele": modele},
        ctx.wyniki_dir / "rozmiary.json",
    )
    return [png, png_probki, json_plik]


def main(argv: list[str] | None = None) -> int:
    """Uruchamia materiał samodzielnie: uv run python -m przewodnik.rozmiary.

    Przykład: main(["--device", "cpu"]) -> 0 (albo 1 z komunikatem, gdy brak plików w data/)
    """
    return uruchom_samodzielnie(zrob, 2, "Materiał 12: rozmiary modeli", argv)


if __name__ == "__main__":
    raise SystemExit(main())
