"""Materiał 6: jak model uczy się pisać. Krótki trening od zera z próbkami tekstu po drodze."""

import math
import textwrap
from pathlib import Path
from typing import Any

from przewodnik.konteksty import KontekstI, uruchom_samodzielnie
from przewodnik.krotki_trening import krotki_trening, nowy_model
from przewodnik.rysowanie import KOLORY, nowa_figura, zapisz_gif
from przewodnik.wspolne import probka_tekstu, zapisz_json

PROMPT: str = "Wieczorem stary kot"
LICZBA_TOKENOW: int = 60
LR: float = 2e-3


def zrob(ctx: KontekstI) -> list[Path]:
    """Tworzy uczenie.gif, uczenie.png i uczenie.json w ctx.wyniki_dir.

    Trenuje nowy model o konfiguracji ctx.model.config przez ctx.kroki_uczenia kroków. Próbka
    tekstu z PROMPT (ten sam seed za każdym razem) przed treningiem i co ctx.kroki_uczenia // 20
    kroków (co najmniej 1) po nim; klatka animacji = próbka plus krzywa straty do tego kroku.
    Zwraca [gif, png, json].
    Przykład: zrob(maly_kontekst_i(tmp)) -> 3 pliki; "probki"[0] to krok 0 ze stratą None
    """
    model = nowy_model(ctx.model.config, ctx.seed, ctx.device)
    co_ile = max(1, ctx.kroki_uczenia // 20)

    def _probka() -> str:
        return probka_tekstu(
            model, ctx.tokenizer, PROMPT, LICZBA_TOKENOW, device=ctx.device, seed=ctx.seed
        )

    probki: list[dict[str, Any]] = [{"krok": 0, "strata": None, "tekst": _probka()}]

    def _na_krok(krok: int, strata: float) -> None:
        probki.append({"krok": krok + 1, "strata": strata, "tekst": _probka()})

    wynik = krotki_trening(
        model,
        ctx.dane_train,
        ctx.dane_val,
        kroki=ctx.kroki_uczenia,
        lr=LR,
        batch_size=ctx.batch_size,
        device=ctx.device,
        seed=ctx.seed,
        co_ile=co_ile,
        na_krok=_na_krok,
    )

    ln_v = math.log(ctx.model.config.vocab_size)
    fig, osie = nowa_figura(11, 4.8, 1, 2)
    ax_strata, ax_tekst = osie[0, 0], osie[0, 1]

    def _rysuj_klatke(i: int) -> None:
        ax_strata.clear()
        ax_tekst.clear()
        p = probki[i]
        krok = p["krok"]
        if krok > 0:
            ax_strata.plot(
                list(range(1, krok + 1)), wynik.train_loss[:krok], color=KOLORY["szary"], lw=0.8
            )
            ax_strata.plot(krok, p["strata"], "o", color=KOLORY["drugi"], markersize=9)
        ax_strata.axhline(ln_v, color=KOLORY["glowny"], linestyle="--", lw=1)
        ax_strata.text(
            ctx.kroki_uczenia, ln_v, "losowe zgadywanie = ln(V)", ha="right", va="bottom"
        )
        ax_strata.set_xlim(0, ctx.kroki_uczenia)
        ax_strata.set_ylim(0, 1.05 * ln_v)
        ax_strata.set_xlabel("krok treningu")
        ax_strata.set_ylabel("strata (entropia krzyżowa)")

        ax_tekst.axis("off")
        linie = textwrap.fill(p["tekst"].replace("\n", " / "), 46).split("\n")[:14]
        ax_tekst.text(
            0.0,
            1.0,
            "\n".join(linie),
            family="monospace",
            ha="left",
            va="top",
            transform=ax_tekst.transAxes,
        )

        if i == 0:
            fig.suptitle("Przed treningiem (losowe wagi)")
        else:
            fig.suptitle(f"Po {krok} krokach, strata {p['strata']:.2f}")

    sciezki = zapisz_gif(
        fig,
        _rysuj_klatke,
        len(probki),
        ctx.wyniki_dir / "uczenie.gif",
        fps=2,
        klatka_png=len(probki) - 1,
    )
    json_sciezka = zapisz_json(
        {
            "material": "uczenie",
            "kroki": ctx.kroki_uczenia,
            "batch_size": ctx.batch_size,
            "lr": LR,
            "prompt": PROMPT,
            "probki": probki,
            "krzywa": {
                "kroki": wynik.kroki,
                "train_loss": wynik.train_loss,
                "val_kroki": wynik.val_kroki,
                "val_loss": wynik.val_loss,
            },
            "czas_s": wynik.czas_s,
        },
        ctx.wyniki_dir / "uczenie.json",
    )
    return [*sciezki, json_sciezka]


def main(argv: list[str] | None = None) -> int:
    """Uruchamia materiał samodzielnie: uv run python -m przewodnik.uczenie.

    Przykład: main(["--device", "cpu", "--kroki-uczenia", "20"]) -> 0 (albo 1, gdy brak plików)
    """
    return uruchom_samodzielnie(zrob, 1, "Materiał 6: jak model uczy się pisać", argv)


if __name__ == "__main__":
    raise SystemExit(main())
