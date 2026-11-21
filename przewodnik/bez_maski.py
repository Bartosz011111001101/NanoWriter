"""Materiał 9: wyłączona maska przyczynowa. Model podgląda przyszłość i „oszukuje”."""

import math
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from nanowriter.train import estimate_loss
from przewodnik.konteksty import KontekstI, uruchom_samodzielnie
from przewodnik.krotki_trening import krotki_trening, nowy_model
from przewodnik.rysowanie import KOLORY, nowa_figura, zapisz_png
from przewodnik.wspolne import probka_tekstu, zapisz_json

if TYPE_CHECKING:
    from nanowriter.model import GPT

PROMPT: str = "Wieczorem stary kot"
LICZBA_TOKENOW: int = 40
LR: float = 2e-3


def _wylacz_maske(model: "GPT") -> None:
    """Wpisuje same True do maski uwagi każdego bloku (uwaga widzi cały kontekst).

    Przykład: po wywołaniu model.blocks[0].attn.mask.all() -> tensor(True)
    """
    for blok in model.blocks:
        blok.attn.mask.fill_(True)


def _przywroc_maske(model: "GPT") -> None:
    """Wpisuje z powrotem maskę przyczynową (dolny trójkąt) do każdego bloku.

    Przykład: po _wylacz_maske i _przywroc_maske maska bloku == torch.tril(torch.ones(n, n)).bool()
    """
    for blok in model.blocks:
        maska = blok.attn.mask
        trojkat = torch.tril(torch.ones(maska.shape, dtype=torch.bool, device=maska.device))
        maska.copy_(trojkat)


def _zawin(tekst: str) -> str:
    """Zawija próbkę do 52 znaków w linii i najwyżej 7 linii; nowe linie zamienia na " / ".

    Przykład: _zawin("a\\nb") -> "a / b"
    """
    linie = textwrap.wrap(tekst.replace("\n", " / "), width=52)
    return "\n".join(linie[:7])


def zrob(ctx: KontekstI) -> list[Path]:
    """Tworzy bez_maski.png i bez_maski.json w ctx.wyniki_dir.

    Dwa nowe modele o tej samej konfiguracji, wagach startowych i batchach: pierwszy z maską
    przyczynową, drugi z wyłączoną maską (ctx.kroki_bez_maski kroków, LR). Strata walidacyjna
    drugiego modelu liczona dwa razy: z wyłączoną maską (podglądająca) i po przywróceniu maski
    (uczciwa). Po przywróceniu maski obie próbki tekstu generowane są uczciwie.
    Zwraca [png, json].
    Przykład: zrob(maly_kontekst_i(tmp)) -> 2 pliki; maska drugiego modelu po zrob to znów tril
    """
    kroki = ctx.kroki_bez_maski
    co_ile = max(1, kroki // 10)

    model_a = nowy_model(ctx.model.config, ctx.seed, ctx.device)
    wynik_a = krotki_trening(
        model_a,
        ctx.dane_train,
        ctx.dane_val,
        kroki=kroki,
        lr=LR,
        batch_size=ctx.batch_size,
        device=ctx.device,
        seed=ctx.seed,
        co_ile=co_ile,
    )
    val_a = wynik_a.val_loss[-1]
    tekst_a = probka_tekstu(
        model_a, ctx.tokenizer, PROMPT, LICZBA_TOKENOW, device=ctx.device, seed=ctx.seed
    )

    model_b = nowy_model(ctx.model.config, ctx.seed, ctx.device)
    _wylacz_maske(model_b)
    wynik_b = krotki_trening(
        model_b,
        ctx.dane_train,
        ctx.dane_val,
        kroki=kroki,
        lr=LR,
        batch_size=ctx.batch_size,
        device=ctx.device,
        seed=ctx.seed,
        co_ile=co_ile,
    )
    val_b_podgladajaca = wynik_b.val_loss[-1]

    _przywroc_maske(model_b)
    val_b_uczciwa = estimate_loss(
        model_b,
        ctx.dane_val,
        ctx.batch_size,
        model_b.config.block_size,
        ctx.device,
        max_batches=5,
    )
    tekst_b = probka_tekstu(
        model_b, ctx.tokenizer, PROMPT, LICZBA_TOKENOW, device=ctx.device, seed=ctx.seed
    )

    fig, osie = nowa_figura(11, 6.6, 2, 2)

    ax = osie[0, 0]
    ax.plot(
        wynik_a.kroki,
        wynik_a.train_loss,
        color=KOLORY["dobry"],
        label="z maską przyczynową",
    )
    ax.plot(
        wynik_b.kroki,
        wynik_b.train_loss,
        color=KOLORY["zly"],
        label="bez maski (podgląda przyszłość)",
    )
    ax.axhline(
        math.log(model_a.config.vocab_size),
        color=KOLORY["szary"],
        linestyle="--",
        label="ln(vocab_size)",
    )
    ax.set_xlabel("krok")
    ax.set_ylabel("strata")
    ax.legend()
    ax.set_title("Strata treningowa")

    ax = osie[0, 1]
    etykiety = [
        "z maską:\nwalidacja",
        "bez maski:\nwalidacja\nz podglądaniem",
        "bez maski:\nwalidacja uczciwa\n(maska przywrócona)",
    ]
    wartosci = [val_a, val_b_podgladajaca, val_b_uczciwa]
    slupki = ax.bar(
        range(3),
        wartosci,
        color=[KOLORY["dobry"], KOLORY["zly"], KOLORY["zly"]],
    )
    slupki[2].set_hatch("//")
    slupki[2].set_edgecolor("white")
    for slupek, wartosc in zip(slupki, wartosci, strict=True):
        ax.text(
            slupek.get_x() + slupek.get_width() / 2,
            wartosc,
            f"{wartosc:.2f}",
            ha="center",
            va="bottom",
        )
    ax.set_xticks(range(3))
    ax.set_xticklabels(etykiety, fontsize=7)
    ax.set_ylabel("strata")
    ax.set_title("Strata walidacyjna")

    for ax, tekst, tytul in (
        (osie[1, 0], tekst_a, "Próbka tekstu: model z maską"),
        (osie[1, 1], tekst_b, "Próbka tekstu: model bez maski (po przywróceniu maski)"),
    ):
        ax.axis("off")
        ax.set_title(tytul)
        ax.text(
            0.0,
            1.0,
            _zawin(tekst),
            ha="left",
            va="top",
            family="monospace",
            fontsize=8,
            transform=ax.transAxes,
        )

    fig.suptitle("Co się stanie, gdy model może podglądać następny token (bez maski przyczynowej)")
    png = zapisz_png(fig, ctx.wyniki_dir / "bez_maski.png")

    dane = {
        "material": "bez_maski",
        "kroki": kroki,
        "prompt": PROMPT,
        "z_maska": {
            "train_loss": wynik_a.train_loss,
            "val_loss_uczciwa": val_a,
            "tekst": tekst_a,
        },
        "bez_maski": {
            "train_loss": wynik_b.train_loss,
            "val_loss_podgladajaca": val_b_podgladajaca,
            "val_loss_uczciwa": val_b_uczciwa,
            "tekst": tekst_b,
        },
    }
    json_sciezka = zapisz_json(dane, ctx.wyniki_dir / "bez_maski.json")
    return [png, json_sciezka]


def main(argv: list[str] | None = None) -> int:
    """Uruchamia materiał samodzielnie: uv run python -m przewodnik.bez_maski.

    Przykład: main(["--device", "cpu", "--kroki-bez-maski", "20"]) -> 0 (albo 1, gdy brak plików)
    """
    return uruchom_samodzielnie(zrob, 1, "Materiał 9: wyłączona maska przyczynowa", argv)


if __name__ == "__main__":
    raise SystemExit(main())
