"""Materiał 10: próbkowanie. Ten sam rozkład następnego tokenu pod różnymi ustawieniami."""

from pathlib import Path
from typing import Any

import torch

from nanowriter.sample import filter_logits
from przewodnik.konteksty import KontekstI, uruchom_samodzielnie
from przewodnik.rysowanie import KOLORY, nowa_figura, zapisz_gif, zapisz_png
from przewodnik.wspolne import ZDANIE, entropia, etykieta_tokenu, zakoduj_zdanie, zapisz_json

NAJWIECEJ_SLUPKOW: int = 12
TEMPERATURY_GIF: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)


def _rozklad(
    logity: torch.Tensor, temperatura: float, top_k: int | None, top_p: float | None
) -> torch.Tensor:
    """Zwraca rozkład prawdopodobieństwa FloatTensor[V] dla logitów FloatTensor[V].

    Kolejność jak w generate: logity / temperatura, potem filter_logits, potem softmax.
    Przykład: _rozklad(torch.zeros(4), 1.0, None, None) -> tensor([0.25, 0.25, 0.25, 0.25])
    Przykład: _rozklad(torch.tensor([0.0, 1.0]), 1.0, 1, None) -> tensor([0., 1.])
    """
    filtrowane = filter_logits((logity / temperatura)[None], top_k, top_p)[0]
    return torch.softmax(filtrowane, dim=-1)


def zrob(ctx: KontekstI) -> list[Path]:
    """Tworzy probkowanie.png, probkowanie_temperatura.gif/.png i probkowanie.json.

    Prawdziwe logity modelu dla ostatniej pozycji prefiksu ZDANIE bez ostatniego słowa
    ("Wieczorem stary kot zasnął przy"). Ustawienia: temperatura 0.5, 1.0, 1.5, top-k = 5
    (temperatura 1.0), top-p = 0.9 (temperatura 1.0). Na wykresach te same
    NAJWIECEJ_SLUPKOW tokenów (najbardziej prawdopodobnych bez filtrów i przy temperaturze
    1.0). Zwraca [png, gif, png_gif, json] (w tej kolejności).
    Przykład: zrob(maly_kontekst_i(tmp)) -> 4 pliki; "ustawienia" ma 5 elementów
    """
    prefiks = ZDANIE.rsplit(" ", 1)[0]
    ids = zakoduj_zdanie(ctx.tokenizer, prefiks, ctx.model.config.block_size).to(ctx.device)
    with torch.no_grad():
        logity = ctx.model(ids)[0, -1].float().cpu()
    tokeny_prefiksu = [etykieta_tokenu(ctx.tokenizer, int(i)) for i in ids[0].tolist()]

    ustawienia_wejscie = (
        ("temperatura 0.5", 0.5, None, None),
        ("temperatura 1.0", 1.0, None, None),
        ("temperatura 1.5", 1.5, None, None),
        ("top-k = 5", 1.0, 5, None),
        ("top-p = 0.9", 1.0, None, 0.9),
    )
    bazowy = _rozklad(logity, 1.0, None, None)
    kolejnosc = [int(i) for i in torch.topk(bazowy, NAJWIECEJ_SLUPKOW).indices.tolist()]

    ustawienia: list[dict[str, Any]] = []
    for nazwa, temperatura, top_k, top_p in ustawienia_wejscie:
        prob = _rozklad(logity, temperatura, top_k, top_p)
        ustawienia.append(
            {
                "nazwa": nazwa,
                "temperatura": temperatura,
                "top_k": top_k,
                "top_p": top_p,
                "entropia": float(entropia(prob)),
                "liczba_niezerowych": int((prob > 0).sum()),
                "top": [
                    {"token": etykieta_tokenu(ctx.tokenizer, i), "p": float(prob[i])}
                    for i in kolejnosc
                ],
            }
        )

    etykiety = [etykieta_tokenu(ctx.tokenizer, i) for i in kolejnosc]
    szczyt = max(t["p"] for u in ustawienia for t in u["top"])

    fig, osie = nowa_figura(15, 4.8, 1, 5)
    for ax, u in zip(osie[0], ustawienia, strict=True):
        ax.bar(
            list(range(NAJWIECEJ_SLUPKOW)),
            [t["p"] for t in u["top"]],
            color=KOLORY["glowny"],
        )
        ax.set_xticks(
            list(range(NAJWIECEJ_SLUPKOW)), labels=etykiety, rotation=70, fontsize=7
        )
        ax.set_ylim(0, 1.1 * szczyt)
        ax.set_title(
            f"{u['nazwa']}\nentropia {u['entropia']:.2f}, tokenów z p>0: {u['liczba_niezerowych']}",
            fontsize=9,
        )
    fig.suptitle(
        f"Następny token po „{prefiks}”: ten sam rozkład, różne ustawienia próbkowania"
    )
    png = zapisz_png(fig, ctx.wyniki_dir / "probkowanie.png")

    klatki = [_rozklad(logity, temperatura, None, None) for temperatura in TEMPERATURY_GIF]
    przemiatanie = [
        {"temperatura": temperatura, "entropia": float(entropia(prob))}
        for temperatura, prob in zip(TEMPERATURY_GIF, klatki, strict=True)
    ]

    fig_gif, osie_gif = nowa_figura(8, 4.6)
    ax_gif = osie_gif[0, 0]

    def rysuj_klatke(i: int) -> None:
        """Rysuje klatkę i (temperatura TEMPERATURY_GIF[i], bez filtrów) na ax_gif."""
        ax_gif.clear()
        ax_gif.bar(
            list(range(NAJWIECEJ_SLUPKOW)),
            [float(klatki[i][t]) for t in kolejnosc],
            color=KOLORY["drugi"],
        )
        ax_gif.set_xticks(
            list(range(NAJWIECEJ_SLUPKOW)), labels=etykiety, rotation=70, fontsize=7
        )
        ax_gif.set_ylim(0, 1)
        ax_gif.set_title(
            f"Temperatura {TEMPERATURY_GIF[i]:.1f}: "
            f"entropia {przemiatanie[i]['entropia']:.2f} nat"
        )

    gif, png_gif = zapisz_gif(
        fig_gif,
        rysuj_klatke,
        len(TEMPERATURY_GIF),
        ctx.wyniki_dir / "probkowanie_temperatura.gif",
        fps=2,
        klatka_png=4,
    )

    json_sciezka = zapisz_json(
        {
            "material": "probkowanie",
            "prefiks": prefiks,
            "tokeny_prefiksu": tokeny_prefiksu,
            "ustawienia": ustawienia,
            "przemiatanie": przemiatanie,
        },
        ctx.wyniki_dir / "probkowanie.json",
    )
    return [png, gif, png_gif, json_sciezka]


def main(argv: list[str] | None = None) -> int:
    """Uruchamia materiał samodzielnie: uv run python -m przewodnik.probkowanie.

    Przykład: main(["--device", "cpu"]) -> 0 (albo 1 z komunikatem, gdy brak plików w data/)
    """
    return uruchom_samodzielnie(zrob, 1, "Materiał 10: próbkowanie", argv)


if __name__ == "__main__":
    raise SystemExit(main())
