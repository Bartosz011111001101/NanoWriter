"""Materiał 4: uwaga krok po kroku i prawdziwe mapy uwagi ze wszystkich warstw i głowic."""

from pathlib import Path

import torch

from przewodnik.konteksty import KontekstI, uruchom_samodzielnie
from przewodnik.rysowanie import KOLORY, nowa_figura, zapisz_gif, zapisz_png
from przewodnik.wspolne import ZDANIE, entropia, etykieta_tokenu, zakoduj_zdanie, zapisz_json


def _wybierz_glowice(entropie: list[list[float]]) -> tuple[int, int]:
    """Zwraca (warstwa, glowica) o najmniejszej entropii; przy remisie pierwsza.

    Przykład: [[0.5, 0.2], [0.9, 0.2]] -> (0, 1)
    Przykład: [[1.0]] -> (0, 0)
    """
    najlepsza = (0, 0)
    najmniejsza = entropie[0][0]
    for warstwa, wiersz in enumerate(entropie):
        for glowica, wartosc in enumerate(wiersz):
            if wartosc < najmniejsza:
                najmniejsza = wartosc
                najlepsza = (warstwa, glowica)
    return najlepsza


def _zaokraglij(macierz: torch.Tensor) -> list[list[float]]:
    """Zamienia macierz [T, T] na listy liczb zaokrąglonych do 4 miejsc."""
    return [[round(v, 4) for v in wiersz] for wiersz in macierz.tolist()]


def _zbierz_slad(
    ctx: KontekstI, ids: torch.Tensor
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Jedno wywołanie modelu ze śladem; zwraca scores, weights ([n_head, T, T]) i masked."""
    bloki = list(ctx.model.blocks)
    for blok in bloki:
        blok.attn.trace = {}
    try:
        with torch.no_grad():
            ctx.model(ids.to(ctx.device))
        slady = [blok.attn.trace for blok in bloki]
    finally:
        for blok in bloki:
            blok.attn.trace = None
    scores = [s["scores"][0].cpu().float() for s in slady]
    wagi = [s["weights"][0].cpu().float() for s in slady]
    zamaskowane = slady[0]["masked"][0, 0].cpu().float()
    return scores, wagi, zamaskowane


def zrob(ctx: KontekstI) -> list[Path]:
    """Tworzy uwaga_krok_po_kroku.gif/.png, uwaga_mapy.png i uwaga.json w ctx.wyniki_dir.

    Jedno wywołanie modelu na ZDANIE z włączonym śladem uwagi we wszystkich blokach. GIF:
    dla głowicy o najmniejszej średniej entropii wag, każdy wiersz t w trzech klatkach
    (wyniki Q·K, maska przyczynowa, softmax). PNG: siatka map uwagi [warstwa x głowica].
    Zwraca [gif, png_podglad, png_mapy, json] (w tej kolejności).
    Przykład: zrob(maly_kontekst_i(tmp)) -> 4 pliki; GIF ma 3 * T klatek (T tokenów zdania)
    """
    ids = zakoduj_zdanie(ctx.tokenizer, ZDANIE, ctx.model.config.block_size)
    liczba_tokenow = ids.shape[1]
    if liczba_tokenow < 2:
        raise ValueError(f"zdanie ma {liczba_tokenow} token, potrzeba co najmniej 2")

    scores, wagi, zamaskowane = _zbierz_slad(ctx, ids)
    n_layer = len(scores)
    n_head = scores[0].shape[0]
    maska_przyczynowa = torch.isfinite(zamaskowane)

    entropie = [
        [float(entropia(wagi[w][g])[1:].mean()) for g in range(n_head)] for w in range(n_layer)
    ]
    warstwa, glowica = _wybierz_glowice(entropie)
    etykiety = [etykieta_tokenu(ctx.tokenizer, int(i)) for i in ids[0].tolist()]

    wyniki_wybrane = scores[warstwa][glowica]
    wagi_wybrane = wagi[warstwa][glowica]
    ymin = min(0.0, float(wyniki_wybrane.min())) * 1.15
    ymax = max(0.0, float(wyniki_wybrane.max())) * 1.15
    pozycje = list(range(liczba_tokenow))
    wyniki_lista = wyniki_wybrane.tolist()
    wagi_lista = wagi_wybrane.tolist()

    fig, osie = nowa_figura(11, 4.8, 1, 2)
    ax_lewa, ax_prawa = osie[0, 0], osie[0, 1]

    def rysuj_klatke(i: int) -> None:
        t, faza = divmod(i, 3)
        ax_lewa.clear()
        ax_prawa.clear()

        if faza == 0:
            ax_lewa.bar(pozycje, wyniki_lista[t], color=KOLORY["glowny"])
            ax_lewa.set_ylim(ymin, ymax)
            ax_lewa.set_title(f"Krok 1: wyniki Q·K dla „{etykiety[t]}” (każdy token jest oceniany)")
        elif faza == 1:
            ax_lewa.bar(pozycje[: t + 1], wyniki_lista[t][: t + 1], color=KOLORY["glowny"])
            for j in range(t + 1, liczba_tokenow):
                ax_lewa.plot(j, 0, marker="x", color=KOLORY["zly"], markersize=9)
                ax_lewa.annotate(
                    "−∞",
                    (j, 0),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    color=KOLORY["zly"],
                    fontsize=8,
                )
            ax_lewa.set_ylim(ymin, ymax)
            ax_lewa.set_title("Krok 2: maska przyczynowa: przyszłość dostaje −∞")
        else:
            ax_lewa.bar(pozycje, wagi_lista[t], color=KOLORY["dobry"])
            ax_lewa.set_ylim(0, 1.05)
            ax_lewa.set_title("Krok 3: softmax: wagi sumują się do 1")
        ax_lewa.set_xticks(pozycje)
        ax_lewa.set_xticklabels(etykiety, rotation=60, ha="right", fontsize=8)

        widoczne = [[float("nan")] * liczba_tokenow for _ in range(liczba_tokenow)]
        for wiersz in range(t + (1 if faza == 2 else 0)):
            widoczne[wiersz] = wagi_lista[wiersz]
        ax_prawa.imshow(widoczne, cmap="Blues", vmin=0, vmax=1)
        ax_prawa.axhspan(t - 0.5, t + 0.5, fill=False, edgecolor=KOLORY["drugi"], linewidth=2.5)
        ax_prawa.set_xticks(pozycje)
        ax_prawa.set_xticklabels(etykiety, rotation=90, fontsize=6)
        ax_prawa.set_yticks(pozycje)
        ax_prawa.set_yticklabels(etykiety, fontsize=6)
        ax_prawa.set_title("Wagi uwagi (po softmaksie)")

        fig.suptitle(
            f"Uwaga, warstwa {warstwa}, głowica {glowica}: wiersz {t + 1} z {liczba_tokenow}"
        )

    gif, podglad = zapisz_gif(
        fig,
        rysuj_klatke,
        3 * liczba_tokenow,
        ctx.wyniki_dir / "uwaga_krok_po_kroku.gif",
        fps=2,
        klatka_png=3 * (liczba_tokenow // 2) + 2,
    )

    fig_mapy, osie_mapy = nowa_figura(3.2 * n_head, 3.2 * n_layer, n_layer, n_head)
    for w in range(n_layer):
        for g in range(n_head):
            ax = osie_mapy[w, g]
            ax.imshow(wagi[w][g].tolist(), cmap="Blues", vmin=0, vmax=1)
            ax.set_title(f"W{w} G{g}")
            if w == n_layer - 1:
                ax.set_xticks(pozycje)
                ax.set_xticklabels(etykiety, rotation=90, fontsize=5)
            else:
                ax.set_xticks([])
            if g == 0:
                ax.set_yticks(pozycje)
                ax.set_yticklabels(etykiety, fontsize=5)
            else:
                ax.set_yticks([])
            if (w, g) == (warstwa, glowica):
                for krawedz in ax.spines.values():
                    krawedz.set_edgecolor(KOLORY["drugi"])
                    krawedz.set_linewidth(2.5)
    fig_mapy.suptitle(
        "Prawdziwe mapy uwagi: wiersz = token pytający, kolumna = token, na który patrzy"
    )
    png_mapy = zapisz_png(fig_mapy, ctx.wyniki_dir / "uwaga_mapy.png")

    json_plik = zapisz_json(
        {
            "material": "uwaga",
            "zdanie": ZDANIE,
            "tokeny": etykiety,
            "wybrana_glowica": {"warstwa": warstwa, "glowica": glowica},
            "entropia": entropie,
            "wyniki_qk": _zaokraglij(wyniki_wybrane),
            "maska_przyczynowa": maska_przyczynowa.tolist(),
            "wagi": [[_zaokraglij(wagi[w][g]) for g in range(n_head)] for w in range(n_layer)],
        },
        ctx.wyniki_dir / "uwaga.json",
    )
    return [gif, podglad, png_mapy, json_plik]


def main(argv: list[str] | None = None) -> int:
    """Uruchamia materiał samodzielnie: uv run python -m przewodnik.uwaga.

    Przykład: main(["--device", "cpu"]) -> 0 (albo 1 z komunikatem, gdy brak plików w data/)
    """
    return uruchom_samodzielnie(zrob, 1, "Materiał 4: uwaga", argv)


if __name__ == "__main__":
    raise SystemExit(main())
