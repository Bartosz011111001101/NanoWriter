"""Rysowanie wykresów i animacji do przewodnika (matplotlib, backend Agg, bez okien)."""

from collections.abc import Callable
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

KOLORY: dict[str, str] = {
    "glowny": "#2a6fdb",
    "drugi": "#e07b00",
    "dobry": "#2a9d4b",
    "zly": "#d1495b",
    "szary": "#9aa0a6",
    "tlo_a": "#dbe7fb",
    "tlo_b": "#fde8c8",
    "liczone": "#b7e4c7",
    "nieliczone": "#e5e7eb",
    "podswietlenie": "#ffd166",
}


def nowa_figura(
    szerokosc: float, wysokosc: float, wiersze: int = 1, kolumny: int = 1
) -> tuple[Figure, np.ndarray]:
    """Tworzy figurę (cale) z siatką osi wiersze x kolumny; osie zawsze jako tablica 2D.

    Przed utworzeniem figury ustawia backend "Agg" (bez okien).
    Przykład: nowa_figura(4, 3) -> (Figure, tablica osi o kształcie (1, 1))
    Przykład: nowa_figura(8, 3, 1, 2)[1].shape -> (1, 2)
    """
    matplotlib.use("Agg")
    fig, osie = plt.subplots(
        wiersze,
        kolumny,
        figsize=(szerokosc, wysokosc),
        squeeze=False,
        layout="constrained",
    )
    return fig, osie


def zapisz_png(fig: Figure, sciezka: Path, dpi: int = 120) -> Path:
    """Zapisuje figurę do PNG, zamyka ją i zwraca sciezka; tworzy brakujące katalogi.

    Przykład: zapisz_png(fig, Path("w/a.png")) -> Path("w/a.png") (plik niepusty)
    """
    sciezka.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(sciezka, dpi=dpi, facecolor="white")
    plt.close(fig)
    return sciezka


def zapisz_gif(
    fig: Figure,
    rysuj_klatke: Callable[[int], None],
    liczba_klatek: int,
    sciezka: Path,
    fps: float = 2.0,
    dpi: int = 80,
    klatka_png: int | None = None,
) -> list[Path]:
    """Zapisuje animację GIF z klatek 0..liczba_klatek-1 i zamyka figurę.

    rysuj_klatke(i) rysuje klatkę i na figurze (najpierw czyści osie).
    klatka_png: numer klatki do podglądu statycznego; wtedy dodatkowo zapisuje PNG o tej samej
    nazwie (rozszerzenie .png) i zwraca [gif, png], inaczej [gif].
    Przykład: zapisz_gif(fig, f, 3, Path("w/a.gif"), klatka_png=2)
        -> [Path("w/a.gif"), Path("w/a.png")]; GIF ma 3 klatki
    Przykład: liczba_klatek=0 -> ValueError; klatka_png=3 przy 3 klatkach -> ValueError
    """
    if liczba_klatek < 1:
        raise ValueError("liczba_klatek musi być co najmniej 1")
    if klatka_png is not None and not 0 <= klatka_png < liczba_klatek:
        raise ValueError("klatka_png musi być w zakresie [0, liczba_klatek)")

    sciezka.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(sciezka), dpi=dpi):
        for i in range(liczba_klatek):
            rysuj_klatke(i)
            writer.grab_frame()

    if klatka_png is None:
        plt.close(fig)
        return [sciezka]

    rysuj_klatke(klatka_png)
    png = sciezka.with_suffix(".png")
    fig.savefig(png, dpi=120, facecolor="white")
    plt.close(fig)
    return [sciezka, png]


def rysuj_pudelka_tokenow(
    ax: Axes,
    etykiety: list[str],
    *,
    kolory: list[str] | None = None,
    podpisy: list[str] | None = None,
    w_wierszu: int = 20,
    fontsize: float = 8.0,
) -> None:
    """Rysuje tokeny jako prostokąty z etykietą w siatce: w_wierszu w wierszu, od góry w dół.

    kolory: kolor tła każdego pudełka (None -> naprzemiennie KOLORY["tlo_a"] i KOLORY["tlo_b"]).
    podpisy: drobny napis pod etykietą w tym samym pudełku (None -> bez podpisów).
    Spacje w etykiecie są pokazane jako "·". Wyłącza ramkę i osie osi ax.
    Przykład: 5 etykiet, w_wierszu=2 -> 3 wiersze (2, 2, 1 pudełka)
    Przykład: len(kolory) != len(etykiety) albo len(podpisy) != len(etykiety) -> ValueError
    Przykład: w_wierszu=0 -> ValueError; etykiety=[] -> pusta oś bez błędu
    """
    if w_wierszu < 1:
        raise ValueError("w_wierszu musi być co najmniej 1")
    if kolory is not None and len(kolory) != len(etykiety):
        raise ValueError("kolory muszą mieć tyle elementów, ile etykiet")
    if podpisy is not None and len(podpisy) != len(etykiety):
        raise ValueError("podpisy muszą mieć tyle elementów, ile etykiet")

    if not etykiety:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        return

    liczba_wierszy = (len(etykiety) + w_wierszu - 1) // w_wierszu
    for i, etykieta in enumerate(etykiety):
        wiersz, kolumna = divmod(i, w_wierszu)
        if kolory is not None:
            kolor = kolory[i]
        else:
            kolor = KOLORY["tlo_a"] if i % 2 == 0 else KOLORY["tlo_b"]
        ax.add_patch(
            Rectangle(
                (kolumna + 0.03, wiersz + 0.03),
                0.94,
                0.94,
                facecolor=kolor,
                edgecolor="#4b5563",
                linewidth=0.6,
            )
        )

        if len(etykieta) <= 4:
            rozmiar = fontsize
        elif len(etykieta) <= 8:
            rozmiar = 0.75 * fontsize
        else:
            rozmiar = 0.55 * fontsize
        tekst = etykieta.replace(" ", "·")

        if podpisy is None:
            ax.text(
                kolumna + 0.5,
                wiersz + 0.5,
                tekst,
                ha="center",
                va="center",
                fontsize=rozmiar,
            )
        else:
            ax.text(
                kolumna + 0.5,
                wiersz + 0.38,
                tekst,
                ha="center",
                va="center",
                fontsize=rozmiar,
            )
            ax.text(
                kolumna + 0.5,
                wiersz + 0.72,
                podpisy[i],
                ha="center",
                va="center",
                fontsize=0.6 * fontsize,
            )

    ax.set_xlim(0, w_wierszu)
    ax.set_ylim(liczba_wierszy, 0)
    ax.axis("off")
