"""Materiał 2: BPE krok po kroku. Scalanie par bajtów zdania w kolejności rang tokenizera."""

from pathlib import Path

from nanowriter.tokenizer import BPETokenizer
from przewodnik.konteksty import KontekstI, uruchom_samodzielnie
from przewodnik.rysowanie import KOLORY, nowa_figura, rysuj_pudelka_tokenow, zapisz_gif
from przewodnik.wspolne import ZDANIE, etykieta_tokenu, zapisz_json


def zrob(ctx: KontekstI) -> list[Path]:
    """Tworzy bpe.gif, bpe.png i bpe.json w ctx.wyniki_dir.

    Klatka 0: ZDANIE jako surowe bajty. Każda kolejna klatka: stan po scaleniu, które zmieniło
    zdanie (tylko takie scalenia, w kolejności rang); nowy token ma kolor KOLORY["podswietlenie"].
    Zwraca [gif, png, json]. Ostatnia klatka == tokenizer.encode(ZDANIE), inaczej RuntimeError.
    Przykład: tokenizer z merges [(97, 98)] i tekst "ab" -> 2 klatki: [97, 98] i [256]
    """
    merges = ctx.tokenizer.merges
    w_wierszu = 13

    # stany[nr] to tokeny zdania w klatce nr; opisy[nr - 1] to scalenie, które tworzy klatkę nr
    stany: list[list[int]] = [BPETokenizer([]).encode(ZDANIE)]
    opisy: list[tuple[int, tuple[int, int], int]] = []
    for k in range(1, len(merges) + 1):
        stan_k = BPETokenizer(merges[:k]).encode(ZDANIE)
        if stan_k != stany[-1]:
            ranga = k - 1
            stany.append(stan_k)
            opisy.append((ranga, merges[ranga], 256 + ranga))

    ostatni = ctx.tokenizer.encode(ZDANIE)
    if stany[-1] != ostatni:
        raise RuntimeError(
            f"ostatni stan ma {len(stany[-1])} tokenów, a tokenizer.encode(ZDANIE) "
            f"ma {len(ostatni)}: {stany[-1]} != {ostatni}"
        )

    liczba_klatek = len(stany)
    etykiety = [[etykieta_tokenu(ctx.tokenizer, token) for token in stan] for stan in stany]
    tytuly = [f"Krok 0: zdanie jako surowe bajty ({len(stany[0])} tokenów)"]
    for nr, (ranga, para, nowy) in enumerate(opisy, start=1):
        lewy, prawy = para
        tytuly.append(
            f"Krok {nr}: scalenie nr {ranga}: "
            f"„{etykieta_tokenu(ctx.tokenizer, lewy)}” + "
            f"„{etykieta_tokenu(ctx.tokenizer, prawy)}” → "
            f"„{etykieta_tokenu(ctx.tokenizer, nowy)}” (tokenów: {len(stany[nr])})"
        )

    fig, osie = nowa_figura(12, 4.6)
    ax = osie[0, 0]
    wiersze_start = (len(stany[0]) + w_wierszu - 1) // w_wierszu

    def rysuj_klatke(nr: int) -> None:
        ax.clear()
        nowy = opisy[nr - 1][2] if nr > 0 else None
        kolory = [
            KOLORY["podswietlenie"]
            if token == nowy
            else (KOLORY["tlo_a"] if i % 2 == 0 else KOLORY["tlo_b"])
            for i, token in enumerate(stany[nr])
        ]
        rysuj_pudelka_tokenow(
            ax,
            etykiety[nr],
            kolory=kolory,
            podpisy=[str(token) for token in stany[nr]],
            w_wierszu=w_wierszu,
        )
        ax.set_ylim(wiersze_start, 0)
        fig.suptitle(tytuly[nr])

    gif, png = zapisz_gif(
        fig,
        rysuj_klatke,
        liczba_klatek,
        ctx.wyniki_dir / "bpe.gif",
        fps=1.5,
        klatka_png=liczba_klatek - 1,
    )

    kroki = [
        {
            "nr": nr,
            "ranga": None if nr == 0 else opisy[nr - 1][0],
            "para": None if nr == 0 else list(opisy[nr - 1][1]),
            "nowy_token": None if nr == 0 else opisy[nr - 1][2],
            "tokeny": list(stany[nr]),
            "etykiety": list(etykiety[nr]),
        }
        for nr in range(liczba_klatek)
    ]
    json_sciezka = zapisz_json(
        {
            "material": "bpe",
            "zdanie": ZDANIE,
            "bajty": list(stany[0]),
            "kroki": kroki,
            "tokeny_koncowe": list(stany[-1]),
            "liczba_scalen": liczba_klatek - 1,
        },
        ctx.wyniki_dir / "bpe.json",
    )
    return [gif, png, json_sciezka]


def main(argv: list[str] | None = None) -> int:
    """Uruchamia materiał samodzielnie: uv run python -m przewodnik.bpe.

    Przykład: main(["--device", "cpu"]) -> 0 (albo 1 z komunikatem, gdy brak plików w data/)
    """
    return uruchom_samodzielnie(zrob, 1, "Materiał 2: BPE krok po kroku", argv)


if __name__ == "__main__":
    raise SystemExit(main())
