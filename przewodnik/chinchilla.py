"""Materiał 14: reguła Chinchilla. Tokeny treningu kontra liczba parametrów."""

from pathlib import Path

from przewodnik.konteksty import KontekstII, PrzebiegTreningu, uruchom_samodzielnie
from przewodnik.rysowanie import KOLORY, nowa_figura, zapisz_png
from przewodnik.wspolne import zapisz_json

TOKENOW_NA_PARAMETR: int = 20


def _policz(przebieg: PrzebiegTreningu) -> tuple[int, float, int, float]:
    """Zwraca (tokeny_treningu, tokenow_na_parametr, optimum_tokenow, epoki) dla przebiegu.

    Przykład: parametry 0 albo tokenow_korpusu 0 -> ValueError
    """
    if przebieg.parametry == 0:
        raise ValueError(f"{przebieg.nazwa}: liczba parametrów wynosi 0")
    if przebieg.tokenow_korpusu == 0:
        raise ValueError(f"{przebieg.nazwa}: korpus treningowy ma 0 tokenów")
    tokeny_treningu = przebieg.kroki * przebieg.batch_size * przebieg.block_size
    tokenow_na_parametr = tokeny_treningu / przebieg.parametry
    optimum_tokenow = TOKENOW_NA_PARAMETR * przebieg.parametry
    epoki = tokeny_treningu / przebieg.tokenow_korpusu
    return tokeny_treningu, tokenow_na_parametr, optimum_tokenow, epoki


def _rysuj(
    przebiegi: list[PrzebiegTreningu],
    wyniki: list[tuple[int, float, int, float]],
    sciezka: Path,
) -> Path:
    """Rysuje wykres log-log tokenów treningu i słupki tokenów na parametr, zapisuje PNG.

    Przykład: trzy przebiegi -> plik PNG w sciezka, zwrócony jako ta sama ścieżka
    """
    fig, osie = nowa_figura(12, 5, 1, 2)
    lewa = osie[0, 0]
    prawa = osie[0, 1]
    kolory = [KOLORY["glowny"], KOLORY["drugi"], KOLORY["dobry"]]

    parametry = [przebieg.parametry for przebieg in przebiegi]
    lo = min(parametry) / 2
    hi = 2 * max(parametry)

    lewa.set_xscale("log")
    lewa.set_yscale("log")
    lewa.plot(
        [lo, hi],
        [TOKENOW_NA_PARAMETR * lo, TOKENOW_NA_PARAMETR * hi],
        linestyle="--",
        color=KOLORY["szary"],
        label=f"optimum Chinchilla: {TOKENOW_NA_PARAMETR} tokenów na parametr",
    )
    lewa.set_xlim(lo, hi)
    for przebieg, (tokeny, tokenow_na_parametr, _, _), kolor in zip(
        przebiegi, wyniki, kolory, strict=True
    ):
        lewa.scatter([przebieg.parametry], [tokeny], s=90, color=kolor, zorder=3)
        lewa.annotate(
            f"{przebieg.nazwa}: {tokenow_na_parametr:.1f} tokenów na parametr",
            (przebieg.parametry, tokeny),
            textcoords="offset points",
            xytext=(8, -4),
            fontsize=8,
        )
    lewa.set_xlabel("liczba parametrów")
    lewa.set_ylabel("tokeny treningu")
    lewa.set_title("Ile tokenów zobaczył model")
    lewa.legend(loc="upper left", fontsize=8)

    nazwy = [przebieg.nazwa for przebieg in przebiegi]
    tokenow_na_parametr = [tpp for _, tpp, _, _ in wyniki]
    prawa.bar(list(range(len(nazwy))), tokenow_na_parametr, color=kolory)
    for i, wartosc in enumerate(tokenow_na_parametr):
        prawa.text(i, wartosc, f"{wartosc:.1f}", ha="center", va="bottom", fontsize=9)
    prawa.axhline(
        TOKENOW_NA_PARAMETR,
        linestyle="--",
        color=KOLORY["szary"],
        label="Chinchilla",
    )
    prawa.set_xticks(list(range(len(nazwy))), nazwy)
    prawa.set_yscale("log")
    prawa.set_ylabel("tokenów na parametr")
    prawa.set_title(f"Tokenów na parametr ({TOKENOW_NA_PARAMETR} = optimum)")
    prawa.legend(loc="center right", fontsize=8)

    return zapisz_png(fig, sciezka)


def zrob(ctx: KontekstII) -> list[Path]:
    """Tworzy chinchilla.png i chinchilla.json w ctx.wyniki_dir.

    Dla przebiegów 1M, 10M wstępny i 10M douczony (w tej kolejności): tokeny treningu =
    kroki * batch_size * block_size, tokenów na parametr, optimum (20 * parametry) i liczba
    epok (tokeny treningu / tokenow_korpusu).
    Zwraca [png, json].
    Przykład: kroki 12000, batch 64, block 256, parametry 10101760 -> tokeny_treningu 196608000,
        tokenow_na_parametr ≈ 19.46
    """
    przebiegi = [ctx.przebieg_1m, ctx.przebieg_wstepny, ctx.przebieg_douczania]
    wyniki = [_policz(przebieg) for przebieg in przebiegi]

    png = _rysuj(przebiegi, wyniki, ctx.wyniki_dir / "chinchilla.png")
    json_plik = zapisz_json(
        {
            "material": "chinchilla",
            "tokenow_na_parametr_chinchilla": TOKENOW_NA_PARAMETR,
            "modele": [
                {
                    "nazwa": przebieg.nazwa,
                    "parametry": przebieg.parametry,
                    "tokeny_treningu": tokeny,
                    "tokenow_na_parametr": tokenow_na_parametr,
                    "optimum_tokenow": optimum,
                    "tokenow_korpusu": przebieg.tokenow_korpusu,
                    "epoki": epoki,
                }
                for przebieg, (tokeny, tokenow_na_parametr, optimum, epoki) in zip(
                    przebiegi, wyniki, strict=True
                )
            ],
        },
        ctx.wyniki_dir / "chinchilla.json",
    )
    return [png, json_plik]


def main(argv: list[str] | None = None) -> int:
    """Uruchamia materiał samodzielnie: uv run python -m przewodnik.chinchilla.

    Przykład: main(["--device", "cpu"]) -> 0 (albo 1 z komunikatem, gdy brak plików w data/)
    """
    return uruchom_samodzielnie(zrob, 2, "Materiał 14: reguła Chinchilla", argv)


if __name__ == "__main__":
    raise SystemExit(main())
