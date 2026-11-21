"""Materiał 13: to samo zdanie pocięte słownikiem 1024 i słownikiem 8192."""

import math
from pathlib import Path

from przewodnik.konteksty import KontekstII, uruchom_samodzielnie
from przewodnik.rysowanie import nowa_figura, rysuj_pudelka_tokenow, zapisz_png
from przewodnik.wspolne import ZDANIE, etykieta_tokenu, zapisz_json

ZDANIA: tuple[str, ...] = (ZDANIE, "Dzień dobry, jak minął ci weekend?", "Zażółć gęślą jaźń.")


def zrob(ctx: KontekstII) -> list[Path]:
    """Tworzy tokenizacja.png i tokenizacja.json w ctx.wyniki_dir.

    Każde zdanie z ZDANIA zakodowane tokenizerem 1M (tokenizer_1m) i 10M (tokenizer_10m).
    Wykres: dla każdego zdania dwa wiersze pudełek z tokenami (id jako podpis pod etykietą).
    Zwraca [png, json]. decode(encode(zdanie)) != zdanie -> RuntimeError.
    Przykład: zrob(maly_kontekst_ii(tmp)) -> 2 pliki; "zdania" ma 3 elementy
    """
    tokenizery = (("slownik_1m", ctx.tokenizer_1m), ("slownik_10m", ctx.tokenizer_10m))
    zdania_json: list[dict[str, object]] = []
    # Jeden wiersz wykresu na jeden słownik: (zdanie, vocab_size, id, etykiety, znakow_na_token).
    wiersze: list[tuple[str, int, list[int], list[str], float]] = []
    for zdanie in ZDANIA:
        wpis: dict[str, object] = {"zdanie": zdanie}
        for klucz, tokenizer in tokenizery:
            ids = tokenizer.encode(zdanie)
            if tokenizer.decode(ids) != zdanie:
                raise RuntimeError(
                    f"decode(encode(zdanie)) != zdanie dla słownika {tokenizer.vocab_size}: "
                    f"{zdanie!r}"
                )
            etykiety = [etykieta_tokenu(tokenizer, token_id) for token_id in ids]
            liczba = len(ids)
            # Pusty tekst nie ma tokenów, więc nie ma długości tokenu: 0.0 zamiast dzielenia.
            znakow_na_token = len(zdanie) / liczba if liczba else 0.0
            wpis[klucz] = {
                "vocab_size": tokenizer.vocab_size,
                "tokeny": etykiety,
                "id": ids,
                "liczba": liczba,
                "znakow_na_token": znakow_na_token,
            }
            wiersze.append((zdanie, tokenizer.vocab_size, ids, etykiety, znakow_na_token))
        zdania_json.append(wpis)

    fig, osie = nowa_figura(13, 9, 6, 1)
    for numer, (zdanie, vocab_size, ids, etykiety, znakow_na_token) in enumerate(wiersze):
        ax = osie[numer, 0]
        liczba = len(ids)
        rysuj_pudelka_tokenow(
            ax,
            etykiety,
            podpisy=[str(token_id) for token_id in ids],
            w_wierszu=20,
        )
        ax.set_ylim(max(2, math.ceil(liczba / 20)), 0)
        ax.set_title(
            f"„{zdanie}”, słownik {vocab_size}: {liczba} tokenów, "
            f"{znakow_na_token:.1f} znaku na token",
            loc="left",
            fontsize=7,
        )
    fig.suptitle("To samo zdanie pocięte dwoma słownikami BPE")
    png = zapisz_png(fig, ctx.wyniki_dir / "tokenizacja.png")

    json_plik = zapisz_json(
        {"material": "tokenizacja", "zdania": zdania_json},
        ctx.wyniki_dir / "tokenizacja.json",
    )
    return [png, json_plik]


def main(argv: list[str] | None = None) -> int:
    """Uruchamia materiał samodzielnie: uv run python -m przewodnik.tokenizacja.

    Przykład: main(["--device", "cpu"]) -> 0 (albo 1 z komunikatem, gdy brak plików w data/)
    """
    return uruchom_samodzielnie(zrob, 2, "Materiał 13: tokenizacja słownikami 1024 i 8192", argv)


if __name__ == "__main__":
    raise SystemExit(main())
