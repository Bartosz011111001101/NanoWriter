"""Materiał 1: następny token. Top-10 prawdopodobieństw modelu dla kolejnych pozycji zdania."""

from pathlib import Path

import torch

from przewodnik.konteksty import KontekstI, uruchom_samodzielnie
from przewodnik.rysowanie import KOLORY, nowa_figura, zapisz_gif
from przewodnik.wspolne import (
    ZDANIE,
    etykieta_tokenu,
    top_k_rozklad,
    zakoduj_zdanie,
    zapisz_json,
)


def zrob(ctx: KontekstI) -> list[Path]:
    """Tworzy nastepny_token.gif, nastepny_token.png i nastepny_token.json w ctx.wyniki_dir.

    Jedno wywołanie modelu na ZDANIE daje logity każdej pozycji. Klatka t animacji to pasek
    top-10 tokenów przewidzianych PO tokenie t; prawdziwy następny token zdania ma kolor
    KOLORY["dobry"], reszta KOLORY["glowny"].
    Zwraca listę [gif, png, json] (ścieżki istniejących plików).
    Przykład: zrob(maly_kontekst_i(tmp)) -> 3 pliki; GIF ma tyle klatek, ile tokenów w zdaniu
    """
    idx_cpu = zakoduj_zdanie(ctx.tokenizer, ZDANIE, ctx.model.config.block_size)
    ids = idx_cpu[0].tolist()
    T = len(ids)
    with torch.no_grad():
        logity = ctx.model(idx_cpu.to(ctx.device))[0].float().cpu()
    etykiety = [etykieta_tokenu(ctx.tokenizer, token_id) for token_id in ids]

    pozycje = []
    for t in range(T):
        prob = torch.softmax(logity[t], dim=0)
        wpis = {
            "pozycja": t,
            "kontekst": etykiety[: t + 1],
            "prawdziwy_nastepny": None,
            "p_prawdziwego": None,
            "ranga_prawdziwego": None,
            "top10": top_k_rozklad(ctx.tokenizer, logity[t], 10),
        }
        if t < T - 1:
            prawdziwy = prob[ids[t + 1]]
            wpis["prawdziwy_nastepny"] = etykiety[t + 1]
            wpis["p_prawdziwego"] = float(prawdziwy)
            wpis["ranga_prawdziwego"] = 1 + int((prob > prawdziwy).sum())
        pozycje.append(wpis)

    fig, osie = nowa_figura(9, 4.8)
    ax = osie[0, 0]

    def rysuj_klatke(t: int) -> None:
        """Rysuje klatkę t: pasek top-10 po tokenie t i podpisy (najpierw czyści osie)."""
        ax.clear()
        wpis = pozycje[t]
        prawdziwy_id = ids[t + 1] if t < T - 1 else None
        top = wpis["top10"]
        n = len(top)
        kolory = [KOLORY["dobry"] if e["id"] == prawdziwy_id else KOLORY["glowny"] for e in top]
        ax.barh(range(n), [e["p"] for e in top], color=kolory)
        ax.set_yticks(range(n), labels=[e["token"].replace(" ", "·") for e in top])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        if prawdziwy_id is None:
            dopisek = "koniec zdania"
        else:
            dopisek = (
                f"prawidłowy następny: „{etykiety[t + 1]}” "
                f"(p = {wpis['p_prawdziwego']:.2f}, miejsce {wpis['ranga_prawdziwego']})"
            )
        ax.set_xlabel(f"prawdopodobieństwo\n{dopisek}")
        ax.set_title(f"Po tokenie {t + 1} z {T}: „{etykiety[t]}”", fontsize=11, pad=20)
        ax.text(
            0.5,
            1.01,
            "|".join(etykiety[: t + 1]),
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    pliki_gif = zapisz_gif(
        fig,
        rysuj_klatke,
        T,
        ctx.wyniki_dir / "nastepny_token.gif",
        fps=1.5,
        klatka_png=T // 2,
    )
    plik_json = zapisz_json(
        {
            "material": "nastepny_token",
            "zdanie": ZDANIE,
            "tokeny": etykiety,
            "pozycje": pozycje,
        },
        ctx.wyniki_dir / "nastepny_token.json",
    )
    return [*pliki_gif, plik_json]


def main(argv: list[str] | None = None) -> int:
    """Uruchamia materiał samodzielnie: uv run python -m przewodnik.nastepny_token.

    Przykład: main(["--device", "cpu"]) -> 0 (albo 1 z komunikatem, gdy brak plików w data/)
    """
    return uruchom_samodzielnie(zrob, 1, "Materiał 1: następny token", argv)


if __name__ == "__main__":
    raise SystemExit(main())
