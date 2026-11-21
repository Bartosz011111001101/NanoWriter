"""Materiał 15: maska straty przy douczaniu. Które tokeny rozmowy liczą się w stracie."""

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from nanowriter.dataset import encode_dialogs
from nanowriter.train import compute_loss
from przewodnik.konteksty import uruchom_samodzielnie
from przewodnik.rysowanie import KOLORY, nowa_figura, rysuj_pudelka_tokenow, zapisz_gif
from przewodnik.wspolne import ROZMOWA_TEKST, etykieta_tokenu, zapisz_json

if TYPE_CHECKING:
    from przewodnik.konteksty import KontekstII

_W_WIERSZU: int = 16


def zrob(ctx: "KontekstII") -> list[Path]:
    """Tworzy maska_straty.gif, maska_straty.png i maska_straty.json w ctx.wyniki_dir.

    ROZMOWA_TEKST kodowana przez encode_dialogs tokenizerem 10M; strata każdego tokenu (poza
    pierwszym) z modelu douczonego. Animacja odsłania rozmowę linia po linii: tokeny liczone w
    stracie (maska 1) w KOLORY["liczone"], pozostałe w KOLORY["nieliczone"], pod etykietą strata
    tokenu. Zwraca [gif, png, json] (w tej kolejności).
    Przykład: zrob(maly_kontekst_ii(tmp)) -> 3 pliki; GIF ma tyle klatek, ile linii rozmowy (4)
    Przykład: rozmowa dłuższa niż block_size modelu -> ValueError
    """
    ids, maska = encode_dialogs(ctx.tokenizer_10m, ROZMOWA_TEKST)
    n = len(ids)
    block_size = ctx.model_douczony.config.block_size
    if n > block_size:
        raise ValueError(f"rozmowa ma {n} tokenów, a block_size modelu to {block_size}")
    id_nl = ctx.tokenizer_10m.encode("\n")[0]

    x = torch.tensor([ids[:-1]], dtype=torch.long, device=ctx.device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=ctx.device)
    my = torch.tensor([maska[1:]], dtype=torch.bool, device=ctx.device)
    with torch.no_grad():
        logity = ctx.model_douczony(x)
        strata = F.cross_entropy(
            logity.reshape(-1, logity.shape[-1]), y.reshape(-1), reduction="none"
        )
        strata_srednia_wszystkie = float(compute_loss(logity, y))
        strata_srednia_maska = float(compute_loss(logity, y, my))
    strata_tokenow: list[float] = strata.cpu().tolist()

    linie = ROZMOWA_TEKST.splitlines()
    liczba_linii = len(linie)
    mowcy = [wiersz[:1] for wiersz in linie]
    linia_tokenu: list[int] = []
    numer_linii = 0
    for token_id in ids:
        linia_tokenu.append(numer_linii)
        if token_id == id_nl:
            numer_linii += 1
    indeksy_linii = [
        [i for i in range(n) if linia_tokenu[i] == nr_linii] for nr_linii in range(liczba_linii)
    ]

    etykiety = [etykieta_tokenu(ctx.tokenizer_10m, token_id) for token_id in ids]
    kolory_tokenow = [
        KOLORY["liczone"] if maska[i] == 1 else KOLORY["nieliczone"] for i in range(n)
    ]
    podpisy_tokenow = ["–" if i == 0 else f"{strata_tokenow[i - 1]:.1f}" for i in range(n)]

    fig, osie = nowa_figura(13, 1.7 * liczba_linii + 1.6, liczba_linii + 1, 1)
    fig.suptitle("Maska straty przy douczaniu na rozmowach")
    osie_linii = [osie[nr_linii, 0] for nr_linii in range(liczba_linii)]
    ax_zbiorcza = osie[liczba_linii, 0]

    def rysuj_klatke(k: int) -> None:
        for nr_linii, ax in enumerate(osie_linii):
            ax.clear()
            if nr_linii > k:
                ax.axis("off")
                continue
            idx = indeksy_linii[nr_linii]
            rysuj_pudelka_tokenow(
                ax,
                [etykiety[i] for i in idx],
                kolory=[kolory_tokenow[i] for i in idx],
                podpisy=[podpisy_tokenow[i] for i in idx],
                w_wierszu=_W_WIERSZU,
            )
            ax.set_ylim(max(1, math.ceil(len(idx) / _W_WIERSZU)), 0)
            nazwa_mowcy = "użytkownik" if mowcy[nr_linii] == "A" else "model"
            ax.set_title(f"linia {nr_linii + 1} ({nazwa_mowcy})", loc="left", fontsize=9)

        ax_zbiorcza.clear()
        ax_zbiorcza.axis("off")
        widoczne = [i for i in range(n) if linia_tokenu[i] <= k]
        liczone_widoczne = sum(maska[i] for i in widoczne)
        tekst = (
            f"Liczone w stracie: {liczone_widoczne} z {len(widoczne)} tokenów "
            "(zielone: treść wypowiedzi B z końcowym \\n; "
            "szare: etykiety A:/B: i wypowiedzi A)"
        )
        if k == liczba_linii - 1:
            tekst += (
                f"\nŚrednia strata po wszystkich pozycjach: {strata_srednia_wszystkie:.2f}; "
                f"tylko po zielonych: {strata_srednia_maska:.2f}"
            )
        ax_zbiorcza.text(
            0.0, 0.5, tekst, transform=ax_zbiorcza.transAxes, va="center", fontsize=9
        )

    gif_i_png = zapisz_gif(
        fig,
        rysuj_klatke,
        liczba_linii,
        ctx.wyniki_dir / "maska_straty.gif",
        fps=0.8,
        klatka_png=liczba_linii - 1,
    )

    tokeny = [
        {
            "i": i,
            "token": etykiety[i],
            "maska": maska[i],
            "strata": None if i == 0 else strata_tokenow[i - 1],
            "linia": linia_tokenu[i],
        }
        for i in range(n)
    ]
    json_plik = zapisz_json(
        {
            "material": "maska_straty",
            "rozmowa": ROZMOWA_TEKST,
            "tokeny": tokeny,
            "liczone": sum(maska),
            "wszystkie": n,
            "strata_srednia_wszystkie": strata_srednia_wszystkie,
            "strata_srednia_maska": strata_srednia_maska,
        },
        ctx.wyniki_dir / "maska_straty.json",
    )
    return [*gif_i_png, json_plik]


def main(argv: list[str] | None = None) -> int:
    """Uruchamia materiał samodzielnie: uv run python -m przewodnik.maska_straty.

    Przykład: main(["--device", "cpu"]) -> 0 (albo 1 z komunikatem, gdy brak plików w data/)
    """
    return uruchom_samodzielnie(zrob, 2, "Materiał 15: maska straty przy douczaniu", argv)


if __name__ == "__main__":
    raise SystemExit(main())
