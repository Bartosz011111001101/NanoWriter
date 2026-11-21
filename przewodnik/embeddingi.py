"""Materiał 3: embeddingi tokenów rzutowane na 2 wymiary (PCA)."""

from pathlib import Path

import torch

from przewodnik.konteksty import KontekstI, uruchom_samodzielnie
from przewodnik.rysowanie import KOLORY, nowa_figura, zapisz_png
from przewodnik.wspolne import ZDANIE, etykieta_tokenu, zakoduj_zdanie, zapisz_json


def zrob(ctx: KontekstI) -> list[Path]:
    """Tworzy embeddingi.png i embeddingi.json w ctx.wyniki_dir.

    PCA (centrowanie, SVD) macierzy tok_emb.weight [V, d] do 2 składowych. Wykres: wszystkie
    tokeny jako szare punkty, 200 najczęstszych w korpusie ctx.dane_train na niebiesko,
    tokeny ZDANIE na pomarańczowo; podpisane 40 najczęstszych i tokeny zdania.
    Zwraca [png, json].
    Przykład: zrob(maly_kontekst_i(tmp)) -> 2 pliki; w JSON "wariancja_wyjasniona" ma 2 liczby
    """
    # Kopia wag na CPU w float32; model i jego wagi nie są zmieniane.
    wagi: torch.Tensor = ctx.model.tok_emb.weight.detach().to(device="cpu", dtype=torch.float32)
    liczba_tokenow, d_model = wagi.shape

    # PCA: centrowanie i SVD (wszystko w nowych tensorach, bez zapisu w miejscu).
    wagi_c = wagi - wagi.mean(dim=0)
    _, sv, vh = torch.linalg.svd(wagi_c, full_matrices=False)
    xy = wagi_c @ vh[:2].T
    wariancja: list[float] = (sv[:2] ** 2 / torch.sum(sv**2)).tolist()

    # Częstość każdego tokenu w korpusie treningowym.
    czestosc = torch.bincount(ctx.dane_train.long(), minlength=liczba_tokenow)

    # Unikalne id tokenów zdania (w kolejności występowania) i 200 najczęstszych tokenów.
    block_size = ctx.model.config.block_size
    ids_zdania = zakoduj_zdanie(ctx.tokenizer, ZDANIE, block_size)[0].tolist()
    id_zdania: list[int] = list(dict.fromkeys(ids_zdania))
    najczestsze: list[int] = torch.argsort(czestosc, descending=True, stable=True)[
        : min(200, liczba_tokenow)
    ].tolist()
    etykietowane: list[int] = list(dict.fromkeys(najczestsze[:40] + id_zdania))

    # Wykres: szare tło, 200 najczęstszych, tokeny zdania na wierzchu.
    fig, osie = nowa_figura(10, 8)
    ax = osie[0, 0]
    wszystkie_x, wszystkie_y = xy[:, 0].tolist(), xy[:, 1].tolist()
    ax.scatter(
        wszystkie_x,
        wszystkie_y,
        s=6,
        color=KOLORY["szary"],
        alpha=0.4,
        zorder=1,
        label="wszystkie tokeny",
    )
    ax.scatter(
        xy[najczestsze, 0].tolist(),
        xy[najczestsze, 1].tolist(),
        s=14,
        color=KOLORY["glowny"],
        zorder=2,
        label="200 najczęstszych",
    )
    ax.scatter(
        xy[id_zdania, 0].tolist(),
        xy[id_zdania, 1].tolist(),
        s=60,
        color=KOLORY["drugi"],
        edgecolors="#1f2937",
        linewidths=0.8,
        zorder=3,
        label="tokeny zdania",
    )
    for token_id in etykietowane:
        x, y = xy[token_id].tolist()
        ax.annotate(
            etykieta_tokenu(ctx.tokenizer, token_id),
            xy=(x, y),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6,
        )
    ax.set_xlabel(f"składowa 1 ({wariancja[0]:.1%} wariancji)")
    ax.set_ylabel(f"składowa 2 ({wariancja[1]:.1%} wariancji)")
    ax.set_title("Embeddingi tokenów modelu 1M rzutowane na dwie główne składowe (PCA)")
    ax.legend(loc="best")
    sciezka_png = zapisz_png(fig, ctx.wyniki_dir / "embeddingi.png")

    # JSON: najpierw 200 najczęstszych (malejąco), potem tokeny zdania, których tam nie ma.
    zbior_zdania = set(id_zdania)
    zbior_najczestszych = set(najczestsze)
    kolejnosc = najczestsze + [i for i in id_zdania if i not in zbior_najczestszych]
    tokeny: list[dict[str, object]] = []
    for token_id in kolejnosc:
        x, y = xy[token_id].tolist()
        tokeny.append(
            {
                "id": token_id,
                "token": etykieta_tokenu(ctx.tokenizer, token_id),
                "czestosc": int(czestosc[token_id]),
                "x": x,
                "y": y,
                "w_zdaniu": token_id in zbior_zdania,
            }
        )
    sciezka_json = zapisz_json(
        {
            "material": "embeddingi",
            "zdanie": ZDANIE,
            "vocab_size": liczba_tokenow,
            "d_model": d_model,
            "wariancja_wyjasniona": wariancja,
            "etykietowane": etykietowane,
            "tokeny": tokeny,
        },
        ctx.wyniki_dir / "embeddingi.json",
    )
    return [sciezka_png, sciezka_json]


def main(argv: list[str] | None = None) -> int:
    """Uruchamia materiał samodzielnie: uv run python -m przewodnik.embeddingi.

    Przykład: main(["--device", "cpu"]) -> 0 (albo 1 z komunikatem, gdy brak plików w data/)
    """
    return uruchom_samodzielnie(zrob, 1, "Materiał 3: embeddingi tokenów (PCA)", argv)


if __name__ == "__main__":
    raise SystemExit(main())
