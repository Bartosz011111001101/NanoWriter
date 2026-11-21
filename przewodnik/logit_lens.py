"""Materiał 5: strumień rezydualny i logit lens: co model przewiduje po każdej warstwie."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from matplotlib.axes import Axes
from torch import nn

from nanowriter.model import GPT
from przewodnik.konteksty import KontekstI, uruchom_samodzielnie
from przewodnik.rysowanie import KOLORY, nowa_figura, zapisz_gif, zapisz_png
from przewodnik.wspolne import ZDANIE, etykieta_tokenu, top_k_rozklad, zakoduj_zdanie, zapisz_json


@dataclass(frozen=True, eq=False)
class _Zebrane:
    """Stany i wkłady zebrane hakami podczas jednego wywołania modelu (tensory [1, T, d])."""

    stany: list[torch.Tensor]
    uwaga: list[torch.Tensor]
    mlp: list[torch.Tensor]
    logity_modelu: torch.Tensor


def _zbierz(model: GPT, ids: torch.Tensor) -> _Zebrane:
    """Jedno wywołanie modelu bez gradientów z hakami; uchwyty zdejmowane w finally."""
    n_layer = len(model.blocks)
    stany: list[torch.Tensor | None] = [None] * (n_layer + 1)
    uwaga: list[torch.Tensor | None] = [None] * n_layer
    mlp: list[torch.Tensor | None] = [None] * n_layer

    def zapis_wejscia(_modul: nn.Module, args: tuple[Any, ...]) -> None:
        stany[0] = args[0].detach()

    def zapis_stanu(i: int) -> Callable[[nn.Module, tuple[Any, ...], torch.Tensor], None]:
        def hak(_modul: nn.Module, _args: tuple[Any, ...], wyjscie: torch.Tensor) -> None:
            stany[i + 1] = wyjscie.detach()

        return hak

    def zapis_wkladu(
        cel: list[torch.Tensor | None], i: int
    ) -> Callable[[nn.Module, tuple[Any, ...], torch.Tensor], None]:
        def hak(_modul: nn.Module, _args: tuple[Any, ...], wyjscie: torch.Tensor) -> None:
            cel[i] = wyjscie.detach()

        return hak

    uchwyty = []
    try:
        uchwyty.append(model.blocks[0].register_forward_pre_hook(zapis_wejscia))
        for i, blok in enumerate(model.blocks):
            uchwyty.append(blok.register_forward_hook(zapis_stanu(i)))
            uchwyty.append(blok.attn.register_forward_hook(zapis_wkladu(uwaga, i)))
            uchwyty.append(blok.mlp.register_forward_hook(zapis_wkladu(mlp, i)))
        with torch.no_grad():
            logity_modelu = model(ids)
    finally:
        for uchwyt in uchwyty:
            uchwyt.remove()
    return _Zebrane(
        stany=[s for s in stany if s is not None],
        uwaga=[u for u in uwaga if u is not None],
        mlp=[m for m in mlp if m is not None],
        logity_modelu=logity_modelu,
    )


def _logity_etapow(model: GPT, zebrane: _Zebrane) -> list[torch.Tensor]:
    """Logit lens: ln_f i lm_head modelu na każdym stanie; zwraca listę [T, V] (CPU, float32)."""
    with torch.no_grad():
        logity = [model.lm_head(model.ln_f(stan))[0].float().cpu() for stan in zebrane.stany]
    if not torch.allclose(logity[-1], zebrane.logity_modelu[0].float().cpu(), atol=1e-5):
        raise RuntimeError("logit lens po ostatnim bloku różni się od logitów modelu")
    return logity


def _srednia_norma(tensor: torch.Tensor) -> float:
    """Średnia (po pozycjach) norma L2 wektorów z tensora [1, T, d]."""
    return float(tensor[0].float().norm(dim=-1).mean())


def _opis_etapu(etap: int, n_layer: int) -> str:
    """Opis etapu po polsku: 0 -> po embeddingach, i -> po bloku i z n_layer."""
    if etap == 0:
        return "po embeddingach (zanim cokolwiek policzono)"
    return f"po bloku {etap} z {n_layer}"


def _dane_etapu(
    ctx: KontekstI, logity: torch.Tensor, ids: list[int], etap: int, n_layer: int
) -> dict[str, Any]:
    """Składa wpis JSON etapu: top3 i p_prawdziwego dla każdej pozycji zdania."""
    prob = torch.softmax(logity, dim=-1)
    pozycje: list[dict[str, Any]] = []
    for t in range(len(ids)):
        top3 = [
            {"token": wpis["token"], "p": wpis["p"]}
            for wpis in top_k_rozklad(ctx.tokenizer, logity[t], 3)
        ]
        prawdziwy = float(prob[t, ids[t + 1]]) if t < len(ids) - 1 else None
        pozycje.append({"top3": top3, "p_prawdziwego": prawdziwy})
    return {"etap": etap, "opis": _opis_etapu(etap, n_layer), "pozycje": pozycje}


def _rysuj_mape(ax: Axes, wpis: dict[str, Any], etykiety: list[str]) -> None:
    """Rysuje mapę cieplną top3 dla jednego etapu wraz z prawidłowymi tokenami po prawej."""
    ax.clear()
    pozycje = wpis["pozycje"]
    liczba = len(pozycje)
    macierz = [[m["p"] for m in poz["top3"]] for poz in pozycje]
    ax.imshow(macierz, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for t, poz in enumerate(pozycje):
        for k, miejsce in enumerate(poz["top3"]):
            ax.text(
                k,
                t,
                f"{miejsce['token']} {miejsce['p']:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if miejsce["p"] > 0.5 else "black",
            )
        if t < liczba - 1:
            ax.text(
                1.02,
                t,
                f"prawidłowy: „{etykiety[t + 1]}” p={poz['p_prawdziwego']:.2f}",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=8,
                clip_on=False,
            )
    ax.set_yticks(range(liczba), labels=etykiety, fontsize=8)
    ax.set_xticks(range(3), labels=["1. typ", "2. typ", "3. typ"])
    ax.set_title(f"Logit lens: co model przewiduje {wpis['opis']}")


def _rysuj_podsumowanie(
    ctx: KontekstI,
    etapy: list[dict[str, Any]],
    srednie: list[float],
    normy: list[dict[str, float]],
) -> Path:
    """Rysuje i zapisuje logit_lens_podsumowanie.png (dwa wykresy obok siebie)."""
    n_blokow = len(normy)
    fig, osie = nowa_figura(11, 4.2, 1, 2)
    lewa, prawa = osie[0, 0], osie[0, 1]
    liczba_pozycji = len(etapy[0]["pozycje"])
    for t in range(liczba_pozycji - 1):
        lewa.plot(
            range(n_blokow + 1),
            [e["pozycje"][t]["p_prawdziwego"] for e in etapy],
            color=KOLORY["szary"],
            linewidth=0.8,
            alpha=0.7,
        )
    lewa.plot(range(n_blokow + 1), srednie, color=KOLORY["glowny"], linewidth=3)
    lewa.set_xticks(range(n_blokow + 1), labels=["emb", *(f"b{i}" for i in range(1, n_blokow + 1))])
    lewa.set_ylim(0, 1)
    lewa.set_title("Prawdopodobieństwo prawidłowego następnego tokenu po kolejnych etapach")

    szerokosc = 0.35
    miejsca = list(range(1, n_blokow + 1))
    prawa.bar(
        [m - szerokosc / 2 for m in miejsca],
        [n["uwaga"] for n in normy],
        szerokosc,
        color=KOLORY["glowny"],
        label="uwaga",
    )
    prawa.bar(
        [m + szerokosc / 2 for m in miejsca],
        [n["mlp"] for n in normy],
        szerokosc,
        color=KOLORY["drugi"],
        label="MLP",
    )
    prawa.plot(
        miejsca,
        [n["reszta"] for n in normy],
        color=KOLORY["szary"],
        marker="o",
        label="strumień po bloku",
    )
    prawa.set_xticks(miejsca, labels=[f"b{i}" for i in miejsca])
    prawa.legend()
    prawa.set_title("Wkład uwagi i MLP do strumienia rezydualnego (średnia norma)")
    return zapisz_png(fig, ctx.wyniki_dir / "logit_lens_podsumowanie.png")


def zrob(ctx: KontekstI) -> list[Path]:
    """Tworzy logit_lens.gif/.png, logit_lens_podsumowanie.png i logit_lens.json.

    Jedno wywołanie modelu na ZDANIE z hakami zbierającymi stan po embeddingach (etap 0),
    po każdym bloku (etapy 1..n_layer) oraz wyjścia uwagi i MLP każdego bloku. Każdy stan
    przechodzi przez ln_f i lm_head tego samego modelu (logit lens).
    Zwraca [gif, png_podglad, png_podsumowanie, json] (w tej kolejności).
    Przykład: zrob(maly_kontekst_i(tmp)) -> 4 pliki; GIF ma n_layer + 1 klatek (3 dla 2 warstw)
    """
    model = ctx.model
    n_layer = len(model.blocks)
    ids_tensor = zakoduj_zdanie(ctx.tokenizer, ZDANIE, model.config.block_size).to(ctx.device)
    ids = ids_tensor[0].tolist()
    etykiety = [etykieta_tokenu(ctx.tokenizer, i) for i in ids]

    zebrane = _zbierz(model, ids_tensor)
    logity = _logity_etapow(model, zebrane)
    etapy = [_dane_etapu(ctx, logity[s], ids, s, n_layer) for s in range(n_layer + 1)]
    srednie = [
        sum(p["p_prawdziwego"] for p in e["pozycje"][:-1]) / max(len(ids) - 1, 1) for e in etapy
    ]
    normy = [
        {
            "blok": i + 1,
            "uwaga": _srednia_norma(zebrane.uwaga[i]),
            "mlp": _srednia_norma(zebrane.mlp[i]),
            "reszta": _srednia_norma(zebrane.stany[i + 1]),
        }
        for i in range(n_layer)
    ]

    fig, osie = nowa_figura(11, 0.4 * len(ids) + 1.8)

    def rysuj_klatke(s: int) -> None:
        _rysuj_mape(osie[0, 0], etapy[s], etykiety)

    gif, png_podglad = zapisz_gif(
        fig, rysuj_klatke, n_layer + 1, ctx.wyniki_dir / "logit_lens.gif", fps=1, klatka_png=n_layer
    )
    png_podsumowanie = _rysuj_podsumowanie(ctx, etapy, srednie, normy)
    plik_json = zapisz_json(
        {
            "material": "logit_lens",
            "zdanie": ZDANIE,
            "tokeny": etykiety,
            "etapy": etapy,
            "srednia_p_prawdziwego": srednie,
            "normy_blokow": normy,
        },
        ctx.wyniki_dir / "logit_lens.json",
    )
    return [gif, png_podglad, png_podsumowanie, plik_json]


def main(argv: list[str] | None = None) -> int:
    """Uruchamia materiał samodzielnie: uv run python -m przewodnik.logit_lens.

    Przykład: main(["--device", "cpu"]) -> 0 (albo 1 z komunikatem, gdy brak plików w data/)
    """
    return uruchom_samodzielnie(zrob, 1, "Materiał 5: logit lens", argv)


if __name__ == "__main__":
    raise SystemExit(main())
