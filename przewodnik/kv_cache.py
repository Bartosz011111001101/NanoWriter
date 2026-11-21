"""Materiał 11: KV cache. Co jest trzymane w pamięci i ile czasu to oszczędza."""

import copy
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from nanowriter.model import KVCache
from nanowriter.sample import generate
from przewodnik.konteksty import KontekstI, uruchom_samodzielnie
from przewodnik.rysowanie import KOLORY, nowa_figura, rysuj_pudelka_tokenow, zapisz_gif, zapisz_png
from przewodnik.wspolne import ZDANIE, etykieta_tokenu, zakoduj_zdanie, zapisz_json

if TYPE_CHECKING:
    from nanowriter.model import GPT

DLUGOSCI_CZASU: tuple[int, ...] = (16, 48, 96, 160)
DLUGOSC_PROMPTU: int = 4
POWTORZENIA: int = 2


def _czas_generowania(
    model: "GPT", prompt: torch.Tensor, n: int, device: str, seed: int, use_cache: bool
) -> float:
    """Zwraca najkrótszy z POWTORZENIA czasów (s) wygenerowania n tokenów po prompcie.

    prompt: LongTensor[1, T] na device. Generator o danym seedzie, block_size modelu;
    na CUDA czas mierzony po torch.cuda.synchronize().
    Przykład: _czas_generowania(model, prompt, 16, "cpu", 0, True) -> liczba > 0
    """
    cuda = torch.device(device).type == "cuda"
    block_size = model.config.block_size

    def _jedno_wywolanie(liczba_tokenow: int) -> float:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        generate(
            model,
            prompt,
            liczba_tokenow,
            generator=generator,
            block_size=block_size,
            use_cache=use_cache,
        )
        if cuda:
            torch.cuda.synchronize()
        return time.perf_counter() - start

    _jedno_wywolanie(2)  # rozgrzewka, nie liczy się do czasu
    return min(_jedno_wywolanie(n) for _ in range(POWTORZENIA))


def _kroki_z_cache(
    model: "GPT", ids: torch.Tensor, etykiety: list[str]
) -> list[dict[str, int | float | str]]:
    """Podaje model po jednym tokenie z KVCache i porównuje z pełnym przeliczeniem prefiksu."""
    config = model.config
    bajtow_na_liczbe = model.tok_emb.weight.element_size()
    cache = KVCache()
    kroki: list[dict[str, int | float | str]] = []
    with torch.no_grad():
        for t in range(ids.shape[1]):
            logity_cache = model(ids[:, t : t + 1], cache=cache)[0, 0]
            logity_pelne = model(ids[:, : t + 1])[0, -1]
            roznica = float((logity_cache - logity_pelne).abs().max().item())
            dlugosc = cache.length
            kroki.append(
                {
                    "krok": t,
                    "token": etykiety[t],
                    "dlugosc_cache": dlugosc,
                    "bajty_cache": 2 * config.n_layer * dlugosc * config.d_model * bajtow_na_liczbe,
                    "tokeny_z_cache": 1,
                    "tokeny_bez_cache": t + 1,
                    "roznica_logitow": roznica,
                }
            )
    return kroki


def _animacja(
    ctx: KontekstI, etykiety: list[str], kroki: list[dict[str, int | float | str]]
) -> list[Path]:
    """Rysuje GIF z pamięcią cache w kolejnych krokach; zwraca [gif, png_podglad]."""
    liczba = len(etykiety)
    n_layer = ctx.model.config.n_layer
    fig, osie = nowa_figura(11, 4.8, 1, 2)
    ax_lewa, ax_prawa = osie[0, 0], osie[0, 1]

    def rysuj_klatke(t: int) -> None:
        ax_lewa.clear()
        ax_prawa.clear()
        krok = kroki[t]
        kolory = [
            KOLORY["tlo_a"]
            if j < t
            else KOLORY["podswietlenie"]
            if j == t
            else KOLORY["nieliczone"]
            for j in range(liczba)
        ]
        podpisy = ["w cache" if j < t else "liczona teraz" if j == t else "" for j in range(liczba)]
        rysuj_pudelka_tokenow(ax_lewa, etykiety, kolory=kolory, podpisy=podpisy, w_wierszu=liczba)
        ax_lewa.set_title(f"Pamięć K i V w każdej z {n_layer} warstw: {t + 1} z {liczba} pozycji")

        ax_prawa.barh(
            ["bez cache: tokenów liczonych w tym kroku", "z KV cache"],
            [t + 1, 1],
            color=[KOLORY["zly"], KOLORY["dobry"]],
        )
        ax_prawa.set_xlim(0, liczba)
        ax_prawa.invert_yaxis()
        ax_prawa.text(
            0.5,
            -0.12,
            f"pamięć cache: {float(krok['bajty_cache']) / 1024:.1f} KiB\n"
            f"różnica logitów względem pełnego przeliczenia: {float(krok['roznica_logitow']):.1e}",
            transform=ax_prawa.transAxes,
            ha="center",
            va="top",
        )
        fig.suptitle(f"Krok {t + 1} z {liczba}: nowy token „{etykiety[t]}”")

    return zapisz_gif(
        fig, rysuj_klatke, liczba, ctx.wyniki_dir / "kv_cache.gif", fps=1.5, klatka_png=liczba // 2
    )


def _pomiary_czasu(
    ctx: KontekstI, ids: torch.Tensor, dlugosci: list[int]
) -> list[dict[str, object]]:
    """Mierzy czasy generowania z cache i bez na ctx.device oraz (gdy to nie CPU) na CPU."""
    prompt_ids = ids[:, :DLUGOSC_PROMPTU]
    urzadzenia: list[tuple[str, GPT]] = [(ctx.device, ctx.model)]
    kopia_cpu: GPT | None = None
    if torch.device(ctx.device).type != "cpu":
        kopia_cpu = copy.deepcopy(ctx.model).to("cpu")
        urzadzenia.append(("cpu", kopia_cpu))

    wynik: list[dict[str, object]] = []
    stan_treningu = ctx.model.training
    try:
        for urzadzenie, model in urzadzenia:
            model.eval()
            prompt = prompt_ids.to(urzadzenie)
            bez_cache = [
                _czas_generowania(model, prompt, n, urzadzenie, ctx.seed, False) for n in dlugosci
            ]
            z_cache = [
                _czas_generowania(model, prompt, n, urzadzenie, ctx.seed, True) for n in dlugosci
            ]
            wynik.append(
                {
                    "urzadzenie": urzadzenie,
                    "dlugosci": list(dlugosci),
                    "bez_cache_s": bez_cache,
                    "z_cache_s": z_cache,
                }
            )
    finally:
        ctx.model.train(stan_treningu)
    return wynik


def _wykres_czasow(czasy: list[dict[str, object]], sciezka: Path) -> Path:
    """Rysuje czasy generowania z cache i bez, osobna linia dla każdego urządzenia."""
    fig, osie = nowa_figura(8, 5)
    ax = osie[0, 0]
    for indeks, wpis in enumerate(czasy):
        styl = "-" if indeks == 0 else "--"
        urzadzenie = wpis["urzadzenie"]
        ax.plot(
            wpis["dlugosci"],
            wpis["bez_cache_s"],
            styl,
            marker="o",
            color=KOLORY["zly"],
            label=f"{urzadzenie}, bez cache",
        )
        ax.plot(
            wpis["dlugosci"],
            wpis["z_cache_s"],
            styl,
            marker="o",
            color=KOLORY["dobry"],
            label=f"{urzadzenie}, z cache",
        )
    ax.set_xlabel("liczba generowanych tokenów")
    ax.set_ylabel("czas (s)")
    ax.set_title("Czas generowania: z KV cache i bez")
    ax.legend()
    return zapisz_png(fig, sciezka)


def zrob(ctx: KontekstI) -> list[Path]:
    """Tworzy kv_cache.gif/.png, kv_cache_czas.png i kv_cache.json w ctx.wyniki_dir.

    Animacja: zdanie ZDANIE podawane modelowi po jednym tokenie z prawdziwym KVCache; w kroku
    t cache ma t + 1 pozycji, model liczy 1 token zamiast t + 1. Dowód: różnica logitów kroku z
    cache i pełnego przeliczenia prefiksu. Czasy: generate z cache i bez dla DLUGOSCI_CZASU
    na ctx.device oraz (gdy to nie CPU) na kopii modelu na CPU.
    Zwraca [gif, png_podglad, png_czas, json] (w tej kolejności).
    Przykład: zrob(maly_kontekst_i(tmp)) -> 4 pliki; GIF ma tyle klatek, ile tokenów w zdaniu
    """
    block_size = ctx.model.config.block_size
    dlugosci = [n for n in DLUGOSCI_CZASU if DLUGOSC_PROMPTU + n <= block_size]
    if not dlugosci:
        raise ValueError(
            f"block_size modelu ({block_size}) jest za mały: potrzeba co najmniej "
            f"{DLUGOSC_PROMPTU + min(DLUGOSCI_CZASU)} pozycji"
        )

    ids = zakoduj_zdanie(ctx.tokenizer, ZDANIE, block_size).to(ctx.device)
    etykiety = [etykieta_tokenu(ctx.tokenizer, int(i)) for i in ids[0].tolist()]

    stan_treningu = ctx.model.training
    ctx.model.eval()
    try:
        kroki = _kroki_z_cache(ctx.model, ids, etykiety)
    finally:
        ctx.model.train(stan_treningu)
    gif, png_podglad = _animacja(ctx, etykiety, kroki)

    czasy = _pomiary_czasu(ctx, ids, dlugosci)
    png_czas = _wykres_czasow(czasy, ctx.wyniki_dir / "kv_cache_czas.png")

    json_sciezka = zapisz_json(
        {"material": "kv_cache", "zdanie": ZDANIE, "kroki": kroki, "czasy": czasy},
        ctx.wyniki_dir / "kv_cache.json",
    )
    return [gif, png_podglad, png_czas, json_sciezka]


def main(argv: list[str] | None = None) -> int:
    """Uruchamia materiał samodzielnie: uv run python -m przewodnik.kv_cache.

    Przykład: main(["--device", "cpu"]) -> 0 (albo 1 z komunikatem, gdy brak plików w data/)
    """
    return uruchom_samodzielnie(zrob, 1, "Materiał 11: KV cache", argv)


if __name__ == "__main__":
    raise SystemExit(main())
