"""Materiał 8: learning rate. Trzy krótkie treningi od tego samego startu."""

import math
from pathlib import Path

from przewodnik.konteksty import KontekstI, uruchom_samodzielnie
from przewodnik.krotki_trening import krotki_trening, nowy_model
from przewodnik.rysowanie import KOLORY, nowa_figura, zapisz_png
from przewodnik.wspolne import probka_tekstu, zapisz_json

PRZEBIEGI: tuple[tuple[str, float], ...] = (("za mały", 1e-5), ("dobry", 2e-3), ("za duży", 1e-1))
PROMPT: str = "Wieczorem stary kot"
LICZBA_TOKENOW: int = 40


def _koncowa_strata(train_loss: list[float]) -> float | None:
    """Średnia z ostatnich 10 wartości train_loss z pominięciem NaN; brak skończonych -> None.

    Przykład: _koncowa_strata([5.0, 3.0]) -> 4.0
    Przykład: _koncowa_strata([float("nan")]) -> None
    """
    ostatnie = [strata for strata in train_loss[-10:] if math.isfinite(strata)]
    if not ostatnie:
        return None
    return sum(ostatnie) / len(ostatnie)


def _srednia_kroczaca(wartosci: list[float], okno: int) -> list[float]:
    """Zwraca trailing średnią kroczącą; okno zawierające NaN daje NaN (przerwa w linii).

    Przykład: _srednia_kroczaca([1.0, 3.0, 5.0], 2) -> [1.0, 2.0, 4.0]
    Przykład: _srednia_kroczaca([1.0, float("nan"), 5.0], 2) -> [1.0, nan, nan]
    """
    wynik: list[float] = []
    for i in range(len(wartosci)):
        fragment = wartosci[max(0, i - okno + 1) : i + 1]
        if any(math.isnan(strata) for strata in fragment):
            wynik.append(math.nan)
        else:
            wynik.append(sum(fragment) / len(fragment))
    return wynik


def _narysuj_wykres(
    przebiegi: list[tuple[str, float, list[float]]],
    okno: int,
    vocab_size: int,
    sciezka: Path,
) -> Path:
    """Rysuje straty przebiegów do PNG: cienka linia kroków i gruba średnia krocząca.

    Przykład: _narysuj_wykres([("dobry", 2e-3, [6.0, 5.0])], 1, 320, p) -> p (plik PNG)
    Przykład: nazwa spoza "za mały", "dobry", "za duży" -> KeyError
    """
    kolory = {
        "za mały": KOLORY["szary"],
        "dobry": KOLORY["dobry"],
        "za duży": KOLORY["zly"],
    }
    fig, osie = nowa_figura(10, 5.5)
    ax = osie[0, 0]
    for nazwa, lr, straty in przebiegi:
        kolor = kolory[nazwa]
        kroki = list(range(len(straty)))
        ax.plot(kroki, straty, color=kolor, alpha=0.3, linewidth=0.8)
        ax.plot(
            kroki,
            _srednia_kroczaca(straty, okno),
            color=kolor,
            linewidth=2.5,
            label=f"{nazwa} (lr = {lr:g})",
        )
    ax.set_ylim(0, 1.3 * math.log(vocab_size))
    ax.set_xlabel("krok treningu")
    ax.set_ylabel("strata")
    ax.set_title(
        "Wpływ learning rate na uczenie "
        "(ten sam start, te same dane; oś strat obcięta)"
    )
    ax.legend()
    return zapisz_png(fig, sciezka)


def zrob(ctx: KontekstI) -> list[Path]:
    """Tworzy learning_rate.png i learning_rate.json w ctx.wyniki_dir.

    Dla każdego z PRZEBIEGI: nowy model z tym samym seedem (te same wagi startowe), trening
    ctx.kroki_lr kroków z danym learning rate (te same batche), próbka tekstu po treningu.
    Zwraca [png, json]. Strata kroku 0 jest identyczna we wszystkich trzech przebiegach.
    Przykład: zrob(maly_kontekst_i(tmp)) -> 2 pliki; "przebiegi" ma 3 elementy
    """
    przebiegi_wykresu: list[tuple[str, float, list[float]]] = []
    przebiegi_json: list[dict] = []
    for nazwa, lr in PRZEBIEGI:
        model = nowy_model(ctx.model.config, ctx.seed, ctx.device)
        wynik = krotki_trening(
            model,
            ctx.dane_train,
            ctx.dane_val,
            kroki=ctx.kroki_lr,
            lr=lr,
            batch_size=ctx.batch_size,
            device=ctx.device,
            seed=ctx.seed,
            co_ile=max(1, ctx.kroki_lr // 10),
        )
        try:
            tekst = probka_tekstu(
                model,
                ctx.tokenizer,
                PROMPT,
                LICZBA_TOKENOW,
                device=ctx.device,
                seed=ctx.seed,
            )
        except RuntimeError:
            tekst = "(generowanie niemożliwe: model rozbieżny)"

        przebiegi_wykresu.append((nazwa, lr, wynik.train_loss))
        przebiegi_json.append(
            {
                "nazwa": nazwa,
                "lr": lr,
                "train_loss": wynik.train_loss,
                "val_kroki": wynik.val_kroki,
                "val_loss": wynik.val_loss,
                "koncowa_strata": _koncowa_strata(wynik.train_loss),
                "tekst": tekst,
                "czas_s": wynik.czas_s,
            }
        )

    okno = max(1, ctx.kroki_lr // 25)
    png = _narysuj_wykres(
        przebiegi_wykresu,
        okno,
        ctx.model.config.vocab_size,
        ctx.wyniki_dir / "learning_rate.png",
    )
    json_plik = zapisz_json(
        {"material": "learning_rate", "kroki": ctx.kroki_lr, "przebiegi": przebiegi_json},
        ctx.wyniki_dir / "learning_rate.json",
    )
    return [png, json_plik]


def main(argv: list[str] | None = None) -> int:
    """Uruchamia materiał samodzielnie: uv run python -m przewodnik.learning_rate.

    Przykład: main(["--device", "cpu", "--kroki-lr", "20"]) -> 0 (albo 1, gdy brak plików)
    """
    return uruchom_samodzielnie(zrob, 1, "Materiał 8: learning rate", argv)


if __name__ == "__main__":
    raise SystemExit(main())
