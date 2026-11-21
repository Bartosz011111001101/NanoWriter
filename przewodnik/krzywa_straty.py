"""Materiał 7: krzywa straty i harmonogram learning rate z logu treningu."""

import math
from pathlib import Path

from przewodnik.konteksty import KontekstI, uruchom_samodzielnie
from przewodnik.rysowanie import KOLORY, nowa_figura, zapisz_png
from przewodnik.wspolne import wczytaj_log, zapisz_json


def zrob(ctx: KontekstI) -> list[Path]:
    """Tworzy krzywa_straty.png i krzywa_straty.json w ctx.wyniki_dir z logu ctx.log_treningu.

    Wykres: u góry strata treningowa i walidacyjna oraz linia ln(vocab_size) (strata losowego
    zgadywania), na dole learning rate. Zwraca [png, json].
    Przykład: zrob(maly_kontekst_i(tmp)) -> 2 pliki; "ln_vocab" == ln(vocab_size) modelu
    Przykład: log bez żadnego wiersza train_loss -> ValueError
    """
    log = wczytaj_log(ctx.log_treningu)
    kroki = log["kroki"]
    if not kroki:
        raise ValueError("log nie zawiera strat treningowych")

    train_loss = log["train_loss"]
    lr = log["lr"]
    val_kroki = log["val_kroki"]
    val_loss = log["val_loss"]
    ln_vocab = math.log(ctx.model.config.vocab_size)
    max_krok = max(kroki)

    fig, osie = nowa_figura(10, 7, 2, 1)
    gora = osie[0, 0]
    dol = osie[1, 0]

    gora.plot(kroki, train_loss, color=KOLORY["glowny"], label="strata treningowa (jeden batch)")
    if val_kroki:
        gora.plot(
            val_kroki,
            val_loss,
            color=KOLORY["drugi"],
            marker="o",
            label="strata walidacyjna",
        )
    gora.axhline(
        ln_vocab,
        color=KOLORY["szary"],
        linestyle="--",
        label="losowe zgadywanie = ln(V)",
    )
    gora.legend()
    gora.set_ylabel("strata (entropia krzyżowa)")
    gora.set_xlim(0, max_krok)
    gora.set_title("Krzywa straty modelu 1M")

    dol.plot(kroki, lr, color=KOLORY["dobry"])
    dol.set_ylabel("learning rate")
    dol.set_xlabel("krok treningu")
    dol.set_xlim(0, max_krok)
    dol.set_title("Harmonogram learning rate")

    png = zapisz_png(fig, ctx.wyniki_dir / "krzywa_straty.png")
    dane = {
        "material": "krzywa_straty",
        "kroki": kroki,
        "train_loss": train_loss,
        "lr": lr,
        "val_kroki": val_kroki,
        "val_loss": val_loss,
        "ln_vocab": ln_vocab,
        "min_val_loss": min(val_loss) if val_loss else None,
        "koncowa_val_loss": val_loss[-1] if val_loss else None,
    }
    json_sciezka = zapisz_json(dane, ctx.wyniki_dir / "krzywa_straty.json")
    return [png, json_sciezka]


def main(argv: list[str] | None = None) -> int:
    """Uruchamia materiał samodzielnie: uv run python -m przewodnik.krzywa_straty.

    Przykład: main(["--device", "cpu"]) -> 0 (albo 1 z komunikatem, gdy brak plików w data/)
    """
    return uruchom_samodzielnie(zrob, 1, "Materiał 7: krzywa straty", argv)


if __name__ == "__main__":
    raise SystemExit(main())
