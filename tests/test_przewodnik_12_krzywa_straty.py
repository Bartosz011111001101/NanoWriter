import dataclasses
import json
import math

import pytest

from przewodnik.krzywa_straty import zrob
from przewodnik.proba import maly_kontekst_i


def _zapisz_log(sciezka, wiersze):
    """Zapisuje listę słowników jako wiersze JSON Lines (UTF-8, końce linii LF)."""
    with open(sciezka, "w", encoding="utf-8", newline="\n") as plik:
        for wiersz in wiersze:
            plik.write(json.dumps(wiersz) + "\n")
    return sciezka


def _wiersze_treningowe():
    """Pięć wierszy train_loss: kroki 0..40 co 10, strata 6.0 .. 4.0, lr 0.001."""
    return [{"step": s, "train_loss": 6.0 - 0.05 * s, "lr": 0.001} for s in (0, 10, 20, 30, 40)]


def _ctx_z_logiem(tmp_path, wiersze):
    """Kontekst próbny z podmienionym logiem treningu."""
    ctx = maly_kontekst_i(tmp_path)
    log = _zapisz_log(tmp_path / "wlasny_log.jsonl", wiersze)
    return dataclasses.replace(ctx, log_treningu=log)


def _wczytaj(sciezka):
    return json.loads(sciezka.read_text(encoding="utf-8"))


def test_pliki_nazwy_kolejnosc_i_format(tmp_path):
    ctx = maly_kontekst_i(tmp_path)

    pliki = zrob(ctx)

    assert [p.name for p in pliki] == ["krzywa_straty.png", "krzywa_straty.json"]
    for p in pliki:
        assert p.parent == ctx.wyniki_dir
        assert p.exists()
        assert p.stat().st_size > 0
    assert pliki[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    surowy = pliki[1].read_bytes()
    assert surowy.endswith(b"\n")
    assert b"\r\n" not in surowy


def test_json_z_logu_proby(tmp_path):
    ctx = maly_kontekst_i(tmp_path)

    dane = _wczytaj(zrob(ctx)[1])

    assert dane["material"] == "krzywa_straty"
    assert dane["kroki"] == [0, 10, 20, 30, 40]
    assert all(type(k) is int for k in dane["kroki"])
    assert dane["train_loss"] == pytest.approx([6.0, 5.5, 5.0, 4.5, 4.0], abs=1e-9)
    assert dane["lr"] == [0.001] * 5
    assert dane["val_kroki"] == [20, 40]
    assert dane["val_loss"] == pytest.approx([5.2, 4.2], abs=1e-9)
    assert dane["min_val_loss"] == pytest.approx(4.2, abs=1e-9)
    assert dane["koncowa_val_loss"] == pytest.approx(4.2, abs=1e-9)
    assert dane["ln_vocab"] == pytest.approx(math.log(320), abs=1e-9)


def test_minimum_rozne_od_ostatniej(tmp_path):
    wiersze = _wiersze_treningowe() + [
        {"step": 20, "val_loss": 3.0},
        {"step": 40, "val_loss": 3.5},
    ]
    ctx = _ctx_z_logiem(tmp_path, wiersze)

    dane = _wczytaj(zrob(ctx)[1])

    assert dane["min_val_loss"] == pytest.approx(3.0)
    assert dane["koncowa_val_loss"] == pytest.approx(3.5)
    assert dane["val_loss"] == pytest.approx([3.0, 3.5])


def test_log_bez_walidacji(tmp_path):
    ctx = _ctx_z_logiem(tmp_path, _wiersze_treningowe())

    pliki = zrob(ctx)
    dane = _wczytaj(pliki[1])

    assert pliki[0].exists()
    assert dane["val_kroki"] == []
    assert dane["val_loss"] == []
    assert dane["min_val_loss"] is None
    assert dane["koncowa_val_loss"] is None


def test_pusty_log_value_error(tmp_path):
    pusty = tmp_path / "pusty.jsonl"
    pusty.write_bytes(b"")
    ctx = dataclasses.replace(maly_kontekst_i(tmp_path), log_treningu=pusty)

    with pytest.raises(ValueError, match="strat treningowych"):
        zrob(ctx)

    tylko_val = _ctx_z_logiem(
        tmp_path, [{"step": 20, "val_loss": 3.0}, {"step": 40, "val_loss": 2.5}]
    )
    with pytest.raises(ValueError):
        zrob(tylko_val)


def test_brak_pliku_logu(tmp_path):
    ctx = dataclasses.replace(maly_kontekst_i(tmp_path), log_treningu=tmp_path / "nie_ma.jsonl")

    with pytest.raises(FileNotFoundError):
        zrob(ctx)


def test_dlugi_log(tmp_path):
    wiersze = []
    for i in range(200):
        wiersze.append({"step": i, "train_loss": round(6.0 - 0.01 * i, 4), "lr": 0.001})
        if i in (49, 99, 149, 199):
            wiersze.append({"step": i, "val_loss": round(5.0 - i / 100, 3)})
    ctx = _ctx_z_logiem(tmp_path, wiersze)

    pliki = zrob(ctx)
    dane = _wczytaj(pliki[1])

    assert len(dane["train_loss"]) == 200
    assert len(dane["kroki"]) == 200
    assert dane["kroki"][-1] == 199
    assert dane["val_kroki"] == [49, 99, 149, 199]
    assert dane["min_val_loss"] == pytest.approx(3.01)
    assert dane["koncowa_val_loss"] == pytest.approx(3.01)
    assert pliki[0].exists()
    assert pliki[0].stat().st_size > 0
