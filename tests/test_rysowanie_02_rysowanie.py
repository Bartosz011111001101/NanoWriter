"""Testy modułu przewodnik.rysowanie oraz zmian w plikach konfiguracji."""

import re
import tomllib
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from PIL import Image

from przewodnik.rysowanie import (
    KOLORY,
    nowa_figura,
    rysuj_pudelka_tokenow,
    zapisz_gif,
    zapisz_png,
)

KORZEN = Path(__file__).resolve().parents[1]

PNG_NAGLOWEK = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(autouse=True)
def _zamknij_figury():
    yield
    plt.close("all")


def zrob_animacje(n):
    """Zwraca (fig, rysuj_klatke, wywolania); każda klatka ma słupek o innej wysokości."""
    fig, ax = nowa_figura(4, 3)
    wywolania = []

    def rysuj_klatke(i):
        wywolania.append(i)
        ax[0, 0].clear()
        ax[0, 0].set_ylim(0, 5)
        ax[0, 0].bar([0], [i + 1])

    return fig, rysuj_klatke, wywolania


def zrob_pudelka(szerokosc=6, wysokosc=3):
    fig, ax = nowa_figura(szerokosc, wysokosc)
    return fig, ax[0, 0]


def rozmiar_obrazu(sciezka):
    with Image.open(sciezka) as obraz:
        return obraz.size


def test_nowa_figura_ksztalt_osi_i_rozmiar():
    fig, ax = nowa_figura(4, 3)
    assert isinstance(fig, Figure)
    assert isinstance(ax, np.ndarray)
    assert ax.shape == (1, 1)
    assert isinstance(ax[0, 0], Axes)
    assert tuple(fig.get_size_inches()) == pytest.approx((4, 3))

    assert nowa_figura(4, 3, 2, 3)[1].shape == (2, 3)
    assert nowa_figura(8, 3, 1, 2)[1].shape == (1, 2)
    assert nowa_figura(3, 6, 3, 1)[1].shape == (3, 1)

    _, ax = nowa_figura(4, 3, 2, 3)
    assert all(isinstance(a, Axes) for a in ax.flat)
    assert len({id(a) for a in ax.flat}) == 6


def test_nowa_figura_backend_agg_i_uklad_constrained():
    fig, _ = nowa_figura(4, 3)
    assert matplotlib.get_backend().lower() == "agg"
    assert type(fig.get_layout_engine()).__name__ == "ConstrainedLayoutEngine"


def test_zapisz_png_plik_naglowek_rozmiar(tmp_path):
    fig, ax = nowa_figura(4, 3)
    ax[0, 0].plot([0, 1], [0, 1])
    plik = tmp_path / "a.png"
    wynik = zapisz_png(fig, plik)
    assert wynik == plik
    assert plik.exists()
    assert plik.stat().st_size > 0
    assert plik.read_bytes()[:8] == PNG_NAGLOWEK
    assert rozmiar_obrazu(plik) == (480, 360)
    assert not plt.fignum_exists(fig.number)

    fig2, ax2 = nowa_figura(4, 3)
    ax2[0, 0].plot([0, 1], [0, 1])
    plik2 = tmp_path / "b.png"
    zapisz_png(fig2, plik2, dpi=60)
    assert rozmiar_obrazu(plik2) == (240, 180)


def test_zapisz_png_tworzy_katalogi(tmp_path):
    fig, ax = nowa_figura(4, 3)
    ax[0, 0].plot([0, 1], [0, 1])
    plik = tmp_path / "w" / "x" / "b.png"
    zapisz_png(fig, plik)
    assert plik.exists()
    assert (tmp_path / "w" / "x").is_dir()


def test_zapisz_gif_liczba_klatek_i_zwrot(tmp_path):
    fig, f, wywolania = zrob_animacje(3)
    plik = tmp_path / "a.gif"
    wynik = zapisz_gif(fig, f, 3, plik)
    assert wynik == [plik]
    assert plik.exists()
    with Image.open(plik) as obraz:
        assert obraz.n_frames == 3
        assert obraz.format == "GIF"
    assert wywolania == [0, 1, 2]
    assert not plt.fignum_exists(fig.number)
    assert list(tmp_path.glob("*.png")) == []

    fig1, f1, _ = zrob_animacje(1)
    jedna = tmp_path / "jedna.gif"
    wynik1 = zapisz_gif(fig1, f1, 1, jedna)
    assert wynik1 == [jedna]
    with Image.open(jedna) as obraz:
        assert obraz.n_frames == 1


def test_zapisz_gif_podglad_png(tmp_path):
    fig, f, wywolania = zrob_animacje(3)
    gif = tmp_path / "a.gif"
    wynik = zapisz_gif(fig, f, 3, gif, klatka_png=2)
    assert wynik == [gif, tmp_path / "a.png"]
    assert gif.exists()
    assert (tmp_path / "a.png").exists()
    with Image.open(gif) as obraz:
        assert obraz.n_frames == 3
    assert (tmp_path / "a.png").read_bytes()[:4] == b"\x89PNG"
    assert rozmiar_obrazu(tmp_path / "a.png") == (480, 360)
    assert wywolania == [0, 1, 2, 2]

    fig_b, f_b, wywolania_b = zrob_animacje(3)
    wynik_b = zapisz_gif(fig_b, f_b, 3, tmp_path / "b.gif", klatka_png=0)
    assert wywolania_b == [0, 1, 2, 0]
    assert wynik_b == [tmp_path / "b.gif", tmp_path / "b.png"]

    fig_c, f_c, _ = zrob_animacje(3)
    zapisz_gif(fig_c, f_c, 3, tmp_path / "c.gif", klatka_png=2)
    fig_d, f_d, _ = zrob_animacje(3)
    zapisz_gif(fig_d, f_d, 3, tmp_path / "d.gif", klatka_png=0)
    assert (tmp_path / "c.png").read_bytes() != (tmp_path / "d.png").read_bytes()


def test_zapisz_gif_rozmiar_klatek_dpi_i_czas(tmp_path):
    fig, f, _ = zrob_animacje(3)
    gif = tmp_path / "a.gif"
    zapisz_gif(fig, f, 3, gif)
    with Image.open(gif) as obraz:
        for k in range(obraz.n_frames):
            obraz.seek(k)
            assert obraz.size == (320, 240)
    with Image.open(gif) as obraz:
        assert obraz.info["duration"] == 500
        assert obraz.info.get("loop") == 0

    fig2, f2, _ = zrob_animacje(3)
    gif2 = tmp_path / "b.gif"
    zapisz_gif(fig2, f2, 3, gif2, dpi=40)
    with Image.open(gif2) as obraz:
        assert obraz.size == (160, 120)

    fig3, f3, _ = zrob_animacje(3)
    gif3 = tmp_path / "c.gif"
    zapisz_gif(fig3, f3, 3, gif3, fps=5.0)
    with Image.open(gif3) as obraz:
        assert obraz.info["duration"] == 200


def test_zapisz_gif_katalogi_nadrzedne(tmp_path):
    fig, f, _ = zrob_animacje(2)
    katalog = tmp_path / "g" / "h"
    zapisz_gif(fig, f, 2, katalog / "a.gif", klatka_png=1)
    assert (katalog / "a.gif").exists()
    assert (katalog / "a.png").exists()


@pytest.mark.parametrize(
    ("liczba_klatek", "klatka_png"),
    [(0, None), (-1, None), (3, 3), (3, -1)],
)
def test_zapisz_gif_bledy_bez_rysowania_i_plikow(tmp_path, liczba_klatek, klatka_png):
    fig, f, wywolania = zrob_animacje(3)
    gif = tmp_path / "a.gif"
    with pytest.raises(ValueError):
        zapisz_gif(fig, f, liczba_klatek, gif, klatka_png=klatka_png)
    assert wywolania == []
    assert not gif.exists()


def test_rysuj_pudelka_liczba_polozenie_zakres():
    _, ax = zrob_pudelka()
    rysuj_pudelka_tokenow(ax, ["a", "b", "c", "d", "e"], w_wierszu=2)
    assert len(ax.patches) == 5
    assert ax.get_xlim() == pytest.approx((0, 2))
    assert ax.get_ylim() == pytest.approx((3, 0))
    for i, pudelko in enumerate(ax.patches):
        wiersz, kolumna = i // 2, i % 2
        assert kolumna <= pudelko.get_x() < kolumna + 1
        assert wiersz <= pudelko.get_y() < wiersz + 1
        assert 0.8 <= pudelko.get_width() <= 1.0
        assert 0.8 <= pudelko.get_height() <= 1.0
    teksty = {t.get_text(): t for t in ax.texts}
    for i, etykieta in enumerate(["a", "b", "c", "d", "e"]):
        wiersz, kolumna = i // 2, i % 2
        x, y = teksty[etykieta].get_position()
        assert kolumna <= x <= kolumna + 1
        assert wiersz <= y <= wiersz + 1


def test_rysuj_pudelka_zakresy_dla_innych_liczb():
    _, ax = zrob_pudelka()
    rysuj_pudelka_tokenow(ax, ["a", "b", "c"])
    assert ax.get_xlim() == pytest.approx((0, 20))
    assert ax.get_ylim() == pytest.approx((1, 0))
    assert len(ax.patches) == 3

    _, ax = zrob_pudelka()
    rysuj_pudelka_tokenow(ax, ["a"] * 20, w_wierszu=20)
    assert ax.get_ylim() == pytest.approx((1, 0))

    _, ax = zrob_pudelka()
    rysuj_pudelka_tokenow(ax, ["a"] * 21, w_wierszu=20)
    assert ax.get_ylim() == pytest.approx((2, 0))

    _, ax = zrob_pudelka()
    rysuj_pudelka_tokenow(ax, ["a"], w_wierszu=1)
    assert ax.get_xlim() == pytest.approx((0, 1))
    assert ax.get_ylim() == pytest.approx((1, 0))
    assert len(ax.patches) == 1


def test_rysuj_pudelka_osie_wylaczone():
    _, ax = zrob_pudelka()
    rysuj_pudelka_tokenow(ax, ["a", "b", "c"])
    assert ax.axison is False


def test_rysuj_pudelka_pusta_lista():
    _, ax = zrob_pudelka()
    rysuj_pudelka_tokenow(ax, [])
    assert len(ax.patches) == 0
    assert len(ax.texts) == 0
    assert ax.get_xlim() == pytest.approx((0, 1))
    assert ax.get_ylim() == pytest.approx((0, 1))

    _, ax2 = zrob_pudelka()
    rysuj_pudelka_tokenow(ax2, [], kolory=[], podpisy=[])


def test_rysuj_pudelka_kolory():
    _, ax = zrob_pudelka()
    kolory = ["#ff0000", "#00ff00", "#0000ff"]
    rysuj_pudelka_tokenow(ax, ["a", "b", "c"], kolory=kolory)
    for i, kolor in enumerate(kolory):
        assert to_hex(ax.patches[i].get_facecolor()).lower() == kolor.lower()

    _, ax2 = zrob_pudelka()
    rysuj_pudelka_tokenow(ax2, ["a", "b", "c", "d"], w_wierszu=4)
    domyslne = [to_hex(p.get_facecolor()).lower() for p in ax2.patches]
    assert domyslne[0] == KOLORY["tlo_a"].lower()
    assert domyslne[1] == KOLORY["tlo_b"].lower()
    for i in range(3):
        assert domyslne[i] != domyslne[i + 1]


def test_rysuj_pudelka_teksty_i_spacje():
    _, ax = zrob_pudelka()
    rysuj_pudelka_tokenow(ax, [" kot", "a b", "ż"])
    assert sorted(t.get_text() for t in ax.texts) == sorted(["·kot", "a·b", "ż"])
    assert len(ax.texts) == 3

    _, ax2 = zrob_pudelka()
    rysuj_pudelka_tokenow(ax2, ["x", "y"])
    assert len(ax2.texts) == 2


def test_rysuj_pudelka_podpisy():
    _, ax = zrob_pudelka()
    rysuj_pudelka_tokenow(ax, ["aa", "bb"], podpisy=["p1", "p2"], w_wierszu=2)
    assert len(ax.texts) == 4
    assert {t.get_text() for t in ax.texts} == {"aa", "bb", "p1", "p2"}
    assert len(ax.patches) == 2
    teksty = {t.get_text(): t for t in ax.texts}
    for kolumna, (etykieta, podpis) in enumerate([("aa", "p1"), ("bb", "p2")]):
        te = teksty[etykieta]
        tp = teksty[podpis]
        xe, ye = te.get_position()
        xp, yp = tp.get_position()
        assert ye < yp
        assert 0 <= ye <= 1
        assert 0 <= yp <= 1
        assert kolumna <= xe <= kolumna + 1
        assert kolumna <= xp <= kolumna + 1
        assert tp.get_fontsize() < te.get_fontsize()


def test_rysuj_pudelka_rozmiary_czcionek():
    _, ax = zrob_pudelka(10, 3)
    rysuj_pudelka_tokenow(ax, ["abcd", "abcde", "abcdefgh", "abcdefghi"], fontsize=8.0)
    rozmiary = {t.get_text(): t.get_fontsize() for t in ax.texts}
    assert rozmiary["abcd"] == pytest.approx(8.0, abs=1e-6)
    assert rozmiary["abcde"] == pytest.approx(6.0, abs=1e-6)
    assert rozmiary["abcdefgh"] == pytest.approx(6.0, abs=1e-6)
    assert rozmiary["abcdefghi"] == pytest.approx(4.4, abs=1e-6)

    for etykieta, oczekiwany in [
        ("a", 10.0),
        ("abcdef", 7.5),
        ("abcdefghijkl", 5.5),
        (" kot", 10.0),
    ]:
        _, ax = zrob_pudelka()
        rysuj_pudelka_tokenow(ax, [etykieta], fontsize=10.0)
        assert ax.texts[0].get_fontsize() == pytest.approx(oczekiwany, abs=1e-6)


@pytest.mark.parametrize(
    "argumenty",
    [
        {"kolory": ["#ff0000"]},
        {"kolory": ["#ff0000", "#00ff00", "#0000ff"]},
        {"podpisy": ["p1"]},
        {"w_wierszu": 0},
        {"w_wierszu": -1},
    ],
)
def test_rysuj_pudelka_bledy(argumenty):
    _, ax = zrob_pudelka()
    with pytest.raises(ValueError):
        rysuj_pudelka_tokenow(ax, ["a", "b"], **argumenty)


def test_kolory_klucze_i_format():
    assert set(KOLORY) == {
        "glowny",
        "drugi",
        "dobry",
        "zly",
        "szary",
        "tlo_a",
        "tlo_b",
        "liczone",
        "nieliczone",
        "podswietlenie",
    }
    assert len(KOLORY) == 10
    for wartosc in KOLORY.values():
        assert re.fullmatch(r"#[0-9a-fA-F]{6}", wartosc)


def test_moduly_nic_nie_wypisuja(tmp_path, capsys):
    fig, ax = nowa_figura(4, 3)
    ax[0, 0].plot([0, 1], [0, 1])
    zapisz_png(fig, tmp_path / "a.png")

    fig2, f2, _ = zrob_animacje(3)
    zapisz_gif(fig2, f2, 3, tmp_path / "a.gif", klatka_png=1)

    _, ax3 = zrob_pudelka()
    rysuj_pudelka_tokenow(ax3, ["a", "b"], podpisy=["p", "q"])

    przechwycone = capsys.readouterr()
    assert przechwycone.out == ""
    assert przechwycone.err == ""


def test_pyproject_matplotlib_i_sekcje_bez_zmian():
    dane = tomllib.loads((KORZEN / "pyproject.toml").read_text(encoding="utf-8"))
    dev = dane["dependency-groups"]["dev"]
    for nazwa in ["matplotlib", "pytest", "ruff", "pylint", "import-linter"]:
        assert any(element.startswith(nazwa) for element in dev), nazwa
    assert dane["project"]["dependencies"] == ["torch>=2.7", "numpy"]
    assert dane["tool"]["uv"]["sources"]["torch"] == {"index": "pytorch-cu128"}
    assert dane["tool"]["ruff"]["line-length"] == 100
    assert dane["tool"]["ruff"]["lint"]["select"] == ["E", "F", "I", "B", "UP"]
    assert dane["tool"]["importlinter"]["root_package"] == "nanowriter"


def test_uv_lock_i_gitignore():
    blokada = (KORZEN / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "matplotlib"' in blokada.splitlines()
    wiersze = [w.strip() for w in (KORZEN / ".gitignore").read_text(encoding="utf-8").splitlines()]
    assert "przewodnik/wyniki/" in wiersze
    assert "Docs/" in wiersze
    assert "*.bin" in wiersze
