from collections import Counter

import pytest

from nanowriter.data import (
    ENGLISH_MARKERS,
    PROFANITY_STEMS,
    PROFANITY_WORDS,
    SubtitleFilter,
)

X = "Nie wiem, co robić."
Y = "Jak się masz?"


def _accept_all(f, lines):
    return [f.accept(line) for line in lines]


def test_stale_sa_krotkami_napisow():
    for const in (PROFANITY_STEMS, PROFANITY_WORDS, ENGLISH_MARKERS):
        assert isinstance(const, tuple)
        assert len(const) > 0
        assert all(isinstance(item, str) for item in const)
    assert "kurw" in PROFANITY_STEMS
    assert "dupa" in PROFANITY_WORDS
    assert "the" in ENGLISH_MARKERS


def test_nowy_filtr_ma_pusty_counts():
    f = SubtitleFilter()

    assert isinstance(f.counts, Counter)
    assert len(f.counts) == 0
    assert f.counts == Counter()


def test_przyklad_myslnik_dialogowy():
    f = SubtitleFilter()

    assert f.accept("- Dzień dobry, panie Kowalski.") == "Dzień dobry, panie Kowalski."
    assert dict(f.counts) == {"przyjete": 1}


def test_czyszczenie_brzegow_i_myslnika():
    assert SubtitleFilter().accept("  - Co tam?  ") == "Co tam?"


def test_czyszczenie_wielu_myslnikow_i_spacji():
    f = SubtitleFilter()

    assert f.accept("-- – Nie wiem  co robić.") == "Nie wiem co robić."
    assert f.accept("—Idę do domu.") == "Idę do domu."


def test_czyszczenie_zwija_biale_znaki():
    f = SubtitleFilter()

    assert f.accept("Ala\tma   kota") == "Ala ma kota"
    assert f.accept("Jestem  zmęczony.") == "Jestem zmęczony."


def test_czyszczenie_nfc():
    result = SubtitleFilter().accept("Café jest zamknięta.")

    assert result is not None
    assert [ord(c) for c in result] == [ord(c) for c in "Café jest zamknięta."]
    assert "é" in result
    assert "́" not in result


def test_zachowuje_wielkosc_liter_wyniku():
    assert SubtitleFilter().accept("Jestem Zmęczony Bardzo.") == "Jestem Zmęczony Bardzo."


def test_pusta_linia():
    f = SubtitleFilter()

    results = _accept_all(f, ["", "   ", "- -", "---"])

    assert results == [None, None, None, None]
    assert dict(f.counts) == {"pusta": 4}


def test_znaki_spoza_zestawu():
    f = SubtitleFilter()
    lines = [
        "/Chodź do nas./",
        "Ala #1",
        "♪ La la la ♪",
        "Jestem tu 😀 ok.",
        "Привет, как дела?",
    ]

    assert _accept_all(f, lines) == [None] * 5
    assert dict(f.counts) == {"znaki": 5}


def test_krotka_linia():
    f = SubtitleFilter()

    assert _accept_all(f, ["Tak.", "Nie.", "...", "!", "123"]) == [None] * 5
    assert dict(f.counts) == {"krotka": 5}


def test_krotka_linia_zalezy_od_min_words():
    assert SubtitleFilter(min_words=1).accept("Tak.") == "Tak."

    f = SubtitleFilter(min_words=3)
    assert f.accept("Nie wiem.") is None
    assert dict(f.counts) == {"krotka": 1}
    assert f.accept(Y) == Y


def test_wielkie_litery():
    f = SubtitleFilter()

    assert _accept_all(f, ["UWAŻAJ NA SIEBIE!", "TAK NIE", "ZADZWOŃ POD 112"]) == [None] * 3
    assert dict(f.counts) == {"wielkie": 3}

    assert SubtitleFilter().accept("Uwaga NA siebie.") == "Uwaga NA siebie."


def test_przeklenstwa_rdzen_w_srodku_slowa():
    f = SubtitleFilter()
    lines = [
        "Co za kurwa dzień.",
        "To jest jebany film.",
        "Co za KURWA dzień.",
        "To jest SHIT, naprawdę.",
        "Ale z ciebie dupa wołowa.",
        "Ta cholera znowu pada.",
    ]

    assert _accept_all(f, lines) == [None] * 6
    assert dict(f.counts) == {"przeklenstwa": 6}


@pytest.mark.parametrize("stem", PROFANITY_STEMS)
def test_kazdy_rdzen_przeklenstwa_odrzuca(stem):
    f = SubtitleFilter()

    assert f.accept(f"To jest {stem} dzień.") is None
    assert dict(f.counts) == {"przeklenstwa": 1}


@pytest.mark.parametrize("word", PROFANITY_WORDS)
def test_kazde_slowo_przeklenstwa_odrzuca(word):
    f = SubtitleFilter()

    assert f.accept(f"To jest {word} dzień.") is None
    assert dict(f.counts) == {"przeklenstwa": 1}


def test_slowa_przeklenstw_to_cale_slowa():
    f = SubtitleFilter()

    assert f.accept("Dupleks jest zepsuty.") == "Dupleks jest zepsuty."
    assert f.accept("Cholerka, zimno dziś.") == "Cholerka, zimno dziś."


def test_znaczniki_tworcow_napisow():
    f = SubtitleFilter()
    lines = [
        "Napisy: michnik",
        "Tłumaczenie: Jan Kowalski",
        "www.napisy24.pl",
        "Film 23.976 fps lektor",
        "Korekta: Anna Nowak",
    ]

    assert _accept_all(f, lines) == [None] * 5
    assert dict(f.counts) == {"znaczniki": 5}


def test_brak_falszywych_znacznikow():
    f = SubtitleFilter()

    assert f.accept("Napisał do mnie list.") == "Napisał do mnie list."
    assert f.accept("Jestem po napisaniu listu.") == "Jestem po napisaniu listu."


def test_angielski():
    f = SubtitleFilter()
    lines = [
        "Why am I so happy?",
        "What do you want?",
        "I don't know.",
        "It's a nice day.",
    ]

    assert _accept_all(f, lines) == [None] * 4
    assert dict(f.counts) == {"angielski": 4}


@pytest.mark.parametrize("marker", ENGLISH_MARKERS)
def test_kazdy_znacznik_angielski_odrzuca(marker):
    f = SubtitleFilter()

    assert f.accept(f"To jest {marker} dzień.") is None
    assert dict(f.counts) == {"angielski": 1}


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ("TO JEST ZŁE /", "znaki"),
        ("NIE.", "krotka"),
        ("KURWA MAĆ", "wielkie"),
        ("Napisy kurwa", "przeklenstwa"),
        ("The subtitles are good", "znaczniki"),
    ],
)
def test_kolejnosc_sprawdzen(line, reason):
    f = SubtitleFilter()

    assert f.accept(line) is None
    assert dict(f.counts) == {reason: 1}


def test_polskie_znaki_cyfry_interpunkcja_przechodza():
    f = SubtitleFilter()
    lines = [
        "Zażółć gęślą jaźń.",
        "Źdźbło trawy.",
        "Mam 5 lat.",
        "„Chodź tu”, powiedział (cicho).",
        "Czy to prawda?! Tak: 50% — chyba…",
    ]

    assert _accept_all(f, lines) == lines
    assert dict(f.counts) == {"przyjete": 5}


def test_powtorka_z_rzedu():
    f = SubtitleFilter()

    assert f.accept("Dzień dobry.") == "Dzień dobry."
    assert f.accept("Dzień dobry.") is None
    assert dict(f.counts) == {"przyjete": 1, "powtorka": 1}


def test_powtorka_bez_wzgledu_na_wielkosc_liter():
    f = SubtitleFilter()

    assert f.accept(X) == X
    assert f.accept("nie wiem, co robić.") is None
    assert f.accept("NIE WIEM, CO ROBIĆ") is None
    assert dict(f.counts) == {"wielkie": 1, "powtorka": 1, "przyjete": 1}


def test_odmienne_linie_nie_sa_powtorka():
    f = SubtitleFilter()

    assert f.accept("Co słychać?") == "Co słychać?"
    assert f.accept(Y) == Y
    assert dict(f.counts) == {"przyjete": 2}


def test_okno_powtorek_ma_dlugosc_window():
    f = SubtitleFilter(window=2)
    results = _accept_all(
        f, ["Pierwsza linia tutaj.", "Druga linia tutaj.", "Pierwsza linia tutaj."]
    )
    assert results == ["Pierwsza linia tutaj.", "Druga linia tutaj.", None]

    f = SubtitleFilter(window=2)
    lines = [
        "Pierwsza linia tutaj.",
        "Druga linia tutaj.",
        "Trzecia linia tutaj.",
        "Pierwsza linia tutaj.",
    ]
    assert _accept_all(f, lines) == lines


def test_domyslne_okno_20_granica():
    f = SubtitleFilter()
    assert f.accept(X) == X
    for n in range(1, 20):
        line = f"Linia numer {n} tutaj."
        assert f.accept(line) == line
    assert f.accept(X) is None

    f = SubtitleFilter()
    assert f.accept(X) == X
    for n in range(1, 21):
        line = f"Linia numer {n} tutaj."
        assert f.accept(line) == line
    assert f.accept(X) == X


def test_okno_zero_wylacza_powtorki_z_rzedu():
    f = SubtitleFilter(window=0)

    assert _accept_all(f, [X, X, X]) == [X, X, X]


def test_globalny_limit_powtorek_krotkich_linii():
    f = SubtitleFilter(window=0, max_repeats=2)

    assert _accept_all(f, [X, X, X, X]) == [X, X, None, None]
    assert dict(f.counts) == {"przyjete": 2, "powtorka": 2}


def test_globalny_limit_granica_piec_slow():
    five = "Nie mam na to czasu."
    f = SubtitleFilter(window=0, max_repeats=2)
    assert _accept_all(f, [five] * 3) == [five, five, None]

    six = "Nie mam już na to czasu."
    f = SubtitleFilter(window=0, max_repeats=2)
    assert _accept_all(f, [six] * 3) == [six] * 3
    assert dict(f.counts) == {"przyjete": 3}

    seven = "Ona mówi to samo już od godziny."
    f = SubtitleFilter(window=0, max_repeats=2)
    assert _accept_all(f, [seven] * 3) == [seven] * 3


def test_globalny_limit_bez_wielkosci_liter():
    f = SubtitleFilter(window=0, max_repeats=2)

    assert f.accept(X) == X
    assert f.accept(X.lower()) == "nie wiem, co robić."
    assert f.accept(X) is None


def test_odrzucona_linia_nie_zwieksza_licznika_powtorek():
    f = SubtitleFilter(window=1, max_repeats=2)

    assert _accept_all(f, [X, X, Y, X, Y, X]) == [X, None, Y, X, Y, None]


def test_suma_counts_rowna_liczbie_wywolan():
    f = SubtitleFilter()
    lines = [
        "- Dzień dobry, panie Kowalski.",
        "Nie.",
        "UWAŻAJ NA SIEBIE!",
        "Co za kurwa dzień.",
        "Why am I so happy?",
        "♪ La la la ♪",
        "Napisy: michnik",
        "Jestem  zmęczony.",
        "Jestem zmęczony.",
        "",
    ]

    assert _accept_all(f, lines) == [
        "Dzień dobry, panie Kowalski.",
        None,
        None,
        None,
        None,
        None,
        None,
        "Jestem zmęczony.",
        None,
        None,
    ]
    assert dict(f.counts) == {
        "przyjete": 2,
        "krotka": 1,
        "wielkie": 1,
        "przeklenstwa": 1,
        "angielski": 1,
        "znaki": 1,
        "znaczniki": 1,
        "powtorka": 1,
        "pusta": 1,
    }
    assert sum(f.counts.values()) == 10


def test_dwa_filtry_nie_dziela_stanu():
    f1 = SubtitleFilter()
    f2 = SubtitleFilter()

    assert f1.accept(X) == X
    assert f2.accept(X) == X
    assert f1.counts is not f2.counts
    assert dict(f2.counts) == {"przyjete": 1}


def test_poprawne_argumenty_konstruktora():
    SubtitleFilter(min_words=1)
    SubtitleFilter(window=0)
    SubtitleFilter(max_repeats=1)


@pytest.mark.parametrize(
    "kwargs",
    [{"min_words": 0}, {"window": -1}, {"max_repeats": 0}],
)
def test_bledne_argumenty_konstruktora(kwargs):
    with pytest.raises(ValueError):
        SubtitleFilter(**kwargs)
