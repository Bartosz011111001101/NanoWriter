"""Testy build_context, extract_reply i chat_reply z nanowriter.sample (CPU, kilka sekund)."""

import copy

import pytest
import torch

from nanowriter.model import GPT, BigramLanguageModel, GPTConfig
from nanowriter.sample import build_context, chat_reply, extract_reply
from nanowriter.tokenizer import BPETokenizer, CharTokenizer

TOK = BPETokenizer([])
TOK_BPE = BPETokenizer.train("A: ala ma kota\nB: kot ma ale\n" * 40, 280)
TOK_CHAR = CharTokenizer.from_text("A: B:\nabxy ")

# id bajtów: ':'=58, ' '=32, 'o'=111, 'k'=107, '\n'=10, 'A'=65
TABELA_OK = [(58, 32), (32, 111), (111, 107), (107, 10), (10, 65), (65, 58)]
TABELA_PUSTA = [(58, 10), (10, 65), (65, 58)]


class _Atrapa(BigramLanguageModel):
    """Bigram z wyzerowaną tabelą i zapamiętywaniem wejść modelu."""

    def __init__(self, tabela: list[tuple[int, int]]) -> None:
        super().__init__(256)
        self.wejscia: list[torch.Tensor] = []
        with torch.no_grad():
            self.token_embedding.weight.zero_()
            for z, do in tabela:
                self.token_embedding.weight[z, do] = 10.0
        self.eval()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        self.wejscia.append(idx.clone())
        return super().forward(idx)


def _atrapa(tabela: list[tuple[int, int]]) -> _Atrapa:
    return _Atrapa(tabela)


def _gpt() -> GPT:
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=256, block_size=32, n_layer=2, n_head=2, d_model=32)
    return GPT(cfg).eval()


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def tekst(history: list[tuple[str, str]], user: str) -> str:
    return "".join(f"A: {u}\nB: {r}\n" for u, r in history) + f"A: {user}\nB:"


# ---------------------------------------------------------------- build_context


def test_build_context_przyklady_z_opisu():
    h = [("x", "y")]
    wyniki = [
        (build_context(TOK, h, "ab", 18), TOK.encode("A: x\nB: y\nA: ab\nB:")),
        (build_context(TOK, h, "ab", 17), TOK.encode("A: ab\nB:")),
        (build_context(TOK, h, "ab", 7), TOK.encode("A: b\nB:")),
        (build_context(TOK, h, "ab", 6), TOK.encode("A: \nB:")),
    ]
    assert len(wyniki[0][1]) == 18
    assert len(wyniki[1][1]) == 8
    for wynik, oczekiwany in wyniki:
        assert wynik == oczekiwany
        assert type(wynik) is list
        assert all(type(i) is int for i in wynik)
    with pytest.raises(ValueError):
        build_context(TOK, h, "ab", 5)


def test_build_context_bez_historii_i_max_turns_zero():
    oczekiwany = TOK.encode("A: ab\nB:")
    assert build_context(TOK, [], "ab", 100) == oczekiwany
    h = [("x", "y"), ("p", "q")]
    assert build_context(TOK, h, "ab", 100, max_turns=0) == oczekiwany


def test_build_context_max_turns_ogranicza_historie():
    h = [("a1", "b1"), ("a2", "b2"), ("a3", "b3"), ("a4", "b4"), ("a5", "b5")]
    assert build_context(TOK, h, "q", 1000, max_turns=2) == TOK.encode(tekst(h[-2:], "q"))
    assert build_context(TOK, h, "q", 1000, max_turns=1) == TOK.encode(tekst(h[-1:], "q"))
    assert build_context(TOK, h, "q", 1000, max_turns=10) == TOK.encode(tekst(h, "q"))
    domyslne = build_context(TOK, h, "q", 1000)
    assert domyslne == TOK.encode(tekst(h[-4:], "q"))
    assert domyslne != TOK.encode(tekst(h, "q"))


def test_build_context_gubi_najstarsze_granica_budzetu():
    h = [("a1", "b1"), ("a2", "b2"), ("a3", "b3")]
    assert len(TOK.encode(tekst(h, "q"))) == 43
    assert len(TOK.encode(tekst(h[-2:], "q"))) == 31
    assert len(TOK.encode(tekst(h[-1:], "q"))) == 19
    assert len(TOK.encode(tekst([], "q"))) == 7
    assert build_context(TOK, h, "q", 43) == TOK.encode(tekst(h, "q"))
    assert build_context(TOK, h, "q", 42) == TOK.encode(tekst(h[-2:], "q"))
    assert build_context(TOK, h, "q", 31) == TOK.encode(tekst(h[-2:], "q"))
    assert build_context(TOK, h, "q", 30) == TOK.encode(tekst(h[-1:], "q"))
    assert build_context(TOK, h, "q", 19) == TOK.encode(tekst(h[-1:], "q"))
    assert build_context(TOK, h, "q", 18) == TOK.encode("A: q\nB:")
    assert build_context(TOK, h, "q", 7) == TOK.encode("A: q\nB:")
    assert build_context(TOK, h, "q", 6) == TOK.encode("A: \nB:")
    with pytest.raises(ValueError):
        build_context(TOK, h, "q", 5)


def test_build_context_wlasnosc_dla_wielu_parametrow():
    h = [("ala", "ma"), ("kot", "ale"), ("ma", "kota"), ("a", "b"), ("tak", "nie")]
    user_text = "abcdef"
    for max_turns in (0, 1, 2, 4, 5):
        for max_tokens in range(6, 71):
            wynik = build_context(TOK, h, user_text, max_tokens, max_turns=max_turns)
            assert len(wynik) <= max_tokens
            assert wynik[-3:] == TOK.encode("\nB:")
            oczekiwany = None
            for k in range(min(max_turns, 5), -1, -1):
                historia = h[len(h) - k :] if k > 0 else []
                kandydat = TOK.encode(tekst(historia, user_text))
                if len(kandydat) <= max_tokens:
                    oczekiwany = kandydat
                    break
            if oczekiwany is None:
                for n in range(len(user_text), -1, -1):
                    sufiks = user_text[len(user_text) - n :] if n > 0 else ""
                    kandydat = TOK.encode("A: " + sufiks + "\nB:")
                    if len(kandydat) <= max_tokens:
                        oczekiwany = kandydat
                        break
            assert wynik == oczekiwany, (max_turns, max_tokens)


def test_build_context_skraca_sufiks_od_poczatku():
    user_text = "abcdefghij"
    assert build_context(TOK, [], user_text, 8) == TOK.encode("A: ij\nB:")
    assert build_context(TOK, [], user_text, 9) == TOK.encode("A: hij\nB:")
    assert build_context(TOK, [("x", "y")], user_text, 7) == TOK.encode("A: j\nB:")
    assert build_context(TOK, [], user_text, 6) == TOK.encode("A: \nB:")
    with pytest.raises(ValueError):
        build_context(TOK, [], user_text, 5)
    assert len(TOK.encode("A: abcdefghij\nB:")) == 16
    assert build_context(TOK, [], user_text, 16) == TOK.encode("A: abcdefghij\nB:")
    assert build_context(TOK, [], user_text, 15) == TOK.encode("A: bcdefghij\nB:")


def test_build_context_sufiks_liczony_w_znakach():
    user_text = "Zażółć"
    przypadki = [
        (8, "A: ć\nB:"),
        (7, "A: \nB:"),
        (10, "A: łć\nB:"),
        (16, "A: Zażółć\nB:"),
    ]
    assert len(TOK.encode(user_text)) == 10
    for max_tokens, tekst_oczekiwany in przypadki:
        wynik = build_context(TOK, [], user_text, max_tokens)
        assert wynik == TOK.encode(tekst_oczekiwany)
        assert "\ufffd" not in TOK.decode(wynik)


def test_build_context_pusty_user_text():
    assert build_context(TOK, [], "", 6) == TOK.encode("A: \nB:")
    with pytest.raises(ValueError):
        build_context(TOK, [], "", 5)
    assert build_context(TOK, [("x", "y")], "", 100) == TOK.encode("A: x\nB: y\nA: \nB:")


def test_build_context_polskie_znaki_i_emoji_w_historii():
    h = [("cześć 🙂", "Zażółć gęślą jaźń"), ("co słychać?", "wszystko dobrze 😀")]
    user_text = "a u Ciebie? 🙂"
    oczekiwany = tekst(h, user_text)

    wynik = build_context(TOK, h, user_text, 1000)
    assert wynik == TOK.encode(oczekiwany)
    assert TOK.decode(wynik) == oczekiwany

    wynik_bpe = build_context(TOK_BPE, h, user_text, 1000)
    assert wynik_bpe == TOK_BPE.encode(oczekiwany)
    assert TOK_BPE.decode(wynik_bpe) == oczekiwany


def test_build_context_bpe_kodowanie_calosci():
    h = [("ala ma kota", "kot ma ale"), ("kot", "ala")]
    wynik = build_context(TOK_BPE, h, "ala ma", 1000)
    assert wynik == TOK_BPE.encode("A: ala ma kota\nB: kot ma ale\nA: kot\nB: ala\nA: ala ma\nB:")

    dolna = len(TOK_BPE.encode("A: ala ma\nB:"))
    for max_tokens in range(dolna, 61):
        wynik = build_context(TOK_BPE, h, "ala ma", max_tokens)
        assert len(wynik) <= max_tokens
        oczekiwany = None
        for k in (2, 1, 0):
            historia = h[len(h) - k :] if k > 0 else []
            kandydat = TOK_BPE.encode(tekst(historia, "ala ma"))
            if len(kandydat) <= max_tokens:
                oczekiwany = kandydat
                break
        assert wynik == oczekiwany, max_tokens


def test_build_context_char_tokenizer():
    h = [("x", "y")]
    assert build_context(TOK_CHAR, h, "ab", 18) == TOK_CHAR.encode("A: x\nB: y\nA: ab\nB:")
    assert len(TOK_CHAR.encode("A: x\nB: y\nA: ab\nB:")) == 18
    assert build_context(TOK_CHAR, h, "ab", 17) == TOK_CHAR.encode("A: ab\nB:")
    assert build_context(TOK_CHAR, h, "ab", 7) == TOK_CHAR.encode("A: b\nB:")


def test_build_context_nowa_linia_w_tekscie():
    with pytest.raises(ValueError):
        build_context(TOK, [], "a\nb", 100)
    with pytest.raises(ValueError):
        build_context(TOK, [("x\ny", "z")], "ab", 100)
    with pytest.raises(ValueError):
        build_context(TOK, [("x", "y\nz")], "ab", 100)
    with pytest.raises(ValueError):
        build_context(TOK, [("x\ny", "z"), ("p", "q")], "ab", 100, max_turns=1)
    with pytest.raises(ValueError):
        build_context(TOK, [], "\n", 100)


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_build_context_zle_argumenty(max_tokens):
    with pytest.raises(ValueError):
        build_context(TOK, [], "ab", max_tokens)
    with pytest.raises(ValueError):
        build_context(TOK, [], "ab", 100, max_turns=-1)


def test_build_context_nie_zmienia_wejscia():
    h = [("a1", "b1"), ("a2", "b2"), ("a3", "b3")]
    kopia = copy.deepcopy(h)
    user = "abcdef"
    for max_tokens in (1000, 20, 8):
        build_context(TOK, h, user, max_tokens)
        assert h == kopia
        assert len(h) == 3
        assert user == "abcdef"

    wynik = build_context(TOK, h, user, 1000)
    wynik.append(999)
    assert build_context(TOK, h, user, 1000) == TOK.encode(tekst(h, user))


# ---------------------------------------------------------------- extract_reply


def test_extract_reply_przyklady_z_opisu():
    assert extract_reply(" Dobrze, dzięki.\nA: a ty?") == "Dobrze, dzięki."
    assert extract_reply("  ok  ") == "ok"
    assert extract_reply("\nA: hej") == ""
    assert extract_reply(" kot\ufffd") == "kot"
    assert extract_reply("") == ""


def test_extract_reply_przypadki_brzegowe():
    przypadki = [
        ("a\nb\nc", "a"),
        ("   \t ", ""),
        ("\n", ""),
        ("  \n ok", ""),
        (" kot\ufffd\ufffd\ufffd ", "kot"),
        (" kot\ufffd ", "kot"),
        (" \ufffd", ""),
        ("ko\ufffdt", "ko\ufffdt"),
        ("ok\r\nA: hej", "ok"),
        (" Zażółć 🙂\nA: x", "Zażółć 🙂"),
    ]
    for wejscie, oczekiwany in przypadki:
        wynik = extract_reply(wejscie)
        assert wynik == oczekiwany, repr(wejscie)
        assert isinstance(wynik, str)
        assert "\n" not in wynik


# ---------------------------------------------------------------- chat_reply


def _odpowiedz(model, **kwargs):
    ustawienia = dict(device="cpu", block_size=32, max_new_tokens=8, temperature=0)
    ustawienia.update(kwargs)
    historia = ustawienia.pop("history", [])
    user = ustawienia.pop("user_text", "hi")
    return chat_reply(model, TOK, historia, user, **ustawienia)


def test_chat_reply_atrapa_ok_nic_po_nowej_linii():
    wynik = _odpowiedz(_atrapa(TABELA_OK))
    assert wynik == "ok"
    assert isinstance(wynik, str)
    assert _odpowiedz(_atrapa(TABELA_OK), max_new_tokens=24) == "ok"
    assert _odpowiedz(_atrapa(TABELA_OK), max_new_tokens=1) == ""
    assert _odpowiedz(_atrapa(TABELA_OK), max_new_tokens=2) == "o"
    assert _odpowiedz(_atrapa(TABELA_OK), max_new_tokens=3) == "ok"


def test_chat_reply_atrapa_pusta_odpowiedz():
    assert _odpowiedz(_atrapa(TABELA_PUSTA)) == ""


@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
def test_chat_reply_device_napis_i_obiekt(device):
    wynik = chat_reply(
        _atrapa(TABELA_OK),
        TOK,
        [],
        "hi",
        device=device,
        block_size=32,
        max_new_tokens=8,
        temperature=0,
    )
    assert wynik == "ok"


def test_chat_reply_pierwsze_wejscie_modelu():
    h = [("a1", "b1"), ("a2", "b2")]
    przypadki = [
        ({}, "A: a1\nB: b1\nA: a2\nB: b2\nA: q\nB:"),
        ({"max_turns": 1}, "A: a2\nB: b2\nA: q\nB:"),
        ({"max_turns": 0}, "A: q\nB:"),
    ]
    for dodatkowe, tekst_oczekiwany in przypadki:
        model = _atrapa(TABELA_OK)
        wynik = chat_reply(
            model,
            TOK,
            h,
            "q",
            device="cpu",
            block_size=64,
            max_new_tokens=8,
            temperature=0,
            **dodatkowe,
        )
        pierwsze = model.wejscia[0]
        oczekiwane = torch.tensor([TOK.encode(tekst_oczekiwany)])
        assert pierwsze.dtype == torch.int64
        assert pierwsze.shape == (1, len(TOK.encode(tekst_oczekiwany)))
        assert torch.equal(pierwsze, oczekiwane)
        assert wynik == "ok"


def test_chat_reply_wejscie_miesci_sie_w_budzecie():
    # Przypadek A: długa historia.
    h = [
        (f"pytanie numer {i} " + "x" * 20, f"odpowiedz numer {i} " + "y" * 20)
        for i in range(20)
    ]
    model = _atrapa(TABELA_OK)
    wynik = chat_reply(
        model,
        TOK,
        h,
        "krotkie",
        device="cpu",
        block_size=64,
        max_new_tokens=16,
        temperature=0,
        max_turns=20,
    )
    pierwsze = model.wejscia[0]
    assert pierwsze.shape[1] <= 48
    assert torch.equal(pierwsze, torch.tensor([build_context(TOK, h, "krotkie", 48, max_turns=20)]))
    assert pierwsze[0, -3:].tolist() == TOK.encode("\nB:")
    assert wynik == "ok"
    assert len(model.wejscia) == 16

    # Przypadek B: bardzo długa linia użytkownika.
    user_text = "z" * 100 + "ABCDEF"
    assert len(user_text) == 106
    model = _atrapa(TABELA_OK)
    wynik = chat_reply(
        model,
        TOK,
        [],
        user_text,
        device="cpu",
        block_size=64,
        max_new_tokens=16,
        temperature=0,
    )
    pierwsze = model.wejscia[0]
    assert pierwsze.shape == (1, 48)
    assert torch.equal(pierwsze, torch.tensor([TOK.encode("A: " + user_text[-42:] + "\nB:")]))
    assert wynik == "ok"
    assert len(model.wejscia) == 16


@pytest.mark.parametrize("max_new_tokens", [0, -1, 32, 40, 31])
def test_chat_reply_zle_argumenty(max_new_tokens):
    model = _atrapa(TABELA_OK)
    with pytest.raises(ValueError):
        chat_reply(
            model,
            TOK,
            [],
            "hi",
            device="cpu",
            block_size=32,
            max_new_tokens=max_new_tokens,
        )
    assert model.wejscia == []


@pytest.mark.parametrize(
    "parametry",
    [{"temperature": -1.0}, {"top_k": 0}, {"top_p": 0.0}, {"top_p": 1.5}],
)
def test_chat_reply_zle_probkowanie(parametry):
    with pytest.raises(ValueError):
        chat_reply(
            _atrapa(TABELA_OK),
            TOK,
            [],
            "hi",
            device="cpu",
            block_size=32,
            max_new_tokens=8,
            **parametry,
        )


def test_chat_reply_powtarzalnosc_seed():
    model = _gpt()
    h = [("ala", "ma kota"), ("kot", "ma ale")]

    def odpowiedz(**kwargs):
        return chat_reply(
            model,
            TOK,
            h,
            "co tam",
            device="cpu",
            block_size=32,
            max_new_tokens=16,
            **kwargs,
        )

    for seed in (0, 1, 2):
        a = odpowiedz(temperature=1.0, generator=_gen(seed))
        b = odpowiedz(temperature=1.0, generator=_gen(seed))
        assert isinstance(a, str)
        assert a == b

    bez_generatora = odpowiedz(temperature=0.0, generator=None)
    assert odpowiedz(temperature=0.0, generator=None) == bez_generatora
    for seed in (0, 7):
        assert odpowiedz(temperature=0.0, generator=_gen(seed)) == bez_generatora


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("przypadek", ["A", "B"])
def test_chat_reply_cache_rowny_bez_cache(seed, przypadek):
    model = _gpt()
    if przypadek == "A":
        h = [("ala", "ma kota")]
        user_text = "hej"
        dodatkowe = {}
    else:
        h = [(f"pytanie {i}", f"odpowiedz {i}") for i in range(10)]
        user_text = "a jak tam u ciebie?"
        dodatkowe = {"max_turns": 10}

    def odpowiedz(use_cache):
        return chat_reply(
            model,
            TOK,
            h,
            user_text,
            device="cpu",
            block_size=32,
            max_new_tokens=16,
            temperature=0.8,
            top_k=10,
            top_p=0.9,
            generator=_gen(seed),
            use_cache=use_cache,
            **dodatkowe,
        )

    assert odpowiedz(True) == odpowiedz(False)


@pytest.mark.parametrize("seed", range(20))
def test_chat_reply_wynik_bez_nowej_linii(seed):
    wynik = chat_reply(
        _gpt(),
        TOK,
        [],
        "hej",
        device="cpu",
        block_size=32,
        max_new_tokens=16,
        temperature=1.0,
        generator=_gen(seed),
    )
    assert isinstance(wynik, str)
    assert "\n" not in wynik


def test_chat_reply_nie_zmienia_historii_i_trybu_modelu():
    h = [("a1", "b1"), ("a2", "b2")]
    kopia = copy.deepcopy(h)
    user = "q"

    model = _atrapa(TABELA_OK)
    assert model.training is False
    chat_reply(model, TOK, h, user, device="cpu", block_size=32, max_new_tokens=8, temperature=0)
    assert h == kopia
    assert user == "q"
    assert model.training is False

    model = _atrapa(TABELA_OK)
    model.train()
    assert model.training is True
    chat_reply(model, TOK, h, user, device="cpu", block_size=32, max_new_tokens=8, temperature=0)
    assert model.training is True
    assert h == kopia
