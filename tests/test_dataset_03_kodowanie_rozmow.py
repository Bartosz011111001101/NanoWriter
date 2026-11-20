from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from nanowriter.dataset import (
    encode_dialogs,
    encode_dialogs_file,
    encode_file,
    load_mask,
    load_tokens,
    write_tokens,
)
from nanowriter.tokenizer import BPETokenizer, CharTokenizer


class _DummyTokenizer:
    """Atrapa tokenizera: zapisuje wywołania encode i zwraca wynik z podanej funkcji."""

    def __init__(self, vocab_size: int, answer: Callable[[str], list[int]]) -> None:
        self.vocab_size = vocab_size
        self._answer = answer
        self.calls: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.calls.append(text)
        return list(self._answer(text))


def _constant(ids: list[int]) -> Callable[[str], list[int]]:
    return lambda text: ids


def _runs(mask: list[int]) -> list[tuple[int, int]]:
    """Ciągłe serie jedynek maski jako pary (początek, koniec_wyłącznie)."""
    result: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value == 1 and start is None:
            start = index
        elif value != 1 and start is not None:
            result.append((start, index))
            start = None
    if start is not None:
        result.append((start, len(mask)))
    return result


def _write(path: Path, text: str) -> Path:
    path.write_bytes(text.encode("utf-8"))
    return path


def _byte_tokenizer() -> BPETokenizer:
    return BPETokenizer([])


# ---------------------------------------------------------------------------
# encode_dialogs
# ---------------------------------------------------------------------------


def test_encode_dialogs_example_a_then_b():
    ids, mask = encode_dialogs(_byte_tokenizer(), "A: hej\nB: ok\n")

    assert ids == [65, 58, 32, 104, 101, 106, 10, 66, 58, 32, 111, 107, 10]
    assert mask == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]


def test_encode_dialogs_example_b_blank_a():
    ids, mask = encode_dialogs(_byte_tokenizer(), "B: a\n\nA: b\n")

    assert ids == [66, 58, 32, 97, 10, 10, 65, 58, 32, 98, 10]
    assert mask == [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]


def test_encode_dialogs_no_trailing_newline():
    tokenizer = _byte_tokenizer()

    ids, mask = encode_dialogs(tokenizer, "B: ok")
    assert (ids, mask) == encode_dialogs(tokenizer, "B: ok\n")
    assert ids == [66, 58, 32, 111, 107, 10]
    assert mask == [0, 0, 1, 1, 1, 1]

    assert encode_dialogs(tokenizer, "A: hej\nB: ok") == encode_dialogs(
        tokenizer, "A: hej\nB: ok\n"
    )


def test_encode_dialogs_empty_text():
    assert encode_dialogs(_byte_tokenizer(), "") == ([], [])


def test_encode_dialogs_only_newline():
    ids, mask = encode_dialogs(_byte_tokenizer(), "\n")

    assert ids == [10]
    assert mask == [0]


def test_encode_dialogs_many_blank_lines():
    ids, mask = encode_dialogs(_byte_tokenizer(), "A: x\n\n\n\nB: y\n")

    assert ids == [65, 58, 32, 120, 10, 10, 10, 10, 66, 58, 32, 121, 10]
    assert mask == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]


def test_encode_dialogs_trailing_blank_line_kept():
    ids, mask = encode_dialogs(_byte_tokenizer(), "A: x\n\n")

    assert ids == [65, 58, 32, 120, 10, 10]
    assert mask == [0, 0, 0, 0, 0, 0]


def test_encode_dialogs_empty_utterances():
    tokenizer = _byte_tokenizer()

    assert encode_dialogs(tokenizer, "B: \n") == ([66, 58, 32, 10], [0, 0, 1, 1])
    assert encode_dialogs(tokenizer, "A: \n") == ([65, 58, 32, 10], [0, 0, 0, 0])


def test_encode_dialogs_consecutive_b_lines():
    ids, mask = encode_dialogs(_byte_tokenizer(), "B: a\nB: b\n")

    assert ids == [66, 58, 32, 97, 10, 66, 58, 32, 98, 10]
    assert mask == [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
    assert _runs(mask) == [(2, 5), (7, 10)]


def test_encode_dialogs_char_tokenizer_example():
    tokenizer = CharTokenizer.from_text("B: ok\n")

    ids, mask = encode_dialogs(tokenizer, "B: ok\n")

    assert ids == [3, 2, 1, 5, 4, 0]
    assert mask == [0, 0, 1, 1, 1, 1]


def test_encode_dialogs_char_tokenizer_polish():
    text = "A: Co słychać?\nB: Zażółć gęślą jaźń.\n\nA: ok\nB: dobrze\n"
    tokenizer = CharTokenizer.from_text(text)

    ids, mask = encode_dialogs(tokenizer, text)

    assert ids == tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert len(mask) == len(ids)
    runs = _runs(mask)
    assert [tokenizer.decode(ids[a:b]) for a, b in runs] == [
        " Zażółć gęślą jaźń.\n",
        " dobrze\n",
    ]
    label = tokenizer.encode("B:")
    for start, _ in runs:
        assert ids[start - len(label) : start] == label


def test_encode_dialogs_trained_bpe_runs():
    text = "A: cześć\nB: dzień dobry.\n\n" * 50
    tokenizer = BPETokenizer.train(text, 300)

    ids, mask = encode_dialogs(tokenizer, text)

    assert ids == tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert len(mask) == len(ids)
    assert all(value in (0, 1) for value in mask)
    runs = _runs(mask)
    assert len(runs) == 50
    for start, end in runs:
        assert tokenizer.decode(ids[start:end]) == " dzień dobry.\n"
        assert ids[end - 1] == 10
        assert ids[start - 2 : start] == [66, 58]
    assert sum(mask) == sum(end - start for start, end in runs)


def test_encode_dialogs_polish_emoji_runs():
    text = (
        "A: Co słychać?\nB: Zażółć gęślą jaźń 😀.\n\nA: ok\nB: Super! 👍\n"
        "B: Jeszcze jedno...\n"
    )
    tokenizers = [BPETokenizer([]), BPETokenizer.train(text * 20, 320)]

    for tokenizer in tokenizers:
        ids, mask = encode_dialogs(tokenizer, text)

        assert ids == tokenizer.encode(text)
        assert tokenizer.decode(ids) == text
        assert [tokenizer.decode(ids[a:b]) for a, b in _runs(mask)] == [
            " Zażółć gęślą jaźń 😀.\n",
            " Super! 👍\n",
            " Jeszcze jedno...\n",
        ]


def test_encode_dialogs_newline_is_separate_token():
    text = "A: hej.\nB: ok.\n" * 50
    tokenizer = BPETokenizer.train(text, 300)
    ids, _ = encode_dialogs(tokenizer, text)

    assert ids.count(10) == 100
    for token in ids:
        assert ("\n" in tokenizer.decode([token])) == (token == 10)

    text2 = "B: Koniec.\nB: Naprawdę?\nB: Tak!\nB: ...\n"
    tokenizer2 = BPETokenizer.train(text2 * 30, 300)
    ids2, _ = encode_dialogs(tokenizer2, text2)

    for token in ids2:
        assert ("\n" in tokenizer2.decode([token])) == (token == 10)


def test_encode_dialogs_a_and_blank_all_zero():
    ids, mask = encode_dialogs(_byte_tokenizer(), "A: hej\n\nA: co tam\n\n\nA: nic\n")

    assert len(ids) > 0
    assert len(mask) == len(ids)
    assert all(value == 0 for value in mask)


def test_encode_dialogs_encodes_each_line_separately():
    def answer(text: str) -> list[int]:
        if text == "B:":
            return [1, 2]
        if text.startswith("B:") and text.endswith("\n"):
            return [1, 2, 5]
        return [7]

    tokenizer = _DummyTokenizer(300, answer)
    source = "A: x\nB: y\n\n"

    ids, mask = encode_dialogs(tokenizer, source)

    assert ids == [7, 1, 2, 5, 7]
    assert mask == [0, 0, 0, 1, 0]
    assert "A: x\n" in tokenizer.calls
    assert "B: y\n" in tokenizer.calls
    assert "\n" in tokenizer.calls
    for call in tokenizer.calls:
        assert call != source
        if call != "B:":
            assert call.endswith("\n")
            assert call.count("\n") == 1


def test_encode_dialogs_label_merged_raises():
    def answer(text: str) -> list[int]:
        return [1, 2] if text == "B:" else [9, 8, 7]

    tokenizer = _DummyTokenizer(300, answer)

    with pytest.raises(ValueError):
        encode_dialogs(tokenizer, "B: ok\n")


def test_encode_dialogs_label_ok_with_dummy():
    def answer(text: str) -> list[int]:
        if text == "B:":
            return [1, 2]
        if text == "B: ok\n":
            return [1, 2, 5, 6]
        raise AssertionError(f"nieoczekiwane wywołanie encode: {text!r}")

    tokenizer = _DummyTokenizer(300, answer)

    ids, mask = encode_dialogs(tokenizer, "B: ok\n")

    assert ids == [1, 2, 5, 6]
    assert mask == [0, 0, 1, 1]


@pytest.mark.parametrize(
    "text",
    [
        "C: x\n",
        "B:ok\n",
        "A:\n",
        "B:\n",
        "A:x\n",
        "a: x\n",
        " A: x\n",
        "tekst bez prefiksu\n",
        "A: dobrze\nB: ok\nxyz\n",
    ],
)
def test_encode_dialogs_invalid_format_raises(text):
    with pytest.raises(ValueError):
        encode_dialogs(_byte_tokenizer(), text)


@pytest.mark.parametrize(
    ("text", "number"),
    [
        ("C: x\n", "1"),
        ("A: a\nxyz\n", "2"),
        ("A: a\nB: b\nqqq\n", "3"),
        ("A: a\nB: b\nA: c\n\nzzz\n", "5"),
    ],
)
def test_encode_dialogs_error_contains_line_number(text, number):
    with pytest.raises(ValueError) as error:
        encode_dialogs(_byte_tokenizer(), text)

    assert number in str(error.value)


def test_encode_dialogs_error_truncates_long_line():
    text = "A: a\n" + "C" + "x" * 100 + "\n"

    with pytest.raises(ValueError) as error:
        encode_dialogs(_byte_tokenizer(), text)

    message = str(error.value)
    assert "2" in message
    assert "x" * 39 in message
    assert "x" * 41 not in message


def test_encode_dialogs_return_types():
    result = encode_dialogs(_byte_tokenizer(), "A: hej\nB: ok\n")

    assert isinstance(result, tuple)
    assert len(result) == 2
    ids, mask = result
    assert isinstance(ids, list)
    assert isinstance(mask, list)
    assert len(ids) == len(mask)
    assert all(type(value) is int for value in ids)
    assert all(type(value) is int for value in mask)


def test_encode_dialogs_prints_nothing(capsys):
    encode_dialogs(_byte_tokenizer(), "A: hej\nB: ok\n")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# ---------------------------------------------------------------------------
# encode_dialogs_file
# ---------------------------------------------------------------------------


def test_encode_dialogs_file_example(tmp_path):
    source = _write(tmp_path / "in.txt", "B: ok\n")
    out_path = tmp_path / "t.bin"
    mask_path = tmp_path / "m.bin"

    result = encode_dialogs_file(_byte_tokenizer(), source, out_path, mask_path)

    assert result == (6, 6)
    assert out_path.stat().st_size == 12
    assert mask_path.stat().st_size == 6
    assert load_tokens(out_path).tolist() == [66, 58, 32, 111, 107, 10]
    mask = load_mask(mask_path)
    assert mask.tolist() == [False, False, True, True, True, True]
    assert mask.dtype == torch.bool


def test_encode_dialogs_file_matches_encode_dialogs_polish(tmp_path):
    text = "A: Zażółć gęślą\nB: Gęś 😀.\n\nA: ok\nB: tak\n"
    source = _write(tmp_path / "in.txt", text)
    out_path = tmp_path / "t.bin"
    mask_path = tmp_path / "m.bin"
    tokenizer = _byte_tokenizer()
    ids, mask = encode_dialogs(tokenizer, text)

    result = encode_dialogs_file(tokenizer, source, out_path, mask_path)

    assert result == (len(ids), len(text))
    assert load_tokens(out_path).tolist() == ids
    assert load_mask(mask_path).tolist() == [bool(value) for value in mask]
    assert out_path.stat().st_size == 2 * len(ids)
    assert mask_path.stat().st_size == len(ids)
    assert len(load_mask(mask_path)) == len(load_tokens(out_path))
    assert set(mask_path.read_bytes()) <= {0, 1}


def test_encode_dialogs_file_crlf(tmp_path):
    crlf = tmp_path / "crlf.txt"
    crlf.write_bytes(b"A: a\r\nB: b\r\n")
    lf = tmp_path / "lf.txt"
    lf.write_bytes(b"A: a\nB: b\n")
    tokenizer = _byte_tokenizer()

    result_crlf = encode_dialogs_file(tokenizer, crlf, tmp_path / "t1.bin", tmp_path / "m1.bin")
    result_lf = encode_dialogs_file(tokenizer, lf, tmp_path / "t2.bin", tmp_path / "m2.bin")

    assert result_crlf == (10, 10)
    assert result_lf == (10, 10)
    tokens_crlf = load_tokens(tmp_path / "t1.bin").tolist()
    assert tokens_crlf == load_tokens(tmp_path / "t2.bin").tolist()
    assert load_mask(tmp_path / "m1.bin").tolist() == load_mask(tmp_path / "m2.bin").tolist()
    assert tokens_crlf == [65, 58, 32, 97, 10, 66, 58, 32, 98, 10]
    assert [int(v) for v in load_mask(tmp_path / "m1.bin")] == [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    assert 13 not in tokens_crlf


def test_encode_dialogs_file_no_trailing_newline(tmp_path):
    source = tmp_path / "in.txt"
    source.write_bytes(b"B: ok")
    out_path = tmp_path / "t.bin"
    mask_path = tmp_path / "m.bin"

    encode_dialogs_file(_byte_tokenizer(), source, out_path, mask_path)

    assert load_tokens(out_path).tolist() == [66, 58, 32, 111, 107, 10]
    assert load_mask(mask_path).tolist() == [False, False, True, True, True, True]


def test_encode_dialogs_file_empty(tmp_path):
    source = tmp_path / "in.txt"
    source.write_bytes(b"")
    out_path = tmp_path / "t.bin"
    mask_path = tmp_path / "m.bin"

    result = encode_dialogs_file(_byte_tokenizer(), source, out_path, mask_path)

    assert result == (0, 0)
    assert out_path.exists()
    assert mask_path.exists()
    assert out_path.stat().st_size == 0
    assert mask_path.stat().st_size == 0


def test_encode_dialogs_file_creates_parent_dirs(tmp_path):
    source = _write(tmp_path / "in.txt", "B: ok\n")
    out_path = tmp_path / "a" / "b" / "t.bin"
    mask_path = tmp_path / "c" / "m.bin"

    encode_dialogs_file(_byte_tokenizer(), source, out_path, mask_path)

    assert out_path.exists()
    assert mask_path.exists()


def test_encode_dialogs_file_vocab_too_big(tmp_path):
    tokenizer = _DummyTokenizer(70000, _constant([1]))
    source = _write(tmp_path / "in.txt", "B: ok\n")
    out_path = tmp_path / "t.bin"
    mask_path = tmp_path / "m.bin"

    with pytest.raises(ValueError):
        encode_dialogs_file(tokenizer, source, out_path, mask_path)
    assert not out_path.exists()
    assert not mask_path.exists()

    with pytest.raises(ValueError):
        encode_dialogs_file(tokenizer, tmp_path / "brak.txt", out_path, mask_path)
    assert not out_path.exists()
    assert not mask_path.exists()


def test_encode_dialogs_file_vocab_limit_ok(tmp_path):
    def answer(text: str) -> list[int]:
        return [1, 2] if text == "B:" else [1, 2, 3]

    tokenizer = _DummyTokenizer(65536, answer)
    source = _write(tmp_path / "in.txt", "B: x\n")
    out_path = tmp_path / "t.bin"
    mask_path = tmp_path / "m.bin"

    result = encode_dialogs_file(tokenizer, source, out_path, mask_path)

    assert result == (3, 5)
    assert load_tokens(out_path).tolist() == [1, 2, 3]
    assert load_mask(mask_path).tolist() == [False, False, True]


def test_encode_dialogs_file_bad_line(tmp_path):
    source = _write(tmp_path / "in.txt", "A: a\nC: x\n")
    out_path = tmp_path / "t.bin"
    mask_path = tmp_path / "m.bin"

    with pytest.raises(ValueError):
        encode_dialogs_file(_byte_tokenizer(), source, out_path, mask_path)

    assert not out_path.exists()
    assert not mask_path.exists()


# ---------------------------------------------------------------------------
# encode_file z block_chars
# ---------------------------------------------------------------------------


def test_encode_file_block_example(tmp_path):
    source = _write(tmp_path / "in.txt", "ab\ncd\n")
    out_path = tmp_path / "t.bin"
    tokenizer = CharTokenizer.from_text("abcd\n")

    result = encode_file(tokenizer, source, out_path, block_chars=3)

    assert result == (6, 6)
    assert load_tokens(out_path).tolist() == [1, 2, 0, 3, 4, 0]


def test_encode_file_block_equals_whole_char_tokenizer(tmp_path):
    text = "".join(f"linia {i} zażółć gęślą jaźń\n" for i in range(200))
    source = _write(tmp_path / "in.txt", text)
    tokenizer = CharTokenizer.from_text(text)
    whole_path = tmp_path / "whole.bin"
    whole_result = encode_file(tokenizer, source, whole_path)

    for block_chars in (1, 7, 100, 1000, 10**9):
        out_path = tmp_path / f"block_{block_chars}.bin"

        result = encode_file(tokenizer, source, out_path, block_chars=block_chars)

        assert out_path.read_bytes() == whole_path.read_bytes()
        assert result == whole_result
        assert load_tokens(out_path).tolist() == tokenizer.encode(text)


def test_encode_file_block_equals_whole_bpe(tmp_path):
    lines = []
    for i in range(400):
        if i % 2 == 0:
            lines.append(f"Zażółć gęślą jaźń {i} 😀 kot ma psa.\n")
        else:
            lines.append(f"A: Co słychać u {i}?\nB: Dobrze, dziękuję 👍...\n")
    text = "".join(lines)
    source = _write(tmp_path / "in.txt", text)
    tokenizer = BPETokenizer.train(text, 400)
    whole_path = tmp_path / "whole.bin"
    whole_result = encode_file(tokenizer, source, whole_path)

    for block_chars in (1, 7, 100, 5000, 10**9):
        out_path = tmp_path / f"block_{block_chars}.bin"

        result = encode_file(tokenizer, source, out_path, block_chars=block_chars)

        assert out_path.read_bytes() == whole_path.read_bytes()
        assert result == whole_result
        assert load_tokens(out_path).tolist() == tokenizer.encode(text)
        assert result[1] == len(text)


def test_encode_file_block_no_trailing_newline(tmp_path):
    source = _write(tmp_path / "in.txt", "ab\ncd")
    tokenizer = CharTokenizer.from_text("abcd\n")

    for block_chars in (1, 3, 100):
        out_path = tmp_path / f"t_{block_chars}.bin"

        result = encode_file(tokenizer, source, out_path, block_chars=block_chars)

        assert result == (5, 5)
        assert load_tokens(out_path).tolist() == [1, 2, 0, 3, 4]


def test_encode_file_block_empty_file(tmp_path):
    source = tmp_path / "in.txt"
    source.write_bytes(b"")
    out_path = tmp_path / "t.bin"

    result = encode_file(CharTokenizer.from_text("ab\n"), source, out_path, block_chars=10)

    assert result == (0, 0)
    assert out_path.exists()
    assert out_path.stat().st_size == 0


def test_encode_file_block_empty_file_overwrites(tmp_path):
    source = tmp_path / "in.txt"
    source.write_bytes(b"")
    out_path = tmp_path / "t.bin"
    out_path.write_bytes(b"xxxxxxxx")

    encode_file(CharTokenizer.from_text("ab\n"), source, out_path, block_chars=10)

    assert out_path.stat().st_size == 0


def test_encode_file_block_overwrites_existing(tmp_path):
    source = _write(tmp_path / "in.txt", "ab\ncd\n")
    out_path = tmp_path / "t.bin"
    out_path.write_bytes(b"xxxxxxxxxxxx")

    encode_file(CharTokenizer.from_text("abcd\n"), source, out_path, block_chars=3)

    assert out_path.read_bytes() == b"\x01\x00\x02\x00\x00\x00\x03\x00\x04\x00\x00\x00"


def test_encode_file_block_creates_parent_dirs(tmp_path):
    tokenizer = CharTokenizer.from_text("ab\n")
    source = _write(tmp_path / "in.txt", "ab\n")
    out_path = tmp_path / "x" / "y" / "o.bin"

    encode_file(tokenizer, source, out_path, block_chars=2)

    assert out_path.exists()

    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    out_path2 = tmp_path / "x2" / "y2" / "o.bin"

    encode_file(tokenizer, empty, out_path2, block_chars=2)

    assert out_path2.exists()


@pytest.mark.parametrize("block_chars", [0, -1])
def test_encode_file_block_chars_invalid(tmp_path, block_chars):
    source = _write(tmp_path / "in.txt", "ab\n")
    out_path = tmp_path / "t.bin"

    with pytest.raises(ValueError):
        encode_file(CharTokenizer.from_text("ab\n"), source, out_path, block_chars=block_chars)

    assert not out_path.exists()


def test_encode_file_vocab_too_big_with_block(tmp_path):
    tokenizer = _DummyTokenizer(70000, _constant([1]))
    source = _write(tmp_path / "in.txt", "ab\n")
    out_path = tmp_path / "t.bin"

    with pytest.raises(ValueError):
        encode_file(tokenizer, source, out_path, block_chars=3)
    assert not out_path.exists()

    with pytest.raises(ValueError):
        encode_file(tokenizer, tmp_path / "brak.txt", out_path, block_chars=3)
    assert not out_path.exists()

    with pytest.raises(ValueError):
        encode_file(tokenizer, tmp_path / "brak.txt", out_path)
    assert not out_path.exists()


def test_encode_file_block_crlf(tmp_path):
    source = tmp_path / "in.txt"
    source.write_bytes(b"ab\r\ncd\r\n")
    out_path = tmp_path / "t.bin"

    result = encode_file(CharTokenizer.from_text("abcd\n"), source, out_path, block_chars=3)

    assert result == (6, 6)
    assert load_tokens(out_path).tolist() == [1, 2, 0, 3, 4, 0]

    dummy = _DummyTokenizer(300, _constant([5, 6]))
    encode_file(dummy, source, tmp_path / "t2.bin", block_chars=1)

    assert dummy.calls == ["ab\n", "cd\n"]


def test_encode_file_block_call_count(tmp_path):
    source = _write(tmp_path / "in.txt", "ab\ncd\nef\n")
    expected = {
        1: (["ab\n", "cd\n", "ef\n"], (6, 9)),
        3: (["ab\n", "cd\n", "ef\n"], None),
        4: (["ab\ncd\n", "ef\n"], (4, 9)),
        6: (["ab\ncd\n", "ef\n"], None),
        7: (["ab\ncd\nef\n"], (2, 9)),
        10**9: (["ab\ncd\nef\n"], None),
    }

    for block_chars, (calls, result) in expected.items():
        dummy = _DummyTokenizer(300, _constant([5, 6]))
        out_path = tmp_path / f"t_{block_chars}.bin"

        actual = encode_file(dummy, source, out_path, block_chars=block_chars)

        assert dummy.calls == calls
        if result is not None:
            assert actual == result
        if block_chars == 1:
            assert load_tokens(out_path).tolist() == [5, 6, 5, 6, 5, 6]


def test_encode_file_block_last_line_without_newline_calls(tmp_path):
    source = _write(tmp_path / "in.txt", "ab\ncd")
    dummy = _DummyTokenizer(300, _constant([5]))

    result = encode_file(dummy, source, tmp_path / "t.bin", block_chars=1)

    assert dummy.calls == ["ab\n", "cd"]
    assert result == (2, 5)


def test_encode_file_without_block_chars_single_call(tmp_path):
    source = _write(tmp_path / "in.txt", "ab\ncd\nef\n")

    dummy = _DummyTokenizer(300, _constant([5, 6, 7]))
    out_path = tmp_path / "t.bin"
    result = encode_file(dummy, source, out_path)

    assert dummy.calls == ["ab\ncd\nef\n"]
    assert result == (3, 9)
    assert load_tokens(out_path).tolist() == [5, 6, 7]

    dummy_none = _DummyTokenizer(300, _constant([5, 6, 7]))
    out_path_none = tmp_path / "t_none.bin"
    result_none = encode_file(dummy_none, source, out_path_none, block_chars=None)

    assert dummy_none.calls == ["ab\ncd\nef\n"]
    assert result_none == (3, 9)
    assert load_tokens(out_path_none).tolist() == [5, 6, 7]


def test_encode_file_block_prints_nothing(tmp_path, capsys):
    source = _write(tmp_path / "in.txt", "ab\ncd\n")

    encode_file(CharTokenizer.from_text("abcd\n"), source, tmp_path / "t.bin", block_chars=3)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# ---------------------------------------------------------------------------
# write_tokens (regresja po refaktoryzacji)
# ---------------------------------------------------------------------------


def test_write_tokens_unchanged(tmp_path):
    p = tmp_path / "p.bin"
    assert write_tokens([1, 2, 1023], p) == 3
    assert p.read_bytes() == b"\x01\x00\x02\x00\xff\x03"

    p2 = tmp_path / "p2.bin"
    assert write_tokens([], p2) == 0
    assert p2.read_bytes() == b""

    p3 = tmp_path / "p3.bin"
    assert write_tokens([65535], p3) == 1
    assert p3.read_bytes() == b"\xff\xff"


def test_write_tokens_out_of_range_no_file(tmp_path):
    p = tmp_path / "p.bin"
    with pytest.raises(ValueError):
        write_tokens([65536], p)
    assert not p.exists()

    with pytest.raises(ValueError):
        write_tokens([1, -1], p)
    assert not p.exists()

    nested = tmp_path / "n" / "p.bin"
    with pytest.raises(ValueError):
        write_tokens([5, 65536], nested)
    assert not nested.exists()
