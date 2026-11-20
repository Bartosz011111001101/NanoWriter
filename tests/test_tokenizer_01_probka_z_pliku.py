"""Testy sample_file i sample_lines (tokenizer, prompt 01 zadania 08)."""

import pytest

from nanowriter.tokenizer import sample_file, sample_lines


def _write(tmp_path, text, name="plik.txt"):
    path = tmp_path / name
    path.write_bytes(text.encode("utf-8"))
    return path


def test_sample_file_example_every_second_line(tmp_path):
    path = _write(tmp_path, "a\nb\nc\nd\n")
    assert sample_file(path, 4) == "a\nc\n"


def test_sample_file_short_file_returned_whole(tmp_path):
    path = _write(tmp_path, "abc")
    assert sample_file(path, 10) == "abc"


def test_sample_file_empty_file(tmp_path):
    path = _write(tmp_path, "")
    result = sample_file(path, 5)
    assert isinstance(result, str)
    assert result == ""


def test_sample_file_crlf_normalized(tmp_path):
    path = _write(tmp_path, "a\r\nb\r\n")
    result = sample_file(path, 3)
    assert result == "a\n"
    assert "\r" not in result


def test_sample_file_crlf_short_enough(tmp_path):
    path = _write(tmp_path, "a\r\nb\r\n")
    assert sample_file(path, 4) == "a\nb\n"


def test_sample_file_exactly_max_chars(tmp_path):
    path = _write(tmp_path, "a\nb\nc\nd\n")
    assert sample_file(path, 8) == "a\nb\nc\nd\n"


def test_sample_file_one_char_over_limit(tmp_path):
    path = _write(tmp_path, "a\nb\nc\nd\n")
    assert sample_file(path, 7) == "a\nc\n"


def test_sample_file_k_equals_3(tmp_path):
    path = _write(tmp_path, "a\nb\nc\nd\ne\nf\ng\n")
    assert sample_file(path, 5) == "a\nd\ng\n"


def test_sample_file_no_trailing_newline_k3(tmp_path):
    path = _write(tmp_path, "a\nb\nc")
    assert sample_file(path, 2) == "a\n"


def test_sample_file_no_trailing_newline_k5(tmp_path):
    path = _write(tmp_path, "a\nb\nc")
    assert sample_file(path, 1) == "a\n"


def test_sample_file_last_selected_line_without_newline(tmp_path):
    path = _write(tmp_path, "a\nb\nc")
    assert sample_file(path, 3) == "a\nc"


def test_sample_file_polish_chars_counted_as_chars(tmp_path):
    path = _write(tmp_path, "zażółć\n")
    assert sample_file(path, 7) == "zażółć\n"


def test_sample_file_polish_chars_single_line_over_limit(tmp_path):
    path = _write(tmp_path, "zażółć\n")
    assert sample_file(path, 6) == "zażółć\n"


def test_sample_file_polish_and_emoji_lines(tmp_path):
    path = _write(tmp_path, "żółw\nkot 🐱\nzażółć gęślą jaźń\nlis\n")
    assert sample_file(path, 18) == "żółw\nzażółć gęślą jaźń\n"


def test_sample_file_invalid_max_chars(tmp_path):
    path = _write(tmp_path, "a\nb\n")
    with pytest.raises(ValueError):
        sample_file(path, 0)
    with pytest.raises(ValueError):
        sample_file(path, -1)


def test_sample_file_invalid_max_chars_missing_file(tmp_path):
    path = tmp_path / "brak.txt"
    with pytest.raises(ValueError):
        sample_file(path, 0)
    with pytest.raises(ValueError):
        sample_file(path, -1)


def test_sample_file_missing_file(tmp_path):
    path = tmp_path / "brak.txt"
    with pytest.raises(FileNotFoundError):
        sample_file(path, 10)


def test_sample_file_300_lines_exact(tmp_path):
    path = _write(tmp_path, "".join(f"linia {i}\n" for i in range(300)))
    expected = "".join(f"linia {i}\n" for i in range(0, 300, 6))
    assert sample_file(path, 500) == expected
    assert expected.count("\n") == 50


def test_sample_file_result_is_whole_lines_in_order(tmp_path):
    lines = ["x" * (3 + (i * 7) % 11) + str(i) + "\n" for i in range(500)]
    text = "".join(lines)
    path = _write(tmp_path, text)
    result = sample_file(path, 1000)
    assert result != ""
    assert len(result) < len(text)
    result_lines = result.split("\n")[:-1]
    file_lines = [line[:-1] for line in lines]
    assert set(result_lines) <= set(file_lines)
    positions = [file_lines.index(line) for line in result_lines]
    assert positions == sorted(positions)
    assert len(set(positions)) == len(positions)


def test_sample_file_size_close_to_max_chars(tmp_path):
    path = _write(tmp_path, "abcdefghi\n" * 1000)
    result = sample_file(path, 1000)
    assert result.count("\n") == 100
    assert len(result) == 1000


def test_sample_file_does_not_modify_file(tmp_path):
    raw = b"a\r\nb\r\nc\r\nd\r\n"
    path = tmp_path / "crlf.txt"
    path.write_bytes(raw)
    sample_file(path, 3)
    assert path.read_bytes() == raw


def test_sample_file_prints_nothing(tmp_path, capsys):
    path = _write(tmp_path, "a\nb\nc\nd\n")
    sample_file(path, 4)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_sample_lines_unchanged():
    assert sample_lines("a\nb\nc\nd", 4) == "a\nc"
    assert sample_lines("abc", 10) == "abc"
    assert sample_lines("", 5) == ""
    with pytest.raises(ValueError):
        sample_lines("abc", 0)
