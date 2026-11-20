import json
import unicodedata
from pathlib import Path

import pytest

from nanowriter.data import (
    ASSISTANT_PHRASES,
    DialogCorpusResult,
    DialogReport,
    build_dialog_corpus,
    format_dialog_stream,
    load_dialogs,
    split_dialog_blocks,
    split_dialogs,
    validate_dialog,
)

V1 = ["A: Hej.", "B: Cześć.", "A: Co tam?", "B: Dobrze."]
V2 = ["A: Jak się masz?", "B: Dobrze, dziękuję.", "A: Co robisz?", "B: Czytam książkę."]
V3 = ["A: Idziemy na spacer?", "B: Chętnie.", "A: Kiedy?", "B: Zaraz."]


def D(i: int) -> list[str]:
    return [f"A: Pytanie {i}.", f"B: Odpowiedz {i}.", f"A: Dalej {i}?", f"B: Koniec {i}."]


def T(i: int) -> str:
    return "\n".join(D(i)) + "\n"


def S(dialogs: list[list[str]]) -> str:
    return "".join("\n".join(lines) + "\n---\n" for lines in dialogs)


def ok4(index: int, line: str) -> list[str]:
    lines = ["A: a", "B: b", "A: c", "B: d"]
    lines[index] = line
    return lines


def write_src(directory: Path, files: dict[str, str | bytes]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        data = content.encode("utf-8") if isinstance(content, str) else content
        (directory / name).write_bytes(data)
    return directory


def read_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


# --- ASSISTANT_PHRASES ---


def test_stala_assistant_phrases():
    assert isinstance(ASSISTANT_PHRASES, tuple)
    assert len(ASSISTANT_PHRASES) > 0
    for phrase in ASSISTANT_PHRASES:
        assert isinstance(phrase, str)
        assert phrase == phrase.lower()
    assert "jako asystent" in ASSISTANT_PHRASES
    assert "oczywiście! oto" in ASSISTANT_PHRASES


# --- split_dialog_blocks ---


def test_split_blocks_przyklad():
    text = "A: Hej.\nB: Cześć.\n---\nA: Co?\nB: Nic.\n---\n"
    assert split_dialog_blocks(text) == [["A: Hej.", "B: Cześć."], ["A: Co?", "B: Nic."]]


def test_split_blocks_biale_znaki_i_crlf():
    assert split_dialog_blocks("A:  Hej.  \r\nB:\tOk.") == [["A: Hej.", "B: Ok."]]
    assert split_dialog_blocks("A:   a    b\nB: c") == [["A: a b", "B: c"]]


def test_split_blocks_samotny_cr():
    assert split_dialog_blocks("A: a\rB: b") == [["A: a", "B: b"]]


def test_split_blocks_bom():
    assert split_dialog_blocks("﻿A: a\nB: b\n---\n") == [["A: a", "B: b"]]


def test_split_blocks_nfc():
    nfd = unicodedata.normalize("NFD", "A: Café\nB: ok")
    assert nfd != "A: Café\nB: ok"
    assert split_dialog_blocks(nfd) == [["A: Café", "B: ok"]]


def test_split_blocks_puste_linie_na_brzegach():
    assert split_dialog_blocks("\n\nA: x\n---\n\n") == [["A: x"]]
    assert split_dialog_blocks("\nA: a\nB: b\n\n\n---\n") == [["A: a", "B: b"]]


def test_split_blocks_pusta_linia_w_srodku_zostaje():
    assert split_dialog_blocks("A: a\n\nB: b") == [["A: a", "", "B: b"]]
    assert split_dialog_blocks("A: a\n   \nB: b") == [["A: a", "", "B: b"]]


def test_split_blocks_puste_wejscia():
    assert split_dialog_blocks("") == []
    assert split_dialog_blocks("---\n---\n") == []
    assert split_dialog_blocks("\n\n") == []
    assert split_dialog_blocks("---") == []
    assert split_dialog_blocks("A: a\n---\n\n  \n---\nB: b\n---") == [["A: a"], ["B: b"]]


def test_split_blocks_separator_z_bialymi_znakami():
    assert split_dialog_blocks("A: a\n --- \nB: b") == [["A: a"], ["B: b"]]
    assert split_dialog_blocks("A: a\n----\nB: b") == [["A: a", "----", "B: b"]]
    assert split_dialog_blocks("A: a\n---\nB: b") == [["A: a"], ["B: b"]]


def test_split_blocks_nie_waliduje():
    result = split_dialog_blocks("C: x\nB: y\n---\nbez prefiksu\n")
    assert result == [["C: x", "B: y"], ["bez prefiksu"]]
    assert type(result) is list
    for block in result:
        assert type(block) is list
        for line in block:
            assert isinstance(line, str)


# --- validate_dialog ---


def test_validate_poprawne():
    assert validate_dialog(["A: Hej.", "B: Cześć.", "A: Co tam?", "B: Dobrze."]) is None
    assert (
        validate_dialog(
            ["A: Zażółć gęślą jaźń.", "B: Źdźbło.", "A: Mam 5 lat.", "B: „Dobrze”, (cicho)…"]
        )
        is None
    )
    assert validate_dialog(["A:  Hej.", "B: Cześć.", "A: Co tam?", "B: Dobrze."]) is None


def test_validate_pusta():
    assert validate_dialog([]) == "pusta"


def test_validate_pusta_linia():
    assert validate_dialog(["A: a", "", "A: b", "B: c"]) == "pusta-linia"
    assert validate_dialog([""]) == "pusta-linia"


def test_validate_zla_linia():
    assert validate_dialog(["A: a", "B:b", "A: c", "B: d"]) == "zla-linia"
    assert validate_dialog(["A: a", "C: b", "A: c", "B: d"]) == "zla-linia"
    assert validate_dialog(["a: a", "B: b", "A: c", "B: d"]) == "zla-linia"
    assert validate_dialog(["A:", "B: b", "A: c", "B: d"]) == "zla-linia"
    assert validate_dialog([" A: a", "B: b", "A: c", "B: d"]) == "zla-linia"


def test_validate_pusta_wypowiedz():
    assert validate_dialog(["A: a", "B: ", "A: c", "B: d"]) == "pusta-wypowiedz"
    assert validate_dialog(["A: a", "B:  ", "A: c", "B: d"]) == "pusta-wypowiedz"
    assert validate_dialog(["A: ", "B: b", "A: c", "B: d"]) == "pusta-wypowiedz"


def test_validate_znaki():
    cases = [
        (1, "B: *b*"),
        (0, "A: Co *tam*?"),
        (0, "A: Hej 😀"),
        (1, "B: a/b"),
        (1, "B: Tom & Jerry"),
        (1, "B: Mail: x@y"),
        (1, "B: Привет"),
        (1, "B: a\tb"),
        (1, "B: [b]"),
        (1, "B: _b_"),
        (1, "B: #b"),
    ]
    for index, line in cases:
        assert validate_dialog(ok4(index, line)) == "znaki", line
    assert validate_dialog(["A: a", "B: *b*", "A: c", "B: d"]) == "znaki"


def test_validate_lista():
    lines = ["B: - punkt", "B: – punkt", "B: — punkt", "B: 2) coś", "B: 1. Idę.", "B: 12. Idę."]
    for line in lines:
        assert validate_dialog(ok4(1, line)) == "lista", line
    assert validate_dialog(["A: 3. coś", "B: b", "A: c", "B: d"]) == "lista"


def test_validate_nie_lista():
    for line in [
        "B: Mam 2 koty.",
        "B: -5 stopni jest zimno.",
        "B: 5.5 kg",
        "B: 2020 był dobry.",
        "B: Idę - ale wolno.",
    ]:
        assert validate_dialog(ok4(1, line)) is None, line


def test_validate_asystent():
    assert validate_dialog(["A: a", "B: Oczywiście! Oto plan.", "A: c", "B: d"]) == "asystent"
    assert validate_dialog(["A: a", "B: JAKO ASYSTENT wiem.", "A: c", "B: d"]) == "asystent"
    assert validate_dialog(["A: a", "B: Wiem, w czym mogę pomóc.", "A: c", "B: d"]) == "asystent"


@pytest.mark.parametrize("phrase", ASSISTANT_PHRASES)
def test_validate_kazda_fraza_asystenta(phrase):
    assert validate_dialog(["A: a", f"B: Dobrze, {phrase} tutaj.", "A: c", "B: d"]) == "asystent"


def test_validate_fraza_w_linii_a_nie_odrzuca():
    assert validate_dialog(["A: Oczywiście! Oto plan.", "B: b", "A: c", "B: d"]) is None


def test_validate_start():
    assert validate_dialog(["B: a", "A: b", "A: c", "B: d"]) == "start"
    assert validate_dialog(["B: a", "A: b", "B: c", "A: d"]) == "start"


def test_validate_kolejnosc():
    assert validate_dialog(["A: a", "B: b", "B: c", "A: d"]) == "kolejnosc"
    assert validate_dialog(["A: a", "A: b", "B: c", "B: d"]) == "kolejnosc"


def test_validate_koniec():
    assert validate_dialog(["A: a", "B: b", "A: c", "B: d", "A: e"]) == "koniec"


def test_validate_dlugosc_granice():
    assert validate_dialog(["A: a", "B: b"]) == "dlugosc"
    assert validate_dialog(["A: a", "B: b", "A: c", "B: d"]) is None
    assert validate_dialog(["A: a", "B: b"] * 8) is None
    assert validate_dialog(["A: a", "B: b"] * 9) == "dlugosc"


def test_validate_dlugosc_wlasne_granice():
    assert validate_dialog(["A: a", "B: b"], min_lines=2) is None
    six = ["A: a", "B: b", "A: c", "B: d", "A: e", "B: f"]
    assert validate_dialog(six, max_lines=4) == "dlugosc"
    assert validate_dialog(six, max_lines=6) is None
    assert validate_dialog(["A: a", "B: b", "A: c", "B: d"], min_lines=6) == "dlugosc"


def test_validate_kolejnosc_sprawdzen():
    cases = [
        (["A: a", "", "C: x", "B: d"], "pusta-linia"),
        (["A: a", "B: ", "C: x", "B: d"], "zla-linia"),
        (["A: a", "B: ", "A: *c*", "B: d"], "pusta-wypowiedz"),
        (["A: a", "B: - *x*", "A: c", "B: d"], "znaki"),
        (["A: a", "B: - Oczywiście! Oto plan.", "A: c", "B: d"], "lista"),
        (["A: a", "B: Oczywiście! Oto *plan*", "A: c", "B: d"], "znaki"),
        (["B: Oczywiście! Oto plan.", "A: b", "A: c", "B: d"], "asystent"),
        (["B: a", "B: b"], "start"),
        (["A: a", "B: b", "B: c", "A: d"], "kolejnosc"),
        (["A: a", "B: b", "A: c", "A: d"], "kolejnosc"),
        (["A: a", "B: b"], "dlugosc"),
    ]
    for lines, expected in cases:
        assert validate_dialog(lines) == expected, lines


def test_validate_nie_zmienia_argumentu():
    lines = ["A: a", "B: b", "A: c", "B: d"]
    copy = list(lines)
    validate_dialog(lines)
    assert lines == copy

    bad = ["A:  a ", "x", "B: d"]
    bad_copy = list(bad)
    validate_dialog(bad)
    assert bad == bad_copy


# --- DialogReport ---


def test_dialog_report_domyslny():
    a = DialogReport()
    b = DialogReport()
    assert a.files == 0
    assert a.found == 0
    assert a.accepted == 0
    assert a.rejected == {}
    assert a.duplicates == {}
    assert a.rejected is not b.rejected
    assert a.duplicates is not b.duplicates
    a.rejected["x"] = "y"
    assert b.rejected == {}
    assert DialogReport(files=1) == DialogReport(files=1)


# --- load_dialogs ---


def test_load_przyklad_z_opisu(tmp_path):
    text = (
        "A: Hej.\nB: Cześć.\nA: Co tam?\nB: Dobrze.\n---\n"
        "A: Hej.\nC: x\nA: Co tam?\nB: Dobrze.\n---\n"
        "A: HEJ.\nB: CZEŚĆ.\nA: CO TAM?\nB: DOBRZE.\n---\n"
    )
    src = write_src(tmp_path / "src", {"001.txt": text})
    result = load_dialogs(src)
    assert isinstance(result, tuple)
    assert isinstance(result[0], list)
    assert result == (
        [V1],
        DialogReport(
            files=1,
            found=3,
            accepted=1,
            rejected={"001.txt#2": "zla-linia"},
            duplicates={"001.txt#3": "001.txt#1"},
        ),
    )


def test_load_kilka_plikow_kolejnosc_i_zrodla(tmp_path):
    src = tmp_path / "src"
    write_src(src, {"002.txt": S([V3, V2, ["A: a", "B: b"]])})
    write_src(src, {"001.txt": S([V1, V2])})
    dialogs, report = load_dialogs(src)
    assert dialogs == [V1, V2, V3]
    assert report == DialogReport(
        files=2,
        found=5,
        accepted=3,
        rejected={"002.txt#3": "dlugosc"},
        duplicates={"002.txt#2": "001.txt#2"},
    )


def test_load_duplikat_wskazuje_pierwsze_zrodlo(tmp_path):
    src = write_src(tmp_path / "src", {"001.txt": S([V1]), "002.txt": S([V1, V1])})
    dialogs, report = load_dialogs(src)
    assert dialogs == [V1]
    assert report.duplicates == {"002.txt#1": "001.txt#1", "002.txt#2": "001.txt#1"}
    assert report.accepted == 1
    assert report.found == 3


def test_load_duplikat_wielkosc_liter_biale_znaki_nfc(tmp_path):
    block2 = ["A: Hej.  ", "B: CZEŚĆ.", "A: co   tam?", "B: dobrze."]
    v2_nfd_lines = [unicodedata.normalize("NFD", line) for line in V2]
    assert v2_nfd_lines != V2
    text = "".join("\n".join(lines) + "\n---\n" for lines in [V1, block2, V2, v2_nfd_lines])
    src = write_src(tmp_path / "src", {"001.txt": text})
    dialogs, report = load_dialogs(src)
    assert report.found == 4
    assert report.accepted == 2
    assert report.duplicates == {"001.txt#2": "001.txt#1", "001.txt#4": "001.txt#3"}
    assert dialogs == [V1, V2]


def test_load_rozne_rozmowy_nie_sa_duplikatami(tmp_path):
    other = ["A: Hej.", "B: Cześć!", "A: Co tam?", "B: Dobrze."]
    src = write_src(tmp_path / "src", {"001.txt": S([V1, other])})
    _, report = load_dialogs(src)
    assert report.accepted == 2
    assert report.duplicates == {}


def test_load_odrzucona_nie_rejestruje_klucza(tmp_path):
    src = write_src(tmp_path / "src", {"001.txt": S([["A: a", "B: b"], ["A: a", "B: b"]])})
    assert load_dialogs(src) == (
        [],
        DialogReport(
            files=1,
            found=2,
            accepted=0,
            rejected={"001.txt#1": "dlugosc", "001.txt#2": "dlugosc"},
            duplicates={},
        ),
    )
    assert load_dialogs(src, min_lines=2) == (
        [["A: a", "B: b"]],
        DialogReport(
            files=1,
            found=2,
            accepted=1,
            rejected={},
            duplicates={"001.txt#2": "001.txt#1"},
        ),
    )


def test_load_numeracja_blokow_niepustych(tmp_path):
    text = "---\n" + "\n".join(V1) + "\n---\n\n---\n" + "\n".join(["A: x", "C: y"]) + "\n---\n"
    src = write_src(tmp_path / "src", {"001.txt": text})
    dialogs, report = load_dialogs(src)
    assert dialogs == [V1]
    assert report.rejected == {"001.txt#2": "zla-linia"}
    assert report.found == 2


def test_load_bom_crlf_bez_koncowego_separatora(tmp_path):
    bom = bytes([0xEF, 0xBB, 0xBF])
    src1 = write_src(tmp_path / "s1", {"001.txt": bom + S([V1]).encode("utf-8")})
    assert load_dialogs(src1)[0] == [V1]

    crlf = S([V1]).replace("\n", "\r\n").encode("utf-8")
    src2 = write_src(tmp_path / "s2", {"002.txt": crlf})
    dialogs2 = load_dialogs(src2)[0]
    assert dialogs2 == [V1]
    assert all("\r" not in line for line in dialogs2[0])

    src3 = write_src(tmp_path / "s3", {"003.txt": "\n".join(V1)})
    assert load_dialogs(src3)[0] == [V1]


def test_load_pusty_plik_i_katalog_bez_txt(tmp_path):
    src1 = write_src(tmp_path / "s1", {"001.txt": b""})
    assert load_dialogs(src1) == ([], DialogReport(files=1))

    src2 = write_src(tmp_path / "s2", {"001.md": S([V1])})
    (src2 / "podkatalog").mkdir()
    assert load_dialogs(src2) == ([], DialogReport())

    src3 = tmp_path / "s3"
    src3.mkdir()
    assert load_dialogs(src3) == ([], DialogReport())


def test_load_brak_katalogu(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dialogs(tmp_path / "nie_ma")
    plik = write_src(tmp_path / "dir", {"001.txt": S([V1])}) / "001.txt"
    with pytest.raises(FileNotFoundError):
        load_dialogs(plik)


def test_load_granice_dlugosci_przekazywane(tmp_path):
    src = write_src(tmp_path / "src", {"001.txt": S([["A: a", "B: b"], V1])})
    assert load_dialogs(src)[0] == [V1]
    assert load_dialogs(src, min_lines=2)[1].accepted == 2
    dialogs, report = load_dialogs(src, min_lines=2, max_lines=2)
    assert dialogs == [["A: a", "B: b"]]
    assert report.rejected == {"001.txt#2": "dlugosc"}


def test_load_niezmienia_plikow(tmp_path):
    src = tmp_path / "src"
    bytes1 = bytes([0xEF, 0xBB, 0xBF]) +S([V1]).replace("\n", "\r\n").encode("utf-8")
    bytes2 = S([V2]).encode("utf-8")
    write_src(src, {"001.txt": bytes1, "002.txt": bytes2})
    names_before = sorted(p.name for p in src.iterdir())
    load_dialogs(src)
    assert (src / "001.txt").read_bytes() == bytes1
    assert (src / "002.txt").read_bytes() == bytes2
    assert sorted(p.name for p in src.iterdir()) == names_before


def test_load_niezmiennik_liczb(tmp_path):
    src = write_src(
        tmp_path / "src",
        {
            "001.txt": S([V1, V2, ["A: a", "C: b"]]),
            "002.txt": S([V2, V3, ["A: a", "B: b"]]),
        },
    )
    dialogs, report = load_dialogs(src)
    assert report.found == report.accepted + len(report.rejected) + len(report.duplicates)
    assert report.accepted == len(dialogs)


# --- split_dialogs ---


def test_split_dialogs_przyklad():
    d = [D(0), D(1), D(2), D(3)]
    assert split_dialogs(d, 0.25, 0) == ([D(0), D(1), D(3)], [D(2)])


def test_split_dialogs_seed():
    d = [D(0), D(1), D(2), D(3)]
    assert split_dialogs(d, 0.25, 1) == ([D(0), D(1), D(2)], [D(3)])
    assert split_dialogs(d, 0.25, 2) == ([D(0), D(2), D(3)], [D(1)])
    assert split_dialogs(d, 0.25, 0) == split_dialogs(d, 0.25, 0)


def test_split_dialogs_dziesiec():
    d = [D(i) for i in range(10)]
    train, val = split_dialogs(d, 0.2, 0)
    assert val == [D(7), D(8)]
    assert train == [D(0), D(1), D(2), D(3), D(4), D(5), D(6), D(9)]


def test_split_dialogs_dwie_rozmowy():
    assert split_dialogs([D(0), D(1)], 0.05, 0) == ([D(1)], [D(0)])
    assert split_dialogs([D(0), D(1)], 0.05, 1) == ([D(0)], [D(1)])


def test_split_dialogs_kompletnosc_rozlacznosc_i_kopie():
    d = [D(i) for i in range(10)]
    copy = [list(x) for x in d]
    train, val = split_dialogs(d, 0.3, 5)
    assert len(train) + len(val) == 10
    assert train
    assert val
    for dialog in train:
        assert dialog not in val
    assert sorted(train + val) == sorted(d)
    train_idx = [d.index(x) for x in train]
    val_idx = [d.index(x) for x in val]
    assert train_idx == sorted(train_idx)
    assert val_idx == sorted(val_idx)
    assert d == copy
    assert train is not d
    assert val is not d


def test_split_dialogs_bledy():
    with pytest.raises(ValueError):
        split_dialogs([D(0)])
    with pytest.raises(ValueError):
        split_dialogs([])
    with pytest.raises(ValueError):
        split_dialogs([D(0), D(1)], val_fraction=0.0)
    with pytest.raises(ValueError):
        split_dialogs([D(0), D(1)], val_fraction=1.0)


# --- format_dialog_stream ---


def test_format_stream_przyklady():
    assert (
        format_dialog_stream([["A: Hej.", "B: Cześć."], ["A: Co?", "B: Nic."]])
        == "A: Hej.\nB: Cześć.\n\nA: Co?\nB: Nic.\n"
    )
    assert format_dialog_stream([["A: Hej.", "B: Cześć."]]) == "A: Hej.\nB: Cześć.\n"
    assert format_dialog_stream([]) == ""
    assert format_dialog_stream([D(0), D(1), D(2)]) == T(0) + "\n" + T(1) + "\n" + T(2)


def test_format_stream_parsowanie_z_powrotem():
    d = [D(0), D(1), D(2)]
    text = format_dialog_stream(d)
    assert [blok.split("\n") for blok in text[:-1].split("\n\n")] == d
    assert text.endswith("\n")
    assert not text.endswith("\n\n")
    assert "\r" not in text


# --- build_dialog_corpus ---


def _scenariusz_przykladowy(tmp_path: Path) -> tuple[Path, Path]:
    src = write_src(tmp_path / "src", {"001.txt": S([D(0), D(1), D(2), D(3)])})
    out = tmp_path / "wyniki" / "podkatalog"
    return src, out


def _scenariusz_z_bledami(tmp_path: Path) -> Path:
    bad_pairs = ["A: a", "B: b"] * 9
    write_src(
        tmp_path / "src",
        {
            "001.txt": S([D(0), D(1), D(2), D(3)]),
            "002.txt": S(
                [
                    ["A: a", "C: b", "A: c", "B: d"],
                    ["A: a", "B: b"],
                    D(0),
                    ["B: a", "A: b", "B: c", "A: d"],
                    ["A: a", "B: b", "A: c", "B: d", "A: e"],
                    bad_pairs,
                ]
            ),
        },
    )
    return tmp_path / "src"


def test_build_przyklad_z_opisu(tmp_path):
    src, out = _scenariusz_przykladowy(tmp_path)
    result = build_dialog_corpus(src, out, val_fraction=0.25, seed=0)
    assert isinstance(result, DialogCorpusResult)
    assert result.train == 3
    assert result.val == 1
    assert result.train_path == out / "train.txt"
    assert result.val_path == out / "val.txt"
    assert result.report == DialogReport(files=1, found=4, accepted=4)
    assert read_text(out / "train.txt") == T(0) + "\n" + T(1) + "\n" + T(3)
    assert read_text(out / "val.txt") == T(2)
    assert sorted(p.name for p in out.iterdir()) == ["report.json", "train.txt", "val.txt"]


def test_build_bajty_bez_cr_i_bom(tmp_path):
    dialogs = [V1, V2, V3, ["A: Źdźbło.", "B: Żółć.", "A: Gęś?", "B: Jaźń."]]
    src = write_src(tmp_path / "src", {"001.txt": S(dialogs)})
    out = tmp_path / "out"
    build_dialog_corpus(src, out, val_fraction=0.25, seed=0)
    for name in ["train.txt", "val.txt", "report.json"]:
        data = (out / name).read_bytes()
        assert b"\r" not in data
        assert not data.startswith(b"\xef\xbb\xbf")
        data.decode("utf-8")
    for name in ["train.txt", "val.txt"]:
        data = (out / name).read_bytes()
        assert data.endswith(b"\n")
        assert not data.endswith(b"\n\n")


def test_build_report_json_format_i_klucze(tmp_path):
    src, out = _scenariusz_przykladowy(tmp_path)
    build_dialog_corpus(src, out, val_fraction=0.25, seed=0)
    t = read_text(out / "report.json")
    expected = {
        "seed": 0,
        "val_fraction": 0.25,
        "min_lines": 4,
        "max_lines": 16,
        "files": 1,
        "found": 4,
        "accepted": 4,
        "train": 3,
        "val": 1,
        "rejected": {},
        "duplicates": {},
        "reasons": {},
    }
    assert json.loads(t) == expected
    assert list(json.loads(t)) == list(expected)
    assert t == json.dumps(json.loads(t), ensure_ascii=False, indent=2) + "\n"


def test_build_odrzucone_duplikaty_i_reasons(tmp_path):
    src = _scenariusz_z_bledami(tmp_path)
    out = tmp_path / "out"
    result = build_dialog_corpus(src, out, val_fraction=0.25, seed=0)
    assert result.train == 3
    assert result.val == 1
    rejected = {
        "002.txt#1": "zla-linia",
        "002.txt#2": "dlugosc",
        "002.txt#4": "start",
        "002.txt#5": "koniec",
        "002.txt#6": "dlugosc",
    }
    duplicates = {"002.txt#3": "001.txt#1"}
    assert result.report.found == 10
    assert result.report.accepted == 4
    assert result.report.rejected == rejected
    assert result.report.duplicates == duplicates

    data = json.loads(read_text(out / "report.json"))
    assert data["found"] == 10
    assert data["accepted"] == 4
    assert data["files"] == 2
    assert data["rejected"] == rejected
    assert data["duplicates"] == duplicates
    assert data["reasons"] == {"dlugosc": 2, "koniec": 1, "start": 1, "zla-linia": 1}
    assert list(data["reasons"]) == ["dlugosc", "koniec", "start", "zla-linia"]
    assert read_text(out / "train.txt") == T(0) + "\n" + T(1) + "\n" + T(3)
    assert read_text(out / "val.txt") == T(2)


def test_build_wlasne_granice_i_seed(tmp_path):
    two_line = [[f"A: a{i}", f"B: b{i}"] for i in range(4)]
    src = write_src(tmp_path / "src", {"001.txt": S(two_line)})

    out0 = tmp_path / "out0"
    result = build_dialog_corpus(
        src, out0, val_fraction=0.25, seed=0, min_lines=2, max_lines=16
    )
    assert result.train == 3
    assert result.val == 1
    assert read_text(out0 / "train.txt") == "A: a0\nB: b0\n\nA: a1\nB: b1\n\nA: a3\nB: b3\n"
    assert read_text(out0 / "val.txt") == "A: a2\nB: b2\n"
    data0 = json.loads(read_text(out0 / "report.json"))
    assert data0["min_lines"] == 2
    assert data0["max_lines"] == 16
    assert data0["seed"] == 0

    out1 = tmp_path / "out1"
    build_dialog_corpus(src, out1, val_fraction=0.25, seed=1, min_lines=2, max_lines=16)
    assert read_text(out1 / "val.txt") == "A: a3\nB: b3\n"
    assert read_text(out1 / "train.txt") == "A: a0\nB: b0\n\nA: a1\nB: b1\n\nA: a2\nB: b2\n"
    assert json.loads(read_text(out1 / "report.json"))["seed"] == 1

    with pytest.raises(ValueError):
        build_dialog_corpus(src, tmp_path / "out2", val_fraction=0.25, seed=0)


def test_build_determinizm(tmp_path):
    src = _scenariusz_z_bledami(tmp_path)
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    build_dialog_corpus(src, out1, val_fraction=0.25, seed=0)
    build_dialog_corpus(src, out2, val_fraction=0.25, seed=0)
    for name in ["train.txt", "val.txt", "report.json"]:
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes()


def test_build_niezmiennik_rozlacznosci_i_calosci(tmp_path):
    d = [D(i) for i in range(10)]
    src = write_src(tmp_path / "src", {"001.txt": S(d)})
    out = tmp_path / "out"
    result = build_dialog_corpus(src, out, val_fraction=0.2, seed=0)
    train_text = read_text(out / "train.txt")
    val_text = read_text(out / "val.txt")
    train_dialogs = train_text[:-1].split("\n\n")
    val_dialogs = val_text[:-1].split("\n\n")
    assert len(train_dialogs) == 8
    assert len(val_dialogs) == 2
    assert result.train == 8
    assert result.val == 2
    assert set(train_dialogs).isdisjoint(val_dialogs)
    assert sorted(train_dialogs + val_dialogs) == sorted(T(i)[:-1] for i in range(10))
    for dialog in train_dialogs + val_dialogs:
        lines = dialog.split("\n")
        assert lines[0].startswith("A: ")
        assert lines[-1].startswith("B: ")
    assert val_text == T(7) + "\n" + T(8)


def test_build_bledy(tmp_path):
    src_bad = write_src(tmp_path / "bad", {"001.txt": S([["A: a", "B: b"]])})
    with pytest.raises(ValueError):
        build_dialog_corpus(src_bad, tmp_path / "o1")

    src_one = write_src(tmp_path / "one", {"001.txt": S([D(0)])})
    with pytest.raises(ValueError):
        build_dialog_corpus(src_one, tmp_path / "o2")

    src_empty = tmp_path / "empty"
    src_empty.mkdir()
    with pytest.raises(ValueError):
        build_dialog_corpus(src_empty, tmp_path / "o3")

    with pytest.raises(FileNotFoundError):
        build_dialog_corpus(tmp_path / "nie_ma", tmp_path / "o4")

    src_four = write_src(tmp_path / "four", {"001.txt": S([D(0), D(1), D(2), D(3)])})
    with pytest.raises(ValueError):
        build_dialog_corpus(src_four, tmp_path / "o5", val_fraction=0.0)
    with pytest.raises(ValueError):
        build_dialog_corpus(src_four, tmp_path / "o6", val_fraction=1.0)


def test_build_niezmienia_wejscia_i_nic_nie_wypisuje(tmp_path, capsys):
    src, out = _scenariusz_przykladowy(tmp_path)
    before = (src / "001.txt").read_bytes()
    build_dialog_corpus(src, out, val_fraction=0.25, seed=0)
    assert (src / "001.txt").read_bytes() == before
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
