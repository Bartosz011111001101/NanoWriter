"""Testy BPETokenizer, SPLIT_PATTERN i sample_lines (tokenizer, prompt 01)."""

import json
import math
import random
import re
import time
from collections import Counter

import pytest

from nanowriter.tokenizer import SPLIT_PATTERN, BPETokenizer, CharTokenizer, sample_lines

POL = (
    "Litwo! Ojczyzno moja! ty jesteś jak zdrowie.\n"
    "Ile cię trzeba cenić, ten tylko się dowie,\n"
    "Kto cię stracił. Dziś piękność twą w całej ozdobie\n"
    "Widzę i opisuję, bo tęsknię po tobie.\n"
)
PATTERN = r"[^\S\n]?\w+|[^\S\n]?[^\w\s]+|\n|[^\S\n]+"
ROUNDTRIP = [
    "",
    "Zażółć gęślą jaźń",
    "😀👍🏽",
    "\n\t",
    "\r\n",
    "Ψ∑ 中文 xyz123",
    "  wiele   spacji  \n\n koniec",
    "ąąąą ęęęę\n\n\n",
    POL,
]


def _replace(seq, pair, new):
    out = []
    i = 0
    while i < len(seq):
        if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
            out.append(new)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def naive_train(text, vocab_size):
    counts = Counter(re.findall(SPLIT_PATTERN, text))
    words = [(list(k.encode("utf-8")), n) for k, n in counts.items()]
    merges = []
    while 256 + len(merges) < vocab_size:
        pairs = Counter()
        for w, n in words:
            for pair in zip(w, w[1:], strict=False):
                pairs[pair] += n
        if not pairs:
            break
        top = max(pairs.values())
        best = min(pair for pair, n in pairs.items() if n == top)
        new_id = 256 + len(merges)
        merges.append(best)
        words = [(_replace(w, best, new_id), n) for w, n in words]
    return merges


def naive_encode(merges, text):
    ranks = {pair: i for i, pair in enumerate(merges)}
    result = []
    for chunk in re.findall(SPLIT_PATTERN, text):
        seq = list(chunk.encode("utf-8"))
        while len(seq) >= 2:
            candidates = [p for p in zip(seq, seq[1:], strict=False) if p in ranks]
            if not candidates:
                break
            pair = min(candidates, key=lambda p: ranks[p])
            seq = _replace(seq, pair, 256 + ranks[pair])
        result.extend(seq)
    return result


def syllable_text(n_chars, seed):
    rng = random.Random(seed)
    syl = ["ka", "to", "la", "ma", "ży", "cie", "rz", "szcz", "ą", "ę", "ó", "ł", "ść", "no",
           "wie", "dzie", "prze", "nia"]
    words = []
    total = 0
    while total < n_chars:
        word = "".join(rng.choice(syl) for _ in range(rng.randint(1, 4)))
        words.append(word)
        total += len(word) + 1
    return " ".join(words)


def losowy(alfabet, n, seed):
    rng = random.Random(seed)
    return "".join(rng.choice(alfabet) for _ in range(n))


@pytest.fixture
def t258():
    return BPETokenizer.train("aaab", 258)


# --- SPLIT_PATTERN i kawałki ---


def test_split_pattern_value():
    assert isinstance(SPLIT_PATTERN, str)
    assert SPLIT_PATTERN == PATTERN


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a, b.\n\nc", ["a", ",", " b", ".", "\n", "\n", "c"]),
        ("Zażółć gęślą jaźń", ["Zażółć", " gęślą", " jaźń"]),
        ("ok 😀!", ["ok", " 😀!"]),
        ("a\n\n b\t\n c", ["a", "\n", "\n", " b", "\t", "\n", " c"]),
        ("x  \n\ty", ["x", "  ", "\n", "\ty"]),
        ("\r\n", ["\r", "\n"]),
        ("😀👍🏽", ["😀👍🏽"]),
    ],
)
def test_split_pattern_examples(text, expected):
    assert re.findall(SPLIT_PATTERN, text) == expected


def test_split_pattern_covers_text():
    for x in [*ROUNDTRIP, syllable_text(2000, 1)]:
        chunks = re.findall(SPLIT_PATTERN, x)
        assert "".join(chunks) == x
        for chunk in chunks:
            if "\n" in chunk:
                assert chunk == "\n"


# --- __init__, vocab_size, merges ---


def test_init_docstring_examples():
    assert BPETokenizer([(97, 97)]).vocab_size == 257
    assert BPETokenizer([]).encode("ż") == [197, 188]
    assert BPETokenizer([(97, 97), (256, 98)]).vocab_size == 258
    assert BPETokenizer([(97, 97)]).merges == [(97, 97)]
    empty = BPETokenizer([])
    assert empty.vocab_size == 256
    assert empty.merges == []


@pytest.mark.parametrize(
    "merges",
    [
        [(97, 256)],
        [(256, 97)],
        [(97, 97), (97, 97)],
        [(97, 97), (257, 97)],
        [(-1, 97)],
        [(97, -1)],
    ],
)
def test_init_invalid_merges(merges):
    with pytest.raises(ValueError):
        BPETokenizer(merges)


def test_init_boundary_ids_allowed():
    t = BPETokenizer([(255, 255), (256, 256)])
    assert t.vocab_size == 258
    assert t.decode([257]) == bytes([255] * 4).decode("utf-8", errors="replace")


def test_init_copies_input():
    lst = [(97, 97)]
    t = BPETokenizer(lst)
    lst.append((98, 98))
    assert t.vocab_size == 257
    assert t.merges == [(97, 97)]


def test_merges_returns_copy():
    t = BPETokenizer([(97, 97)])
    m = t.merges
    m.append((98, 98))
    m[0] = (1, 2)
    assert t.merges == [(97, 97)]
    assert t.vocab_size == 257
    assert t.encode("aa") == [256]


def test_types(t258):
    assert isinstance(t258.vocab_size, int)
    assert isinstance(t258.merges, list)
    for pair in t258.merges:
        assert isinstance(pair, tuple)
        assert len(pair) == 2
        assert all(isinstance(i, int) for i in pair)
    ids = t258.encode("aaab")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert isinstance(t258.decode([256]), str)
    assert isinstance(BPETokenizer.train("aaab", 258), BPETokenizer)
    assert not issubclass(BPETokenizer, CharTokenizer)


# --- train ---


def test_train_docstring_examples():
    assert BPETokenizer.train("aaab", 257).merges == [(97, 97)]
    assert BPETokenizer.train("aaab", 258).merges == [(97, 97), (97, 98)]
    assert BPETokenizer.train("ala ma kota", 260).merges == [
        (32, 107),
        (32, 109),
        (97, 108),
        (111, 116),
    ]
    ab = BPETokenizer.train("ab", 300)
    assert ab.vocab_size == 257
    assert ab.merges == [(97, 98)]
    assert BPETokenizer.train("", 300).vocab_size == 256
    with pytest.raises(ValueError):
        BPETokenizer.train("abc", 255)


@pytest.mark.parametrize(("text", "vocab_size"), [("aaab", 0), ("aaab", -1), ("", 255)])
def test_train_invalid_vocab_size(text, vocab_size):
    with pytest.raises(ValueError):
        BPETokenizer.train(text, vocab_size)


def test_train_vocab_256_has_no_merges():
    t = BPETokenizer.train("aaab", 256)
    assert t.merges == []
    assert t.vocab_size == 256


def test_train_overlapping_pairs():
    t4 = BPETokenizer.train("aaaa", 300)
    assert t4.merges == [(97, 97), (256, 256)]
    assert t4.vocab_size == 258
    assert BPETokenizer.train("aaaaaaa", 300).merges == [
        (97, 97),
        (256, 256),
        (256, 97),
        (257, 258),
    ]


def test_train_respects_chunk_boundaries():
    assert BPETokenizer.train("a b", 300).merges == [(32, 98)]
    assert BPETokenizer.train("a\na\na", 300).merges == []
    assert BPETokenizer.train("ab\nab", 300).merges == [(97, 98)]


@pytest.mark.parametrize(
    ("text", "vocab_size"),
    [
        ("aaab", 258),
        ("aaaa", 300),
        ("aaaaaaa", 300),
        ("ala ma kota", 300),
        ("", 300),
        (POL, 400),
        (POL * 3 + "Zażółć gęślą jaźń 😀😀😀\n", 350),
        (losowy("ab \n", 3000, 0), 320),
        (losowy("aab", 500, 1), 300),
        (syllable_text(20_000, 2), 400),
    ],
    ids=["aaab", "aaaa", "a7", "ala", "empty", "pol", "pol3", "rand_ab", "rand_aab", "syl"],
)
def test_train_matches_naive(text, vocab_size):
    assert BPETokenizer.train(text, vocab_size).merges == naive_train(text, vocab_size)


# --- encode ---


def test_encode_docstring_examples(t258):
    assert t258.encode("aaab") == [256, 257]
    assert t258.encode("aaa") == [256, 97]
    assert t258.encode("ab ab") == [257, 32, 257]
    assert BPETokenizer.train("ala ma kota", 260).encode("ala ma kota") == [
        258, 97, 257, 97, 256, 259, 97,
    ]
    assert t258.encode("") == []
    assert BPETokenizer([]).encode("") == []


def test_encode_more_examples(t258):
    assert t258.encode("aaab aaab") == [256, 257, 32, 256, 257]
    t4 = BPETokenizer.train("aaaa", 300)
    assert t4.encode("aaaa") == [257]
    assert t4.encode("aaaaa") == [257, 97]
    assert BPETokenizer.train("aaaaaaa", 300).encode("aaaaaaa") == [259]
    assert BPETokenizer([]).encode("😀") == [240, 159, 152, 128]


def test_encode_matches_naive():
    t = BPETokenizer.train(POL, 300)
    for x in [*ROUNDTRIP, syllable_text(3000, 3)]:
        assert t.encode(x) == naive_encode(t.merges, x)


def test_encode_cache_not_shared():
    t1 = BPETokenizer.train("aaab", 258)
    t1.encode("aaab")
    t2 = BPETokenizer([])
    assert t2.encode("aaab") == [97, 97, 97, 98]
    assert t1.encode("aaab") == [256, 257]


def test_encode_ids_in_range():
    t = BPETokenizer.train(POL, 300)
    for x in ROUNDTRIP:
        for i in t.encode(x):
            assert 0 <= i < t.vocab_size


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Litwo! Ojczyzno moja!\n", "ty jesteś jak zdrowie."),
        ("\n", "\nx"),
        ("Zażółć\n", ""),
        ("x\n", " y"),
        ("ąę \n", "\t😀"),
    ],
)
def test_encode_concat_after_newline(a, b):
    t = BPETokenizer.train(POL, 300)
    assert t.encode(a + b) == t.encode(a) + t.encode(b)


# --- decode ---


def test_decode_docstring_examples():
    assert BPETokenizer([(97, 97)]).decode([256, 98]) == "aab"
    assert BPETokenizer([]).decode([197, 188]) == "ż"
    assert BPETokenizer([]).decode([197]) == "�"
    assert BPETokenizer([]).decode([]) == ""
    with pytest.raises(ValueError):
        BPETokenizer([]).decode([256])


def test_decode_invalid_bytes_in_middle():
    assert BPETokenizer([]).decode([97, 255, 98]) == "a�b"


def test_decode_out_of_range():
    t = BPETokenizer([(97, 97)])
    for ids in ([257], [-1], [97, 300]):
        with pytest.raises(ValueError):
            t.decode(ids)
    assert t.decode([256]) == "aa"
    assert t.decode([0]) == "\x00"


@pytest.mark.parametrize("x", ROUNDTRIP)
def test_roundtrip(x):
    for t in (BPETokenizer.train(POL, 300), BPETokenizer([])):
        assert t.decode(t.encode(x)) == x


# --- save / load ---


def test_save_exact_file(tmp_path):
    p = tmp_path / "t.json"
    BPETokenizer([(97, 97)]).save(p)
    expected_literal = (
        b'{\n  "type": "bpe-bytes",\n'
        b'  "pattern": "[^\\\\S\\\\n]?\\\\w+|[^\\\\S\\\\n]?[^\\\\w\\\\s]+|\\\\n|[^\\\\S\\\\n]+",\n'
        b'  "merges": [\n    [\n      97,\n      97\n    ]\n  ]\n}\n'
    )
    expected_dumps = (
        json.dumps({"type": "bpe-bytes", "pattern": SPLIT_PATTERN, "merges": [[97, 97]]}, indent=2)
        + "\n"
    ).encode("utf-8")
    data = p.read_bytes()
    assert data == expected_literal
    assert data == expected_dumps

    p2 = tmp_path / "e.json"
    BPETokenizer([]).save(p2)
    data2 = p2.read_bytes()
    assert data2 == (
        json.dumps({"type": "bpe-bytes", "pattern": SPLIT_PATTERN, "merges": []}, indent=2) + "\n"
    ).encode("utf-8")
    assert b"\r" not in data2
    assert b"\r" not in data


def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "t.json"
    t = BPETokenizer.train(POL, 300)
    t.save(p)
    u = BPETokenizer.load(p)
    assert u.merges == t.merges
    assert all(isinstance(m, tuple) for m in u.merges)
    assert u.vocab_size == t.vocab_size
    for x in ROUNDTRIP:
        assert u.encode(x) == t.encode(x)


def test_load_merges_are_tuples(tmp_path):
    p = tmp_path / "t.json"
    content = json.dumps(
        {"type": "bpe-bytes", "pattern": SPLIT_PATTERN, "merges": [[97, 97], [256, 98]]}
    )
    p.write_bytes(content.encode("utf-8"))
    t = BPETokenizer.load(p)
    assert t.merges == [(97, 97), (256, 98)]
    assert all(isinstance(m, tuple) for m in t.merges)
    assert t.vocab_size == 258
    assert t.decode([257]) == "aab"


def _good_file():
    return {"type": "bpe-bytes", "pattern": SPLIT_PATTERN, "merges": [[97, 97]]}


def _mod(**changes):
    d = _good_file()
    d.update(changes)
    return d


def _without(key):
    d = _good_file()
    del d[key]
    return d


@pytest.mark.parametrize(
    "content",
    [
        _mod(type="inny"),
        _mod(pattern="\\w+"),
        _without("type"),
        _without("pattern"),
        _without("merges"),
        _mod(merges=[[97, 256]]),
        _mod(merges=[[97, 97], [97, 97]]),
    ],
    ids=["type", "pattern", "no_type", "no_pattern", "no_merges", "bad_id", "duplicate"],
)
def test_load_invalid_file(tmp_path, content):
    p = tmp_path / "t.json"
    p.write_bytes(json.dumps(content).encode("utf-8"))
    with pytest.raises(ValueError):
        BPETokenizer.load(p)


def test_save_does_not_create_dirs(tmp_path):
    with pytest.raises(OSError):
        BPETokenizer([]).save(tmp_path / "brak" / "t.json")
    assert not (tmp_path / "brak").exists()


# --- sample_lines ---


def test_sample_lines_docstring_examples():
    assert sample_lines("a\nb\nc\nd", 4) == "a\nc"
    assert sample_lines("abc", 10) == "abc"
    assert sample_lines("", 5) == ""
    for text, n in (("abc", 0), ("abc", -1), ("", 0)):
        with pytest.raises(ValueError):
            sample_lines(text, n)


@pytest.mark.parametrize(
    ("text", "max_chars", "expected"),
    [
        ("a\nb\nc", 5, "a\nb\nc"),
        ("a\nb\nc", 4, "a\nc"),
        ("a\nb\nc", 1, "a"),
        ("abc", 1, "abc"),
        ("1\n2\n3\n4\n5\n6\n7", 5, "1\n4\n7"),
        ("a\nb\nc\nd\n", 4, "a\nc\n"),
        ("ąę\nść", 5, "ąę\nść"),
        ("ąę\nść", 4, "ąę"),
    ],
)
def test_sample_lines_boundaries(text, max_chars, expected):
    assert sample_lines(text, max_chars) == expected


def test_sample_lines_whole_lines_in_order():
    text = "\n".join(f"linia {i} " + "x" * (i % 7) for i in range(1000))
    out = sample_lines(text, 500)
    k = math.ceil(len(text) / 500)
    assert out.split("\n") == text.split("\n")[::k]
    lines = set(text.split("\n"))
    assert all(line in lines for line in out.split("\n"))
    assert out.split("\n")[0] == "linia 0 "


# --- ogólne ---


def test_module_prints_nothing(tmp_path, capsys):
    t = BPETokenizer.train(POL, 300)
    t.decode(t.encode(POL))
    t.save(tmp_path / "t.json")
    BPETokenizer.load(tmp_path / "t.json")
    sample_lines(POL, 50)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_train_speed():
    text = syllable_text(200_000, 0)
    start = time.perf_counter()
    t = BPETokenizer.train(text, 1024)
    elapsed = time.perf_counter() - start
    assert elapsed < 5
    assert t.vocab_size == 1024


def test_encode_speed():
    t = BPETokenizer.train(syllable_text(20_000, 4), 512)
    text = syllable_text(200_000, 5)
    start = time.perf_counter()
    ids = t.encode(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 5
    assert t.decode(ids) == text
