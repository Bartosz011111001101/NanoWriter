"""Tokenizery: znakowy (etap 1) i BPE na bajtach (etap 3)."""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

SPLIT_PATTERN: str = r"[^\S\n]?\w+|[^\S\n]?[^\w\s]+|\n|[^\S\n]+"


class CharTokenizer:
    """Tokenizer, w którym id znaku to jego pozycja na liście chars.

    Przykład: CharTokenizer(["a", "b", "c"]).encode("cab") -> [2, 0, 1]
    """

    def __init__(self, chars: list[str]) -> None:
        """Tworzy tokenizer z listy różnych pojedynczych znaków; id = pozycja na liście.

        Przykład: CharTokenizer(["x", "a"]).encode("ax") -> [1, 0]
        Przykład: CharTokenizer([]) -> tokenizer z vocab_size == 0
        Przykład: CharTokenizer(["a", "a"]) -> ValueError
        Przykład: CharTokenizer(["ab"]) -> ValueError
        Przykład: CharTokenizer([""]) -> ValueError
        """
        for element in chars:
            if len(element) != 1:
                raise ValueError(
                    f"element słownika musi mieć długość 1: {element!r}"
                )
        seen: set[str] = set()
        for element in chars:
            if element in seen:
                raise ValueError(f"znak powtarza się w słowniku: {element!r}")
            seen.add(element)
        self._chars: list[str] = list(chars)
        self._ids: dict[str, int] = {
            element: position for position, element in enumerate(self._chars)
        }

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Tworzy tokenizer ze wszystkich różnych znaków tekstu,
        ułożonych rosnąco według punktu kodowego.

        Przykład: CharTokenizer.from_text("abca").chars -> ["a", "b", "c"]
        Przykład: CharTokenizer.from_text("żaba").chars -> ["a", "b", "ż"]
        Przykład: CharTokenizer.from_text("") -> tokenizer z vocab_size == 0
        """
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        """Zwraca liczbę znaków w słowniku.

        Przykład: CharTokenizer.from_text("abca").vocab_size -> 3
        """
        return len(self._chars)

    @property
    def chars(self) -> list[str]:
        """Zwraca kopię listy znaków słownika (id = pozycja).

        Przykład: CharTokenizer(["x", "a"]).chars -> ["x", "a"]
        """
        return list(self._chars)

    def encode(self, text: str) -> list[int]:
        """Zamienia tekst na listę id znaków.

        Przykład: CharTokenizer.from_text("abca").encode("cab") -> [2, 0, 1]
        Przykład: CharTokenizer.from_text("żaba").encode("żaba") -> [2, 0, 1, 0]
        Przykład: encode("") -> []
        Przykład: CharTokenizer.from_text("abc").encode("d") -> ValueError
        """
        ids: list[int] = []
        for char in text:
            if char not in self._ids:
                raise ValueError(f"znak spoza słownika: {char!r}")
            ids.append(self._ids[char])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Zamienia listę id z powrotem na tekst.

        Przykład: CharTokenizer.from_text("abca").decode([2, 0, 1]) -> "cab"
        Przykład: decode([]) -> ""
        Przykład: CharTokenizer.from_text("abc").decode([3]) -> ValueError
        Przykład: CharTokenizer.from_text("abc").decode([-1]) -> ValueError
        """
        pieces: list[str] = []
        for token_id in ids:
            if not 0 <= token_id < len(self._chars):
                raise ValueError(f"id poza zakresem słownika: {token_id}")
            pieces.append(self._chars[token_id])
        return "".join(pieces)


def _merge_word(word: list[int], first: int, second: int, new_id: int) -> list[int]:
    """Zamienia wystąpienia pary (first, second) od lewej, bez nakładania, na new_id.

    Przykład: _merge_word([97, 97, 97], 97, 97, 256) -> [256, 97]
    """
    merged: list[int] = []
    position = 0
    while position < len(word):
        if position + 1 < len(word) and word[position] == first and word[position + 1] == second:
            merged.append(new_id)
            position += 2
        else:
            merged.append(word[position])
            position += 1
    return merged


class BPETokenizer:
    """Tokenizer BPE na bajtach UTF-8: id 0..255 to bajty, id 256 + i to scalenie merges[i].

    Przykład: BPETokenizer.train("aaab", 258).encode("aaab") -> [256, 257]
    """

    def __init__(self, merges: list[tuple[int, int]]) -> None:
        """Tworzy tokenizer z listy scaleń; scalenie i daje token o id 256 + i.

        Przykład: BPETokenizer([(97, 97)]).vocab_size -> 257
        Przykład: BPETokenizer([]).encode("ż") -> [197, 188]
        Przykład: BPETokenizer([(97, 256)]) -> ValueError (id 256 jeszcze nie istnieje)
        Przykład: BPETokenizer([(97, 97), (97, 97)]) -> ValueError (powtórzona para)
        """
        self._merges: list[tuple[int, int]] = []
        self._ranks: dict[tuple[int, int], int] = {}
        self._vocab: list[bytes] = [bytes([byte]) for byte in range(256)]
        for rank, (first, second) in enumerate(merges):
            limit = 256 + rank
            if not (0 <= first < limit and 0 <= second < limit):
                raise ValueError(
                    f"scalenie {rank} używa id spoza zakresu [0, {limit}): {(first, second)}"
                )
            pair = (first, second)
            if pair in self._ranks:
                raise ValueError(f"para powtarza się w scaleniach: {pair}")
            self._merges.append(pair)
            self._ranks[pair] = rank
            self._vocab.append(self._vocab[first] + self._vocab[second])
        self._cache: dict[str, tuple[int, ...]] = {}

    @classmethod
    def train(cls, text: str, vocab_size: int) -> "BPETokenizer":
        """Uczy scaleń na tekście, aż słownik ma vocab_size tokenów albo brak par.

        Przykład: BPETokenizer.train("aaab", 257).merges -> [(97, 97)]
        Przykład: BPETokenizer.train("aaab", 258).merges -> [(97, 97), (97, 98)]
        Przykład: BPETokenizer.train("ala ma kota", 260).merges
            -> [(32, 107), (32, 109), (97, 108), (111, 116)]
        Przykład: BPETokenizer.train("ab", 300).vocab_size -> 257 (brak dalszych par)
        Przykład: BPETokenizer.train("", 300).vocab_size -> 256
        Przykład: BPETokenizer.train("abc", 255) -> ValueError
        """
        if vocab_size < 256:
            raise ValueError(f"vocab_size musi być co najmniej 256: {vocab_size}")
        if not text:
            return cls([])
        chunk_counts = Counter(match.group() for match in re.finditer(SPLIT_PATTERN, text))
        words: list[list[int]] = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        weights: list[int] = list(chunk_counts.values())
        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_index: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
        for index, word in enumerate(words):
            for pair in zip(word, word[1:], strict=False):
                pair_counts[pair] += weights[index]
                pair_index[pair].add(index)
        merges: list[tuple[int, int]] = []
        while 256 + len(merges) < vocab_size and pair_counts:
            best = max(pair_counts, key=lambda pair: (pair_counts[pair], -pair[0], -pair[1]))
            new_id = 256 + len(merges)
            merges.append(best)
            for index in sorted(pair_index.pop(best, set())):
                word = words[index]
                weight = weights[index]
                for pair in zip(word, word[1:], strict=False):
                    pair_counts[pair] -= weight
                    if pair_counts[pair] == 0:
                        del pair_counts[pair]
                new_word = _merge_word(word, best[0], best[1], new_id)
                words[index] = new_word
                for pair in zip(new_word, new_word[1:], strict=False):
                    pair_counts[pair] += weight
                    pair_index[pair].add(index)
        return cls(merges)

    @property
    def vocab_size(self) -> int:
        """Zwraca liczbę tokenów: 256 + liczba scaleń.

        Przykład: BPETokenizer([(97, 97), (256, 98)]).vocab_size -> 258
        """
        return 256 + len(self._merges)

    @property
    def merges(self) -> list[tuple[int, int]]:
        """Zwraca kopię listy scaleń w kolejności.

        Przykład: BPETokenizer([(97, 97)]).merges -> [(97, 97)]
        """
        return list(self._merges)

    def encode(self, text: str) -> list[int]:
        """Zamienia tekst na listę id tokenów (kawałki SPLIT_PATTERN, scalenia wg kolejności).

        Przykład: BPETokenizer.train("aaab", 258).encode("aaab") -> [256, 257]
        Przykład: BPETokenizer.train("aaab", 258).encode("aaa") -> [256, 97]
        Przykład: BPETokenizer.train("aaab", 258).encode("ab ab") -> [257, 32, 257]
        Przykład: BPETokenizer.train("ala ma kota", 260).encode("ala ma kota")
            -> [258, 97, 257, 97, 256, 259, 97]
        Przykład: encode("") -> []
        """
        ids: list[int] = []
        for match in re.finditer(SPLIT_PATTERN, text):
            chunk = match.group()
            cached = self._cache.get(chunk)
            if cached is None:
                cached = self._encode_chunk(chunk)
                self._cache[chunk] = cached
            ids.extend(cached)
        return ids

    def _encode_chunk(self, chunk: str) -> tuple[int, ...]:
        """Koduje jeden kawałek tekstu scaleniami w kolejności ich numerów."""
        word = list(chunk.encode("utf-8"))
        while len(word) >= 2:
            pairs = zip(word, word[1:], strict=False)
            ranks = [self._ranks[pair] for pair in pairs if pair in self._ranks]
            if not ranks:
                break
            best_rank = min(ranks)
            first, second = self._merges[best_rank]
            word = _merge_word(word, first, second, 256 + best_rank)
        return tuple(word)

    def decode(self, ids: list[int]) -> str:
        """Zamienia id z powrotem na tekst; błędne sekwencje UTF-8 dają znak "�".

        Przykład: BPETokenizer([(97, 97)]).decode([256, 98]) -> "aab"
        Przykład: BPETokenizer([]).decode([197, 188]) -> "ż"
        Przykład: BPETokenizer([]).decode([197]) -> "�"
        Przykład: decode([]) -> ""
        Przykład: BPETokenizer([]).decode([256]) -> ValueError
        """
        size = self.vocab_size
        for token_id in ids:
            if not 0 <= token_id < size:
                raise ValueError(f"id poza zakresem słownika: {token_id}")
        return b"".join(self._vocab[token_id] for token_id in ids).decode(
            "utf-8", errors="replace"
        )

    def save(self, path: Path) -> None:
        """Zapisuje słownik jako JSON (type, pattern, merges) w UTF-8.

        Przykład: t.save(p); BPETokenizer.load(p).merges == t.merges -> True
        """
        data = {
            "type": "bpe-bytes",
            "pattern": SPLIT_PATTERN,
            "merges": [[first, second] for first, second in self._merges],
        }
        text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        path.write_text(text, encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        """Wczytuje słownik zapisany przez save.

        Przykład: BPETokenizer.load(p).encode(x) == t.encode(x) -> True (p z t.save(p))
        Przykład: plik z "type": "inny" albo innym "pattern" -> ValueError
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(data, dict)
            or data.get("type") != "bpe-bytes"
            or data.get("pattern") != SPLIT_PATTERN
            or not isinstance(data.get("merges"), list)
        ):
            raise ValueError(f"nieprawidłowy słownik BPE: {path}")
        merges: list[tuple[int, int]] = []
        for item in data["merges"]:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError(f"nieprawidłowe scalenie w słowniku: {item!r}")
            first, second = item
            if not isinstance(first, int) or not isinstance(second, int):
                raise ValueError(f"nieprawidłowe scalenie w słowniku: {item!r}")
            merges.append((first, second))
        return cls(merges)


def sample_lines(text: str, max_chars: int) -> str:
    r"""Zwraca co k-tą linię tekstu (k = ceil(len(text) / max_chars)), gdy tekst jest za długi.

    Przykład: sample_lines("a\nb\nc\nd", 4) -> "a\nc"
    Przykład: sample_lines("abc", 10) -> "abc"
    Przykład: sample_lines("", 5) -> ""
    Przykład: sample_lines("abc", 0) -> ValueError
    """
    if max_chars < 1:
        raise ValueError(f"max_chars musi być co najmniej 1: {max_chars}")
    if len(text) <= max_chars:
        return text
    step = math.ceil(len(text) / max_chars)
    return "\n".join(text.split("\n")[::step])


def sample_file(path: Path, max_chars: int) -> str:
    r"""Zwraca co k-tą linię pliku (k = ceil(liczba znaków / max_chars)), gdy plik jest za długi.

    Plik jest czytany liniami w dwóch przebiegach, bez wczytywania całości do pamięci.
    Wynik składa się z całych linii pliku razem z ich "\n".
    Przykład: plik "a\nb\nc\nd\n", max_chars=4 -> "a\nc\n"
    Przykład: plik "abc", max_chars=10 -> "abc"
    Przykład: pusty plik, max_chars=5 -> ""
    Przykład: plik "a\r\nb\r\n", max_chars=3 -> "a\n"
    Przykład: max_chars=0 -> ValueError
    """
    if max_chars < 1:
        raise ValueError(f"max_chars musi być co najmniej 1: {max_chars}")
    total = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            total += len(line)
    if total <= max_chars:
        return path.read_text(encoding="utf-8")
    step = math.ceil(total / max_chars)
    kept: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle):
            if number % step == 0:
                kept.append(line)
    return "".join(kept)
