"""Tokenizer znakowy: jeden znak Unicode to jeden token."""


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
