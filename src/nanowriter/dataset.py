"""Batche (x, y) z ciągu tokenów oraz pliki .bin z tokenami."""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import numpy as np
import torch

from nanowriter.data import load_text

if TYPE_CHECKING:
    from nanowriter.tokenizer import BPETokenizer, CharTokenizer


def _random_starts(
    n: int, batch_size: int, block_size: int, generator: torch.Generator
) -> list[int]:
    """Sprawdza rozmiary i losuje batch_size startów okien z [0, n - block_size).

    Przykład: _random_starts(10, 2, 4, torch.Generator().manual_seed(0)) -> 2 liczby z [0, 6)
    Przykład: _random_starts(4, 1, 4, g) -> ValueError (n <= block_size)
    Przykład: _random_starts(10, 0, 4, g) -> ValueError (batch_size < 1)
    """
    if batch_size < 1 or block_size < 1:
        raise ValueError(
            "batch_size i block_size muszą być >= 1, "
            f"jest batch_size={batch_size}, block_size={block_size}"
        )
    if n <= block_size:
        raise ValueError(f"za krótkie data: N={n} <= block_size={block_size}")
    return torch.randint(0, n - block_size, (batch_size,), generator=generator).tolist()


def _windows(
    source: torch.Tensor, starts: list[int], block_size: int, offset: int
) -> torch.Tensor:
    """Zwraca tensor [len(starts), block_size] z oknami źródła przesuniętymi o offset.

    Wiersz i to source[s + offset : s + offset + block_size] dla s = starts[i].
    Przykład: _windows(torch.arange(10), [0, 3], 4, 1) -> [[1, 2, 3, 4], [4, 5, 6, 7]]
    """
    return torch.stack([source[s + offset : s + offset + block_size] for s in starts])


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    generator: torch.Generator,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Losuje batch_size fragmentów długości block_size; y to x przesunięte o jeden token.

    data: tensor 1D id tokenów (dowolny całkowity dtype), długość N > block_size.
    generator: torch.Generator na CPU.
    Zwraca: (x, y), oba LongTensor[batch_size, block_size] na device.
    Przykład: get_batch(torch.arange(10), 2, 4, torch.Generator().manual_seed(0))
        -> x o kształcie [2, 4], y == x + 1, każdy wiersz x to kolejne liczby od s do s+3,
        0 <= s <= 5
    Przykład: get_batch(torch.arange(5), 3, 4, g) -> x == [[0, 1, 2, 3]] * 3,
        y == [[1, 2, 3, 4]] * 3
    Przykład: get_batch(torch.arange(4), 1, 4, g) -> ValueError (N <= block_size)
    """
    if data.dim() != 1:
        raise ValueError(f"data musi być jednowymiarowy, ma {data.dim()} wymiary")
    starts = _random_starts(data.shape[0], batch_size, block_size, generator)
    x = _windows(data, starts, block_size, 0).to(torch.int64).to(device)
    y = _windows(data, starts, block_size, 1).to(torch.int64).to(device)
    return x, y


def get_masked_batch(
    data: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int,
    block_size: int,
    generator: torch.Generator,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Jak get_batch, a do tego maska straty m: m[b, t] to maska tokenu y[b, t].

    data: tensor 1D id tokenów; mask: tensor 1D tego samego kształtu (bool albo 0/1),
    mask[i] == 1 znaczy, że token i jest celem liczonym w stracie.
    Zwraca: (x, y, m): x, y LongTensor[B, T], m BoolTensor[B, T], wszystkie na device;
    m[b] = mask[s + 1 : s + T + 1] dla tego samego s, z którego powstały x[b] i y[b].
    Przykład: data = torch.arange(10), mask = (data % 2 == 0),
        get_masked_batch(data, mask, 2, 4, g) -> x, y jak get_batch(data, 2, 4, g tego samego
        seeda), m == (y % 2 == 0)
    Przykład: mask o innym kształcie niż data -> ValueError
    """
    if data.dim() != 1:
        raise ValueError(f"data musi być jednowymiarowy, ma {data.dim()} wymiary")
    if mask.shape != data.shape:
        raise ValueError(
            f"mask musi mieć kształt data: mask={tuple(mask.shape)}, data={tuple(data.shape)}"
        )
    starts = _random_starts(data.shape[0], batch_size, block_size, generator)
    x = _windows(data, starts, block_size, 0).to(torch.int64).to(device)
    y = _windows(data, starts, block_size, 1).to(torch.int64).to(device)
    m = _windows(mask, starts, block_size, 1).to(torch.bool).to(device)
    return x, y, m


TOKEN_DTYPE: str = "uint16"
MAX_TOKEN_ID: int = 65535


def _tokens_to_bytes(ids: list[int]) -> bytes:
    """Sprawdza zakres id (0..MAX_TOKEN_ID) i zwraca je jako uint16 little-endian.

    Przykład: _tokens_to_bytes([1, 2, 1023]) -> b"\\x01\\x00\\x02\\x00\\xff\\x03"
    Przykład: _tokens_to_bytes([]) -> b""
    Przykład: _tokens_to_bytes([65536]) -> ValueError
    Przykład: _tokens_to_bytes([-1]) -> ValueError
    """
    arr = np.asarray(ids, dtype=np.int64)
    if arr.size and (arr.min() < 0 or arr.max() > MAX_TOKEN_ID):
        raise ValueError(f"Id tokenów poza zakresem 0..{MAX_TOKEN_ID}")
    return arr.astype("<u2").tobytes()


def write_tokens(ids: list[int], path: Path) -> int:
    """Zapisuje id tokenów jako uint16 little-endian (bez nagłówka) i zwraca ich liczbę.

    Przykład: write_tokens([1, 2, 1023], p) -> 3, plik o bajtach b"\\x01\\x00\\x02\\x00\\xff\\x03"
    Przykład: write_tokens([], p) -> 0, pusty plik
    Przykład: write_tokens([65536], p) -> ValueError, plik nie powstaje
    Przykład: write_tokens([-1], p) -> ValueError, plik nie powstaje
    """
    data = _tokens_to_bytes(ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data) // 2


def load_tokens(path: Path) -> torch.Tensor:
    """Wczytuje plik .bin i zwraca tensor 1D int64 (CPU) z id tokenów.

    Przykład: load_tokens(p) po write_tokens([1, 2, 1023], p) -> tensor([1, 2, 1023]), int64
    Przykład: pusty plik -> tensor o kształcie [0], int64
    Przykład: plik o 3 bajtach -> ValueError
    """
    raw = path.read_bytes()
    if len(raw) % 2 != 0:
        raise ValueError(f"Plik tokenów ma nieparzystą liczbę bajtów: {len(raw)}")
    if not raw:
        return torch.zeros(0, dtype=torch.int64)

    # astype tworzy kopię, więc tensor nie wskazuje na bufor tylko do odczytu
    ids = np.frombuffer(raw, dtype="<u2").astype(np.int64)
    return torch.from_numpy(ids)


MASK_DTYPE: str = "uint8"


def write_mask(mask: list[int], path: Path) -> int:
    """Zapisuje maskę jako uint8 (1 bajt na token, bez nagłówka) i zwraca jej długość.

    Przykład: write_mask([0, 1, 1], p) -> 3, plik o bajtach b"\\x00\\x01\\x01"
    Przykład: write_mask([], p) -> 0, pusty plik
    Przykład: write_mask([0, 2], p) -> ValueError, plik nie powstaje
    Przykład: write_mask([-1], p) -> ValueError, plik nie powstaje
    """
    arr = np.asarray(mask, dtype=np.int64)
    if arr.size and (arr.min() < 0 or arr.max() > 1):
        raise ValueError("Wartości maski muszą być równe 0 albo 1")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(arr.astype(MASK_DTYPE).tobytes())
    return int(arr.size)


def load_mask(path: Path) -> torch.Tensor:
    """Wczytuje plik maski i zwraca tensor 1D bool (CPU), jedna wartość na bajt.

    Przykład: load_mask(p) po write_mask([0, 1, 1], p) -> tensor([False, True, True])
    Przykład: pusty plik -> tensor o kształcie [0], dtype bool
    Przykład: plik z bajtem 2 -> ValueError
    """
    raw = path.read_bytes()
    if not raw:
        return torch.zeros(0, dtype=torch.bool)

    # astype tworzy kopię, więc tensor nie wskazuje na bufor tylko do odczytu
    values = np.frombuffer(raw, dtype=MASK_DTYPE).astype(np.uint8)
    if values.max() > 1:
        raise ValueError(f"Plik maski zawiera wartości inne niż 0 i 1: {path}")
    return torch.from_numpy(values.astype(np.bool_))


def encode_file(
    tokenizer: "BPETokenizer | CharTokenizer",
    text_path: Path,
    out_path: Path,
    block_chars: int | None = None,
) -> tuple[int, int]:
    """Koduje plik tekstowy tokenizerem, zapisuje tokeny do out_path i zwraca (tokeny, znaki).

    block_chars None: cały plik jednym wywołaniem encode (jak dotąd).
    block_chars N: plik czytany liniami, kodowany blokami po co najmniej N znaków (blok kończy
    się na końcu linii), każdy blok dopisywany od razu do out_path; wynik taki sam jak bez N.
    Przykład: plik "ab\\ncd\\n", tokenizer CharTokenizer.from_text("abcd\\n"), block_chars=3
        -> (6, 6), load_tokens(out_path) -> tensor([1, 2, 0, 3, 4, 0])
    Przykład: pusty plik, block_chars=10 -> (0, 0), pusty out_path
    Przykład: block_chars=0 -> ValueError, nic nie zapisane
    Przykład: tokenizer z vocab_size 70000 -> ValueError, nic nie zapisane
    """
    if tokenizer.vocab_size > MAX_TOKEN_ID + 1:
        raise ValueError(f"Słownik za duży dla {TOKEN_DTYPE}: {tokenizer.vocab_size}")

    if block_chars is None:
        text = load_text(text_path)
        ids = tokenizer.encode(text)
        write_tokens(ids, out_path)
        return len(ids), len(text)

    if block_chars < 1:
        raise ValueError(f"block_chars musi być >= 1, jest {block_chars}")

    total_tokens = 0
    total_chars = 0
    lines: list[str] = []
    lines_chars = 0

    def flush(out_file: BinaryIO) -> None:
        nonlocal total_tokens, total_chars, lines_chars
        if not lines:
            return
        ids = tokenizer.encode("".join(lines))
        out_file.write(_tokens_to_bytes(ids))
        total_tokens += len(ids)
        total_chars += lines_chars
        lines.clear()
        lines_chars = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        open(text_path, encoding="utf-8") as text_file,
        open(out_path, "wb") as out_file,
    ):
        for line in text_file:
            lines.append(line)
            lines_chars += len(line)
            if lines_chars >= block_chars:
                flush(out_file)
        flush(out_file)
    return total_tokens, total_chars


_USER_PREFIX: str = "A: "
_MODEL_PREFIX: str = "B: "
_MODEL_LABEL: str = "B:"


def encode_dialogs(
    tokenizer: "BPETokenizer | CharTokenizer", text: str
) -> tuple[list[int], list[int]]:
    """Koduje strumień rozmów linia po linii i zwraca (ids, mask) tej samej długości.

    Linia "A: ..." i pusta linia (separator rozmów): maska 0 na wszystkich tokenach.
    Linia "B: ...": maska 0 na tokenach etykiety "B:", maska 1 na reszcie linii z końcowym "\\n".
    Przykład: BPETokenizer([]), "A: hej\\nB: ok\\n"
        -> ids [65, 58, 32, 104, 101, 106, 10, 66, 58, 32, 111, 107, 10],
           mask [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    Przykład: BPETokenizer([]), "B: a\\n\\nA: b\\n"
        -> ids [66, 58, 32, 97, 10, 10, 65, 58, 32, 98, 10],
           mask [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    Przykład: "B: ok" (bez końcowego "\\n") -> to samo co "B: ok\\n"
    Przykład: "" -> ([], [])
    Przykład: "C: x\\n" -> ValueError (linia 1); "B:ok\\n" -> ValueError; "A:\\n" -> ValueError
    """
    if not text:
        return [], []

    lines = text.split("\n")
    if text.endswith("\n"):
        lines.pop()

    ids: list[int] = []
    mask: list[int] = []
    label_ids: list[int] | None = None
    for number, line in enumerate(lines, start=1):
        line_ids = tokenizer.encode(line + "\n")
        if line == "" or line.startswith(_USER_PREFIX):
            line_mask = [0] * len(line_ids)
        elif line.startswith(_MODEL_PREFIX):
            if label_ids is None:
                label_ids = tokenizer.encode(_MODEL_LABEL)
            if line_ids[: len(label_ids)] != label_ids:
                raise ValueError(
                    f"Etykieta {_MODEL_LABEL} scaliła się z tekstem w linii {number}: "
                    f"{line[:40]!r}"
                )
            line_mask = [0] * len(label_ids) + [1] * (len(line_ids) - len(label_ids))
        else:
            raise ValueError(f"Zła linia {number} (oczekiwano A: albo B:): {line[:40]!r}")
        ids.extend(int(token) for token in line_ids)
        mask.extend(line_mask)
    return ids, mask


def encode_dialogs_file(
    tokenizer: "BPETokenizer | CharTokenizer", text_path: Path, out_path: Path, mask_path: Path
) -> tuple[int, int]:
    """Koduje plik z rozmowami: tokeny do out_path, maska do mask_path; zwraca (tokeny, znaki).

    Przykład: plik "B: ok\\n", BPETokenizer([]) -> (6, 6), out_path 6 tokenów,
        load_mask(mask_path) -> tensor([False, False, True, True, True, True])
    Przykład: tokenizer z vocab_size 70000 albo zła linia w pliku -> ValueError, żaden plik
        nie powstaje
    """
    if tokenizer.vocab_size > MAX_TOKEN_ID + 1:
        raise ValueError(f"Słownik za duży dla {TOKEN_DTYPE}: {tokenizer.vocab_size}")

    text = load_text(text_path)
    ids, mask = encode_dialogs(tokenizer, text)
    write_tokens(ids, out_path)
    write_mask(mask, mask_path)
    return len(ids), len(text)


@dataclass(frozen=True)
class TokenMeta:
    """Opis plików tokenów: rozmiar słownika, dtype i liczby tokenów oraz znaków train/val.

    Przykład: TokenMeta(1024, "uint16", 10, 25, 2, 5).train_chars -> 25
    """

    vocab_size: int
    dtype: str
    train_tokens: int
    train_chars: int
    val_tokens: int
    val_chars: int


def save_meta(meta: TokenMeta, path: Path) -> None:
    """Zapisuje TokenMeta jako JSON (UTF-8, wcięcie 2, klucze w kolejności pól).

    Przykład: save_meta(TokenMeta(1024, "uint16", 10, 25, 2, 5), p) -> plik jak w prompcie
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(asdict(meta), indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def load_meta(path: Path) -> TokenMeta:
    """Wczytuje TokenMeta zapisane przez save_meta.

    Przykład: load_meta(p) po save_meta(m, p) -> m
    Przykład: brak klucza "val_chars" -> ValueError
    Przykład: "dtype": "int32" -> ValueError
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Plik meta nie zawiera obiektu JSON: {path}")

    names = [field.name for field in fields(TokenMeta)]
    missing = [name for name in names if name not in data]
    if missing:
        raise ValueError(f"Brak pól w pliku meta: {missing}")
    if data["dtype"] != TOKEN_DTYPE:
        raise ValueError(f"Nieobsługiwany dtype tokenów: {data['dtype']!r}")
    return TokenMeta(**{name: data[name] for name in names})
