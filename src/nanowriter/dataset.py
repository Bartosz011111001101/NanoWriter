"""Batche (x, y) z ciągu tokenów oraz pliki .bin z tokenami."""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from nanowriter.data import load_text

if TYPE_CHECKING:
    from nanowriter.tokenizer import BPETokenizer, CharTokenizer


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
    if batch_size < 1 or block_size < 1:
        raise ValueError(
            "batch_size i block_size muszą być >= 1, "
            f"jest batch_size={batch_size}, block_size={block_size}"
        )
    n = data.shape[0]
    if n <= block_size:
        raise ValueError(f"za krótkie data: N={n} <= block_size={block_size}")

    starts = torch.randint(0, n - block_size, (batch_size,), generator=generator).tolist()
    x_rows = [data[s : s + block_size].to(torch.int64) for s in starts]
    y_rows = [data[s + 1 : s + block_size + 1].to(torch.int64) for s in starts]
    x = torch.stack(x_rows).to(device)
    y = torch.stack(y_rows).to(device)
    return x, y


TOKEN_DTYPE: str = "uint16"
MAX_TOKEN_ID: int = 65535


def write_tokens(ids: list[int], path: Path) -> int:
    """Zapisuje id tokenów jako uint16 little-endian (bez nagłówka) i zwraca ich liczbę.

    Przykład: write_tokens([1, 2, 1023], p) -> 3, plik o bajtach b"\\x01\\x00\\x02\\x00\\xff\\x03"
    Przykład: write_tokens([], p) -> 0, pusty plik
    Przykład: write_tokens([65536], p) -> ValueError, plik nie powstaje
    Przykład: write_tokens([-1], p) -> ValueError, plik nie powstaje
    """
    arr = np.asarray(ids, dtype=np.int64)
    if arr.size and (arr.min() < 0 or arr.max() > MAX_TOKEN_ID):
        raise ValueError(f"Id tokenów poza zakresem 0..{MAX_TOKEN_ID}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(arr.astype("<u2").tobytes())
    return int(arr.size)


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


def encode_file(
    tokenizer: "BPETokenizer | CharTokenizer", text_path: Path, out_path: Path
) -> tuple[int, int]:
    """Koduje plik tekstowy tokenizerem, zapisuje tokeny do out_path i zwraca (tokeny, znaki).

    Przykład: plik "cab", tokenizer CharTokenizer.from_text("abca")
        -> (3, 3), load_tokens(out_path) -> tensor([2, 0, 1])
    Przykład: tokenizer z vocab_size 70000 -> ValueError, nic nie zapisane
    """
    if tokenizer.vocab_size > MAX_TOKEN_ID + 1:
        raise ValueError(f"Słownik za duży dla {TOKEN_DTYPE}: {tokenizer.vocab_size}")

    text = load_text(text_path)
    ids = tokenizer.encode(text)
    write_tokens(ids, out_path)
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
