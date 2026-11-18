"""Losowanie batchy (x, y) z ciągu tokenów."""

import torch


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
