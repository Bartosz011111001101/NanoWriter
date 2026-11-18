"""Generowanie tekstu (ciągów tokenów) z modelu językowego."""

import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def generate(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    *,
    generator: torch.Generator | None = None,
    block_size: int | None = None,
) -> torch.Tensor:
    """Dopisuje do idx max_new_tokens tokenów losowanych z softmaksu logitów ostatniej pozycji.

    model: forward(idx[B, T]) -> logits[B, T, V], na urządzeniu idx.
    idx: LongTensor[B, T], T >= 1. Nie jest zmieniany.
    generator: na urządzeniu idx; None -> domyślny generator PyTorch.
    block_size: None -> model dostaje cały ciąg; liczba -> tylko ostatnie block_size tokenów.
    Zwraca: nowy LongTensor[B, T + max_new_tokens] na urządzeniu idx, pierwsze T kolumn równe idx.
    Przykład: generate(nn.Embedding(5, 5), torch.tensor([[1, 2]]), 4)
        -> tensor o kształcie [1, 6], out[0, :2] == [1, 2], nowe id w [0, 5)
    Przykład: max_new_tokens=0 -> kopia idx (równa, ale nie ten sam obiekt)
    Przykład: max_new_tokens=-1 -> ValueError
    Przykład: idx o kształcie [1, 0] -> ValueError
    """
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens musi być >= 0, jest {max_new_tokens}")
    if idx.dim() != 2 or idx.shape[1] == 0:
        raise ValueError(f"idx musi mieć kształt [B, T] z T >= 1, jest {tuple(idx.shape)}")
    if block_size is not None and block_size < 1:
        raise ValueError(f"block_size musi być None albo >= 1, jest {block_size}")
    if max_new_tokens == 0:
        return idx.clone()

    seq = idx
    for _ in range(max_new_tokens):
        context = seq if block_size is None else seq[:, -block_size:]
        logits = model(context)[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1, generator=generator)
        seq = torch.cat([seq, next_tok], dim=1)
    return seq
