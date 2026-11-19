"""Generowanie tekstu (ciągów tokenów) z modelu językowego."""

import torch
import torch.nn.functional as F
from torch import nn

from nanowriter.model import KVCache


def _validate_sampling(temperature: float, top_k: int | None, top_p: float | None) -> None:
    """Rzuca ValueError dla niepoprawnych temperature, top_k, top_p."""
    if temperature < 0:
        raise ValueError(f"temperature musi być >= 0, jest {temperature}")
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k musi być None albo >= 1, jest {top_k}")
    if top_p is not None and not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p musi być None albo w (0, 1], jest {top_p}")


def filter_logits(
    logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None
) -> torch.Tensor:
    """Zwraca kopię logits[B, V] z -inf na pozycjach odrzuconych przez top-k, potem top-p.

    Przykład: filter_logits(torch.tensor([[1.0, 3.0, 2.0, 0.0]]), top_k=2)
        -> [[-inf, 3.0, 2.0, -inf]]
    Przykład: logits = torch.tensor([0.5, 0.3, 0.2]).log()[None]:
        top_p=0.4 -> zostaje tylko pozycja 0; top_p=0.6 -> pozycje 0 i 1;
        top_p=0.9 -> wszystkie trzy
    Przykład: top_k=None, top_p=None -> kopia bez zmian
    Przykład: top_k=0 -> ValueError; top_p=0.0 albo 1.5 -> ValueError
    """
    if logits.dim() != 2:
        raise ValueError(f"logits musi mieć kształt [B, V], jest {tuple(logits.shape)}")
    _validate_sampling(1.0, top_k, top_p)
    out = logits.clone()
    vocab = out.shape[-1]
    if top_k is not None and top_k < vocab:
        kth = torch.topk(out, top_k, dim=-1).values[:, -1:]
        out = out.masked_fill(out < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, order = torch.sort(out, dim=-1, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        before = torch.cumsum(probs, dim=-1) - probs
        drop_sorted = before >= top_p
        drop_sorted[:, 0] = False
        drop = torch.zeros_like(drop_sorted).scatter(-1, order, drop_sorted)
        out = out.masked_fill(drop, float("-inf"))
    return out


def _pick_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Wybiera jeden token na wiersz z logits[B, V]; zwraca LongTensor[B, 1]."""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    filtered = filter_logits(logits / temperature, top_k, top_p)
    probs = F.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.no_grad()
def generate(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    *,
    generator: torch.Generator | None = None,
    block_size: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    use_cache: bool = False,
) -> torch.Tensor:
    """Dopisuje do idx max_new_tokens tokenów losowanych z logitów ostatniej pozycji.

    model: forward(idx[B, T]) -> logits[B, T, V], na urządzeniu idx;
        przy use_cache=True przyjmuje też cache=KVCache (np. GPT).
    idx: LongTensor[B, T], T >= 1. Nie jest zmieniany.
    generator: na urządzeniu idx; None -> domyślny generator PyTorch.
    block_size: None -> model dostaje cały ciąg; liczba -> tylko ostatnie block_size tokenów.
    temperature: logity / temperature przed filtrami; 0 -> zawsze najwyższy logit (argmax).
    top_k, top_p: filtry z filter_logits; None -> bez filtra.
    use_cache: KV cache; wynik identyczny jak bez cache przy tym samym seedzie.
    Zwraca: nowy LongTensor[B, T + max_new_tokens] na urządzeniu idx,
        pierwsze T kolumn równe idx.
    Przykład: generate(nn.Embedding(5, 5), torch.tensor([[1, 2]]), 4)
        -> tensor o kształcie [1, 6], out[0, :2] == [1, 2], nowe id w [0, 5)
    Przykład: max_new_tokens=0 -> kopia idx (równa, ale nie ten sam obiekt)
    Przykład: max_new_tokens=-1 -> ValueError
    Przykład: idx o kształcie [1, 0] -> ValueError
    Przykład: temperature=-0.5 -> ValueError; top_k=0 -> ValueError
    """
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens musi być >= 0, jest {max_new_tokens}")
    if idx.dim() != 2 or idx.shape[1] == 0:
        raise ValueError(f"idx musi mieć kształt [B, T] z T >= 1, jest {tuple(idx.shape)}")
    if block_size is not None and block_size < 1:
        raise ValueError(f"block_size musi być None albo >= 1, jest {block_size}")
    _validate_sampling(temperature, top_k, top_p)
    if max_new_tokens == 0:
        return idx.clone()

    cache = KVCache() if use_cache else None
    seq = idx
    for step in range(max_new_tokens):
        if cache is None:
            context = seq if block_size is None else seq[:, -block_size:]
            logits = model(context)[:, -1, :]
        else:
            if step == 0:
                context = seq if block_size is None else seq[:, -block_size:]
            elif block_size is not None and cache.length + 1 > block_size:
                cache.reset()
                context = seq[:, -block_size:]
            else:
                context = seq[:, -1:]
            logits = model(context, cache=cache)[:, -1, :]
        next_tok = _pick_token(logits, temperature, top_k, top_p, generator)
        seq = torch.cat([seq, next_tok], dim=1)
    return seq
