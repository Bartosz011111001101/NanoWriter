"""Generowanie tekstu (ciągów tokenów) z modelu językowego."""

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from nanowriter.model import KVCache

if TYPE_CHECKING:
    from nanowriter.tokenizer import BPETokenizer, CharTokenizer


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


def _fit_user_text(
    tokenizer: "BPETokenizer | CharTokenizer", user_text: str, max_tokens: int
) -> list[int]:
    r"""Zwraca id kontekstu "A: <sufiks>\nB:" z najdłuższym sufiksem user_text, który się mieści.

    Szuka binarnie po długości sufiksu w znakach. Rzuca ValueError, gdy nawet pusty
    sufiks nie mieści się w max_tokens.
    Przykład: tok = BPETokenizer([]), user_text="abcdefghij", max_tokens=8
        -> tok.encode("A: ij\nB:")
    Przykład: max_tokens=5 -> ValueError
    """

    def ids_for(n: int) -> list[int]:
        return tokenizer.encode(f"A: {user_text[len(user_text) - n:]}\nB:")

    best = ids_for(0)
    if len(best) > max_tokens:
        raise ValueError(f"max_tokens={max_tokens} za mało nawet na samą linię użytkownika")
    lo, hi = 0, len(user_text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        ids = ids_for(mid)
        if len(ids) <= max_tokens:
            lo, best = mid, ids
        else:
            hi = mid - 1
    return best


def build_context(
    tokenizer: "BPETokenizer | CharTokenizer",
    history: list[tuple[str, str]],
    user_text: str,
    max_tokens: int,
    *,
    max_turns: int = 4,
) -> list[int]:
    r"""Buduje id tokenów kontekstu: ostatnie wymiany z historii i bieżąca linia użytkownika.

    Tekst kontekstu: dla każdej wymiany (u, r) linie "A: u\n" i "B: r\n", na końcu
    "A: <user_text>\nB:". Bierze najwyżej max_turns ostatnich wymian, potem gubi najstarsze,
    aż wynik ma <= max_tokens tokenów. Gdy nie mieści się nawet sama bieżąca wymiana,
    skraca user_text od początku (zostaje najdłuższy sufiks, w znakach, który się mieści).
    Przykład (tok = BPETokenizer([]), czyli 1 token na bajt):
        build_context(tok, [("x", "y")], "ab", 18) -> tok.encode("A: x\nB: y\nA: ab\nB:")
    Przykład: to samo z max_tokens=17 -> tok.encode("A: ab\nB:")
    Przykład: to samo z max_tokens=7 -> tok.encode("A: b\nB:")
    Przykład: to samo z max_tokens=6 -> tok.encode("A: \nB:")
    Przykład: to samo z max_tokens=5 -> ValueError
    Przykład: max_turns=0 -> historia ignorowana; history=[] -> tylko bieżąca wymiana
    Przykład: "\n" w user_text albo w tekście z historii -> ValueError; max_tokens < 1 albo
        max_turns < 0 -> ValueError
    """
    if max_tokens < 1:
        raise ValueError(f"max_tokens musi być >= 1, jest {max_tokens}")
    if max_turns < 0:
        raise ValueError(f"max_turns musi być >= 0, jest {max_turns}")
    if "\n" in user_text or any("\n" in u or "\n" in r for u, r in history):
        raise ValueError("tekst wypowiedzi nie może zawierać znaku nowej linii")
    turns = history[-max_turns:] if max_turns > 0 else []
    for k in range(len(turns), -1, -1):
        kept = turns[len(turns) - k :]
        text = "".join(f"A: {u}\nB: {r}\n" for u, r in kept) + f"A: {user_text}\nB:"
        ids = tokenizer.encode(text)
        if len(ids) <= max_tokens:
            return ids
    return _fit_user_text(tokenizer, user_text, max_tokens)


def extract_reply(text: str) -> str:
    r"""Zwraca odpowiedź: tekst do pierwszego "\n", bez spacji na brzegach i bez "�" na końcu.

    Przykład: extract_reply(" Dobrze, dzięki.\nA: a ty?") -> "Dobrze, dzięki."
    Przykład: extract_reply("  ok  ") -> "ok"
    Przykład: extract_reply("\nA: hej") -> ""
    Przykład: extract_reply(" kot�") -> "kot"
    Przykład: extract_reply("") -> ""
    """
    end = text.find("\n")
    if end != -1:
        text = text[:end]
    return text.strip().rstrip("�").rstrip()


def chat_reply(
    model: nn.Module,
    tokenizer: "BPETokenizer | CharTokenizer",
    history: list[tuple[str, str]],
    user_text: str,
    *,
    device: str | torch.device,
    block_size: int,
    max_new_tokens: int = 64,
    max_turns: int = 4,
    generator: torch.Generator | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    use_cache: bool = False,
) -> str:
    r"""Zwraca odpowiedź modelu na user_text, pamiętając ostatnie wymiany z history.

    Kontekst z build_context mieści się w block_size - max_new_tokens tokenach, więc cała
    odpowiedź mieści się w oknie modelu. Model generuje max_new_tokens tokenów (generate),
    a odpowiedzią jest tekst do pierwszego "\n" (extract_reply): nic po "\n" nie trafia do
    wyniku, więc model nie mówi za użytkownika. history nie jest zmieniana.
    Przykład: model, który po "B:" pisze " ok\nA: hej\nB: ok", daje "ok"
    Przykład: model, który po "B:" od razu pisze "\n", daje ""
    Przykład: max_new_tokens=0 albo block_size <= max_new_tokens -> ValueError
    """
    if max_new_tokens < 1 or block_size - max_new_tokens < 1:
        raise ValueError(
            "max_new_tokens musi być >= 1 i block_size > max_new_tokens, "
            f"jest {max_new_tokens} i {block_size}"
        )
    ids = build_context(
        tokenizer, history, user_text, block_size - max_new_tokens, max_turns=max_turns
    )
    idx = torch.tensor([ids], dtype=torch.int64, device=device)
    out = generate(
        model,
        idx,
        max_new_tokens,
        generator=generator,
        block_size=block_size,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        use_cache=use_cache,
    )
    return extract_reply(tokenizer.decode(out[0, len(ids) :].tolist()))
