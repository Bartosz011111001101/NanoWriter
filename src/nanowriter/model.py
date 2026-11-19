"""Modele językowe NanoWriter: bigram (etap 1) i transformer GPT (etap 4)."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class BigramLanguageModel(nn.Module):
    """Model bigramowy: logity następnego tokenu to wiersz tablicy dla bieżącego tokenu.

    Przykład: BigramLanguageModel(102) -> 10404 parametry (102 * 102)
    """

    def __init__(self, vocab_size: int) -> None:
        """Tworzy tablicę logitów vocab_size x vocab_size z wagami N(0, 0.02).

        Przykład: BigramLanguageModel(65).token_embedding.weight.shape -> torch.Size([65, 65])
        Przykład: BigramLanguageModel(0) -> ValueError
        """
        if vocab_size < 1:
            raise ValueError(f"vocab_size musi być >= 1, otrzymano {vocab_size}")
        super().__init__()
        self.vocab_size: int = vocab_size
        self.token_embedding: nn.Embedding = nn.Embedding(vocab_size, vocab_size)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Zwraca logity następnego tokenu dla każdej pozycji.

        idx: LongTensor[B, T], wartości 0 <= id < vocab_size.
        Zwraca: FloatTensor[B, T, vocab_size] (float32).
        Przykład: idx o kształcie [2, 5] przy vocab_size=102 -> logity o kształcie [2, 5, 102]
        Przykład: idx o kształcie [1, 1] -> logity o kształcie [1, 1, vocab_size]
        """
        return self.token_embedding(idx)


@dataclass(frozen=True)
class GPTConfig:
    """Rozmiar transformera.

    Przykład: GPTConfig() -> vocab_size=1024, block_size=256, n_layer=4, n_head=4, d_model=128
    Przykład: GPTConfig(d_model=130, n_head=4) -> ValueError (130 nie dzieli się przez 4)
    Przykład: GPTConfig(n_layer=0) -> ValueError
    """

    vocab_size: int = 1024
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128

    def __post_init__(self) -> None:
        """Sprawdza, że każde pole >= 1 i d_model dzieli się przez n_head.

        Przykład: GPTConfig(vocab_size=1) -> poprawny
        Przykład: GPTConfig(block_size=0) -> ValueError z "block_size" w komunikacie
        """
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "d_model"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} musi być >= 1, otrzymano {value}")
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) musi dzielić się przez n_head ({self.n_head})"
            )


class KVCache:
    """Pamięć kluczy i wartości uwagi dla kolejnych warstw (KV cache) jednego generowania.

    Przykład: c = KVCache(); c.update(0, k[1, 2, 3, 4], v[1, 2, 3, 4]) -> (k, v); c.length -> 3
    """

    def __init__(self) -> None:
        """Tworzy pustą pamięć.

        Przykład: KVCache().length -> 0
        """
        self._keys: list[torch.Tensor] = []
        self._values: list[torch.Tensor] = []

    @property
    def length(self) -> int:
        """Zwraca liczbę zapamiętanych pozycji (warstwy 0); 0, gdy pusta.

        Przykład: po update(0, k[1, 2, 3, 4], ...) i update(0, k[1, 2, 1, 4], ...) -> 4
        """
        if not self._keys:
            return 0
        return int(self._keys[0].shape[2])

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dopisuje k, v [B, n_head, T_new, head_dim] do warstwy layer; zwraca jej pełne k, v.

        Przykład: pusta c, c.update(0, k[1, 2, 3, 4], v[1, 2, 3, 4])
            -> k, v o kształcie [1, 2, 3, 4]
        Przykład: potem c.update(0, k2[1, 2, 1, 4], v2[1, 2, 1, 4]) -> kształt [1, 2, 4, 4]
        Przykład: pusta c, c.update(1, k, v) -> ValueError (brak warstwy 0)
        """
        stored = len(self._keys)
        if layer < 0 or layer > stored:
            raise ValueError(f"layer musi być w [0, {stored}], otrzymano {layer}")
        if layer == stored:
            self._keys.append(k)
            self._values.append(v)
        else:
            self._keys[layer] = torch.cat([self._keys[layer], k], dim=2)
            self._values[layer] = torch.cat([self._values[layer], v], dim=2)
        return self._keys[layer], self._values[layer]

    def reset(self) -> None:
        """Czyści pamięć wszystkich warstw.

        Przykład: c.reset(); c.length -> 0
        """
        self._keys = []
        self._values = []


class CausalSelfAttention(nn.Module):
    """Wielogłowicowa uwaga z maską przyczynową: pozycja t widzi tylko pozycje 0..t.

    Przykład: CausalSelfAttention(GPTConfig(d_model=32, n_head=2, block_size=16))(x[2, 5, 32])
        -> tensor [2, 5, 32]
    """

    def __init__(self, config: GPTConfig) -> None:
        """Tworzy qkv (d_model -> 3*d_model), proj (d_model -> d_model) i bufor maski.

        Przykład: CausalSelfAttention(GPTConfig()).head_dim -> 32
        """
        super().__init__()
        self.n_head: int = config.n_head
        self.head_dim: int = config.d_model // config.n_head
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=True)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=True)
        mask = torch.tril(torch.ones(config.block_size, config.block_size)).bool()
        self.register_buffer("mask", mask, persistent=False)

    def forward(
        self, x: torch.Tensor, cache: KVCache | None = None, layer: int = 0
    ) -> torch.Tensor:
        """Zwraca wynik uwagi dla x: FloatTensor[B, T, d_model] -> FloatTensor[B, T, d_model].

        cache: k, v nowych pozycji dopisane do warstwy layer; nowa pozycja start + t widzi
        klucze 0..start + t, gdzie start = liczba pozycji w cache tej warstwy przed dopisaniem.
        Przykład: x[1, 1, d_model] -> wynik [1, 1, d_model] (uwaga tylko na samą siebie)
        """
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=-1)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        if cache is not None:
            k, v = cache.update(layer, k, v)
        t_all = k.shape[2]
        start = t_all - t

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~self.mask[start : start + t, :t_all], float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class MLP(nn.Module):
    """Sieć feed-forward: d_model -> 4*d_model -> GELU -> d_model.

    Przykład: MLP(GPTConfig(d_model=32))(x[2, 5, 32]) -> tensor [2, 5, 32]
    """

    def __init__(self, config: GPTConfig) -> None:
        """Tworzy fc, act (dokładny GELU) i proj.

        Przykład: MLP(GPTConfig()).fc.out_features -> 512
        """
        super().__init__()
        self.fc = nn.Linear(config.d_model, 4 * config.d_model, bias=True)
        self.act = nn.GELU(approximate="none")
        self.proj = nn.Linear(4 * config.d_model, config.d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Zwraca proj(act(fc(x))) o kształcie x.

        Przykład: x[2, 5, 128] -> wynik [2, 5, 128]
        """
        return self.proj(self.act(self.fc(x)))


class Block(nn.Module):
    """Blok transformera w układzie pre-norm: x + attn(ln1(x)), potem x + mlp(ln2(x)).

    Przykład: Block(GPTConfig(d_model=32, n_head=2, block_size=16))(x[2, 5, 32]) -> [2, 5, 32]
    """

    def __init__(self, config: GPTConfig) -> None:
        """Tworzy ln1, attn, ln2, mlp.

        Przykład: Block(GPTConfig()).ln1.normalized_shape -> (128,)
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(
        self, x: torch.Tensor, cache: KVCache | None = None, layer: int = 0
    ) -> torch.Tensor:
        """Zwraca wynik bloku o kształcie x; cache i layer przekazane do attn.

        Przykład: x[1, 3, 128] -> wynik [1, 3, 128]
        """
        x = x + self.attn(self.ln1(x), cache, layer)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Decoder-only transformer: embedding tokenów i pozycji, n_layer bloków, ln_f, lm_head.

    Wagi lm_head są wspólne z tok_emb (weight tying).
    Przykład: GPT(GPTConfig()) -> 957184 parametrów
    Przykład: GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, d_model=32))
        -> 28032 parametrów
    """

    def __init__(self, config: GPTConfig) -> None:
        """Buduje warstwy, wiąże lm_head.weight z tok_emb.weight i inicjalizuje wagi.

        Przykład: m = GPT(GPTConfig()); m.lm_head.weight is m.tok_emb.weight -> True
        """
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.block_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        resid_std = 0.02 / math.sqrt(2 * config.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=resid_std)
            nn.init.normal_(block.mlp.proj.weight, mean=0.0, std=resid_std)

    def forward(self, idx: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        """Zwraca logity następnego tokenu dla każdej pozycji.

        idx: LongTensor[B, T], wartości 0 <= id < vocab_size; z cache tylko nowe tokeny,
        na pozycjach cache.length .. cache.length + T - 1 (razem najwyżej block_size).
        Zwraca: FloatTensor[B, T, vocab_size] (float32).
        Przykład: idx [2, 16] przy GPTConfig(vocab_size=64, block_size=16, ...)
            -> logity [2, 16, 64]
        Przykład: c = KVCache(); model(idx[:, :5], cache=c) i model(idx[:, 5:], cache=c)
            -> sklejone logity == model(idx) (atol=1e-5), c.length == T
        Przykład: idx [1, 17] przy block_size=16 -> ValueError
        Przykład: c.length == 16, model(idx[1, 1], cache=c) przy block_size=16 -> ValueError
        Przykład: idx [1, 0] -> ValueError; idx [5] (1 wymiar) -> ValueError
        """
        if idx.dim() != 2:
            raise ValueError(f"idx musi mieć 2 wymiary [B, T], otrzymano {idx.dim()}")
        t = idx.shape[1]
        if t == 0:
            raise ValueError("T musi być >= 1, otrzymano 0")
        start = cache.length if cache is not None else 0
        if start + t > self.config.block_size:
            raise ValueError(
                f"pozycje cache ({start}) + T ({t}) przekraczają block_size "
                f"({self.config.block_size})"
            )

        pos = torch.arange(start, start + t, dtype=torch.long, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for i, block in enumerate(self.blocks):
            x = block(x, cache, i)
        return self.lm_head(self.ln_f(x))
