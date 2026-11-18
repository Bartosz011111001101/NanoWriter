"""Modele językowe NanoWriter. Etap 1: model bigramowy."""

import torch
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
