"""Pętla treningu modeli językowych."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from nanowriter.dataset import get_batch


@dataclass(frozen=True)
class TrainConfig:
    """Ustawienia treningu.

    Przykład: TrainConfig() -> batch_size=32, block_size=8, max_steps=5000,
        learning_rate=0.01, seed=0, log_every=500
    Przykład: TrainConfig(batch_size=0) -> ValueError
    """

    batch_size: int = 32
    block_size: int = 8
    max_steps: int = 5000
    learning_rate: float = 1e-2
    seed: int = 0
    log_every: int = 500

    def __post_init__(self) -> None:
        """Sprawdza zakresy pól.

        Przykład: TrainConfig(max_steps=0) -> poprawny
        Przykład: TrainConfig(learning_rate=0.0) -> ValueError
        """
        if self.batch_size < 1:
            raise ValueError(f"batch_size musi być >= 1, jest {self.batch_size}")
        if self.block_size < 1:
            raise ValueError(f"block_size musi być >= 1, jest {self.block_size}")
        if self.max_steps < 0:
            raise ValueError(f"max_steps musi być >= 0, jest {self.max_steps}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate musi być > 0, jest {self.learning_rate}")
        if self.log_every < 1:
            raise ValueError(f"log_every musi być >= 1, jest {self.log_every}")


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Zwraca średnią entropię krzyżową po wszystkich pozycjach batcha.

    logits: FloatTensor[B, T, V], targets: LongTensor[B, T].
    Zwraca: skalar (tensor o kształcie []) z gradientem względem logits.
    Przykład: compute_loss(torch.zeros(1, 2, 4), torch.tensor([[0, 3]])) -> tensor(1.3863) (ln 4)
    Przykład: logits [1, 2, 4] i targets [1, 3] -> ValueError
    Przykład: logits [2, 4] (2 wymiary) -> ValueError
    """
    if logits.dim() != 3:
        raise ValueError(f"logits musi mieć 3 wymiary [B, T, V], ma {logits.dim()}")
    if targets.shape != logits.shape[:2]:
        raise ValueError(
            f"targets {tuple(targets.shape)} nie pasuje do logits {tuple(logits.shape)} "
            "(oczekiwane [B, T])"
        )
    vocab = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))


def train_step(
    model: nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor
) -> float:
    """Wykonuje jeden krok uczenia na batchu (x, y) i zwraca stratę sprzed aktualizacji wag.

    x, y: LongTensor[B, T] na urządzeniu modelu.
    Przykład: model nn.Embedding(4, 4) z wagami zerowymi, x = [[0, 1]], y = [[1, 2]]
        -> 1.3862943... (ln 4), wagi po kroku inne niż zera
    """
    optimizer.zero_grad()
    logits = model(x)
    loss = compute_loss(logits, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def train(
    model: nn.Module,
    data: torch.Tensor,
    config: TrainConfig,
    device: str | torch.device,
    on_log: Callable[[int, float], None] | None = None,
) -> list[float]:
    """Trenuje model przez config.max_steps kroków na losowych batchach z data
    i zwraca stratę każdego kroku.

    model: już na device; data: tensor 1D id tokenów na CPU.
    on_log(step, loss): wołana dla step % log_every == 0 i dla ostatniego kroku (kroki od 0).
    Przykład: max_steps=5, log_every=2 -> lista 5 strat, on_log wołana dla kroków 0, 2, 4
    Przykład: max_steps=4, log_every=2 -> on_log wołana dla kroków 0, 2, 3
    Przykład: max_steps=0 -> [], on_log nie wołana
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    losses: list[float] = []
    for step in range(config.max_steps):
        x, y = get_batch(data, config.batch_size, config.block_size, generator, device)
        loss = train_step(model, optimizer, x, y)
        losses.append(loss)
        if on_log is not None and (
            step % config.log_every == 0 or step == config.max_steps - 1
        ):
            on_log(step, loss)
    return losses
