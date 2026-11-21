"""Krótkie treningi od zera do pokazania uczenia modelu (kilkadziesiąt sekund)."""

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from nanowriter.model import GPT, GPTConfig
from nanowriter.train import TrainConfig, train


@dataclass(frozen=True, eq=False)
class WynikTreningu:
    """Wynik krótkiego treningu.

    Przykład: kroki=4 -> kroki == [0, 1, 2, 3], len(train_loss) == 4
    """

    model: GPT
    kroki: list[int]
    train_loss: list[float]
    val_kroki: list[int]
    val_loss: list[float]
    lr: float
    czas_s: float


def nowy_model(konfig: GPTConfig, seed: int, device: str) -> GPT:
    """Tworzy GPT z wagami wylosowanymi przy zadanym seedzie i przenosi go na device.

    Przykład: nowy_model(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2,
        d_model=32), 0, "cpu") -> model z 28032 parametrami
    Przykład: dwa wywołania z tym samym seedem -> identyczne wagi
    """
    torch.manual_seed(seed)
    model = GPT(konfig)
    return model.to(device)


def krotki_trening(
    model: GPT,
    dane_train: torch.Tensor,
    dane_val: torch.Tensor,
    *,
    kroki: int,
    lr: float,
    batch_size: int,
    device: str,
    seed: int,
    co_ile: int,
    na_krok: Callable[[int, float], None] | None = None,
) -> WynikTreningu:
    """Trenuje model przez kroki kroków (stały lr po krótkim rozgrzewaniu) i zwraca wynik.

    Strata każdego kroku trafia do train_loss; co co_ile kroków (i po ostatnim) liczona
    jest strata walidacyjna na 5 batchach. na_krok(krok, strata) jest wołana dla kroków
    0, co_ile, 2 * co_ile, ... i dla ostatniego kroku (z modelem w trakcie treningu).
    Po treningu model jest w trybie eval().
    Przykład: kroki=4, co_ile=2 -> len(train_loss) == 4, val_kroki == [1, 3],
        na_krok wołana dla kroków 0, 2, 3
    Przykład: kroki=5, co_ile=2 -> val_kroki == [1, 3, 4], na_krok dla kroków 0, 2, 4
    Przykład: kroki < 1, co_ile < 1 albo batch_size < 1 -> ValueError
    """
    if kroki < 1:
        raise ValueError(f"kroki musi być >= 1, jest {kroki}")
    if co_ile < 1:
        raise ValueError(f"co_ile musi być >= 1, jest {co_ile}")
    if batch_size < 1:
        raise ValueError(f"batch_size musi być >= 1, jest {batch_size}")
    if dane_train.dim() != 1:
        raise ValueError(f"dane_train musi mieć 1 wymiar, ma {dane_train.dim()}")
    if dane_val.dim() != 1:
        raise ValueError(f"dane_val musi mieć 1 wymiar, ma {dane_val.dim()}")

    config = TrainConfig(
        batch_size=batch_size,
        block_size=model.config.block_size,
        max_steps=kroki,
        learning_rate=lr,
        seed=seed,
        log_every=co_ile,
        warmup_steps=min(20, kroki // 10),
        lr_schedule="constant",
        grad_clip=1.0,
        bf16=True,
        eval_every=co_ile,
        eval_batches=5,
    )

    val_kroki: list[int] = []
    val_loss: list[float] = []

    def _na_log(krok: int, strata: float) -> None:
        if na_krok is not None:
            na_krok(krok, strata)

    def _na_eval(krok: int, strata: float) -> None:
        val_kroki.append(krok)
        val_loss.append(strata)

    start = time.perf_counter()
    train_loss = train(
        model,
        dane_train,
        config,
        device,
        _na_log,
        val_data=dane_val,
        on_eval=_na_eval,
    )
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    czas_s = time.perf_counter() - start

    model.eval()
    return WynikTreningu(
        model=model,
        kroki=list(range(kroki)),
        train_loss=train_loss,
        val_kroki=val_kroki,
        val_loss=val_loss,
        lr=lr,
        czas_s=czas_s,
    )
