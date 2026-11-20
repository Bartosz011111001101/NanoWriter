"""Pętla treningu modeli językowych: harmonogram LR, walidacja, checkpointy."""

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from nanowriter.dataset import get_batch, get_masked_batch
from nanowriter.model import GPT, GPTConfig


@dataclass(frozen=True)
class TrainConfig:
    """Ustawienia treningu.

    Przykład: TrainConfig() -> batch_size=32, block_size=8, max_steps=5000,
        learning_rate=0.01, seed=0, log_every=500, weight_decay=0.01, beta1=0.9,
        beta2=0.999, warmup_steps=0, lr_schedule="constant", min_lr=0.0, grad_clip=0.0,
        bf16=False, eval_every=0, eval_batches=20, checkpoint_every=0
    Przykład: TrainConfig(batch_size=0) -> ValueError
    Przykład: TrainConfig(lr_schedule="linear") -> ValueError
    """

    batch_size: int = 32
    block_size: int = 8
    max_steps: int = 5000
    learning_rate: float = 1e-2
    seed: int = 0
    log_every: int = 500
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_steps: int = 0
    lr_schedule: str = "constant"
    min_lr: float = 0.0
    grad_clip: float = 0.0
    bf16: bool = False
    eval_every: int = 0
    eval_batches: int = 20
    checkpoint_every: int = 0

    def __post_init__(self) -> None:
        """Sprawdza zakresy pól; błąd ma nazwę pola w komunikacie.

        Przykład: TrainConfig(max_steps=0) -> poprawny
        Przykład: TrainConfig(learning_rate=0.0) -> ValueError
        Przykład: TrainConfig(min_lr=0.1, learning_rate=0.01) -> ValueError ("min_lr")
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
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay musi być >= 0, jest {self.weight_decay}")
        if not 0 <= self.beta1 < 1:
            raise ValueError(f"beta1 musi być w [0, 1), jest {self.beta1}")
        if not 0 <= self.beta2 < 1:
            raise ValueError(f"beta2 musi być w [0, 1), jest {self.beta2}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps musi być >= 0, jest {self.warmup_steps}")
        if self.lr_schedule not in ("constant", "cosine"):
            raise ValueError(
                f"lr_schedule musi być 'constant' albo 'cosine', jest {self.lr_schedule!r}"
            )
        if self.min_lr < 0:
            raise ValueError(f"min_lr musi być >= 0, jest {self.min_lr}")
        if self.min_lr > self.learning_rate:
            raise ValueError(
                f"min_lr ({self.min_lr}) nie może przekraczać learning_rate ({self.learning_rate})"
            )
        if self.grad_clip < 0:
            raise ValueError(f"grad_clip musi być >= 0, jest {self.grad_clip}")
        if self.eval_every < 0:
            raise ValueError(f"eval_every musi być >= 0, jest {self.eval_every}")
        if self.eval_batches < 1:
            raise ValueError(f"eval_batches musi być >= 1, jest {self.eval_batches}")
        if self.checkpoint_every < 0:
            raise ValueError(f"checkpoint_every musi być >= 0, jest {self.checkpoint_every}")


def compute_loss(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Zwraca średnią entropię krzyżową po pozycjach batcha, opcjonalnie tylko tam, gdzie mask = 1.

    logits: FloatTensor[B, T, V], targets: LongTensor[B, T], mask: bool albo 0/1 [B, T] albo None.
    Zwraca: skalar (tensor o kształcie []) z gradientem względem logits.
    Przykład: compute_loss(torch.zeros(1, 2, 4), torch.tensor([[0, 3]])) -> tensor(1.3863) (ln 4)
    Przykład: l = zeros[1, 2, 4] z l[0, 1, 3] = 10, t = [[0, 3]]:
        mask [[1, 0]] -> 1.3863, [[0, 1]] -> 0.0001, [[1, 1]] -> 0.6932, [[0, 0]] -> 0.0
    Przykład: mask o kształcie [1, 3] przy targets [1, 2] -> ValueError
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
    if mask is None:
        return F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))
    if mask.shape != targets.shape:
        raise ValueError(
            f"mask {tuple(mask.shape)} nie pasuje do targets {tuple(targets.shape)} "
            "(oczekiwane [B, T])"
        )
    per_position = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1), reduction="none")
    weights = mask.reshape(-1).to(device=per_position.device, dtype=per_position.dtype)
    return (per_position * weights).sum() / weights.sum().clamp(min=1.0)


def lr_at(step: int, config: TrainConfig) -> float:
    """Zwraca learning rate dla kroku step (od 0): liniowy warmup, potem stały albo cosinus.

    Przykład: config = TrainConfig(learning_rate=1.0, warmup_steps=2, max_steps=6,
                                   lr_schedule="cosine", min_lr=0.0)
        kroki 0..5 -> 0.5, 1.0, 1.0, 0.853553, 0.5, 0.146447
    Przykład: TrainConfig(learning_rate=0.01) dla każdego kroku -> 0.01
    Przykład: lr_at(-1, config) -> ValueError
    """
    if step < 0:
        raise ValueError(f"step musi być >= 0, jest {step}")
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    if config.lr_schedule == "constant":
        return config.learning_rate
    progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    scale = 0.5 * (1 + math.cos(math.pi * progress))
    return config.min_lr + scale * (config.learning_rate - config.min_lr)


def make_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.AdamW:
    """Tworzy AdamW z dwiema grupami: parametry 2D+ z weight_decay, pozostałe bez.

    Przykład: make_optimizer(GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2,
                                           n_head=2, d_model=32)), TrainConfig(weight_decay=0.1))
        -> 2 grupy: 10 tensorów z weight_decay=0.1 i 18 tensorów z weight_decay=0.0
    Przykład: make_optimizer(nn.Embedding(4, 4), TrainConfig()) -> 1 grupa, weight_decay=0.01
    """
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": config.weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=config.learning_rate, betas=(config.beta1, config.beta2))


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    grad_clip: float = 0.0,
    bf16: bool = False,
    mask: torch.Tensor | None = None,
) -> float:
    """Wykonuje jeden krok uczenia na batchu (x, y) i zwraca stratę sprzed aktualizacji wag.

    x, y: LongTensor[B, T] na urządzeniu modelu; mask [B, T] zawęża stratę do pozycji z maską 1.
    grad_clip > 0: norma gradientów przycięta do grad_clip przed krokiem optymalizatora.
    bf16: forward i strata w autocast bfloat16 na urządzeniu x.
    Przykład: model nn.Embedding(4, 4) z wagami zerowymi, x = [[0, 1]], y = [[1, 2]]
        -> 1.3862943... (ln 4), wagi po kroku inne niż zera
    Przykład: mask None -> to samo co dotąd; mask bez jedynek -> zwraca 0.0, wagi bez zmian
    """
    optimizer.zero_grad()
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=bf16):
        logits = model(x)
        loss = compute_loss(logits, y, mask)
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item()


def estimate_loss(
    model: nn.Module,
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str | torch.device,
    *,
    max_batches: int | None = None,
    mask: torch.Tensor | None = None,
) -> float:
    """Zwraca średnią stratę na oknach data bez nakładania (deterministycznie, bez gradientów).

    Okno i: x = data[i*block_size : (i+1)*block_size], y przesunięte o 1; okna po kolei
    w batchach po batch_size; max_batches -> tylko pierwsze batche. Tryb modelu przywrócony.
    mask 1D (jak data) zawęża średnią do tokenów z maską 1: maska okna i to
    mask[i*bs + 1 : i*bs + bs + 1] (maska celów y), wynik to średnia po wszystkich
    pozycjach z maską 1 w użytych oknach.
    Przykład: model nn.Embedding(4, 4) z wagami zerowymi, data = torch.arange(9) % 4,
        batch_size=2, block_size=4 -> 1.3862943... (ln 4), 2 okna
    Przykład: data = torch.arange(4), block_size=4 -> ValueError (N <= block_size)
    Przykład: ta sama sytuacja z mask = torch.ones(9) -> 1.3862943... (ln 4)
    Przykład: ta sama sytuacja z mask = torch.zeros(9) -> ValueError (brak tokenów z maską 1)
    Przykład: mask o innej długości niż data -> ValueError
    """
    if data.dim() != 1:
        raise ValueError(f"data musi mieć 1 wymiar, ma {data.dim()}")
    if batch_size < 1:
        raise ValueError(f"batch_size musi być >= 1, jest {batch_size}")
    if block_size < 1:
        raise ValueError(f"block_size musi być >= 1, jest {block_size}")
    n_data = data.shape[0]
    if n_data <= block_size:
        raise ValueError(f"data ma {n_data} tokenów, potrzeba więcej niż block_size ({block_size})")
    if max_batches is not None and max_batches < 1:
        raise ValueError(f"max_batches musi być >= 1, jest {max_batches}")
    if mask is not None and mask.shape != data.shape:
        raise ValueError(f"mask {tuple(mask.shape)} nie pasuje do data {tuple(data.shape)}")
    n_windows = (n_data - 1) // block_size
    was_training = model.training
    model.eval()
    total = 0.0
    used = 0
    mask_count = 0
    try:
        with torch.no_grad():
            for batch_index, first in enumerate(range(0, n_windows, batch_size)):
                if max_batches is not None and batch_index >= max_batches:
                    break
                ids = range(first, min(first + batch_size, n_windows))
                x = torch.stack([data[i * block_size : (i + 1) * block_size] for i in ids])
                y = torch.stack([data[i * block_size + 1 : (i + 1) * block_size + 1] for i in ids])
                x = x.to(device=device, dtype=torch.int64)
                y = y.to(device=device, dtype=torch.int64)
                if mask is None:
                    loss = compute_loss(model(x), y)
                    total += loss.item() * len(ids)
                    used += len(ids)
                    continue
                m = torch.stack([mask[i * block_size + 1 : (i + 1) * block_size + 1] for i in ids])
                m = m.to(device=device).bool()
                ones = int(m.sum().item())
                if ones == 0:
                    continue
                loss = compute_loss(model(x), y, m)
                total += loss.item() * ones
                mask_count += ones
    finally:
        model.train(was_training)
    if mask is not None:
        if mask_count == 0:
            raise ValueError("brak tokenów z maską 1 w użytych oknach")
        return total / mask_count
    return total / used


def bits_per_char(loss: float, n_tokens: int, n_chars: int) -> float:
    """Przelicza stratę (nat na token) na bity na znak tekstu: loss * n_tokens / (n_chars * ln 2).

    Przykład: bits_per_char(2.0, 100, 250) -> 1.154156...
    Przykład: bits_per_char(math.log(2), 3, 3) -> 1.0
    Przykład: bits_per_char(1.0, 10, 0) -> ValueError
    """
    if n_chars < 1:
        raise ValueError(f"n_chars musi być >= 1, jest {n_chars}")
    if n_tokens < 0:
        raise ValueError(f"n_tokens musi być >= 0, jest {n_tokens}")
    return loss * n_tokens / (n_chars * math.log(2))


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    generator: torch.Generator,
    config: TrainConfig,
) -> None:
    """Zapisuje model, optymalizator, krok, stan generatora batchy i konfiguracje do path.

    Przykład: save_checkpoint(d / "c.pt", gpt, opt, 10, gen, cfg)
        -> plik d/"c.pt", load_checkpoint(d / "c.pt")["step"] == 10
    """
    model_config = getattr(model, "config", None)
    has_config = is_dataclass(model_config) and not isinstance(model_config, type)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "generator": generator.get_state(),
        "config": asdict(config),
        "model_config": asdict(model_config) if has_config else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Wczytuje checkpoint zapisany przez save_checkpoint; wszystkie tensory na CPU.

    Przykład: load_checkpoint(p).keys()
        -> {"model", "optimizer", "step", "generator", "config", "model_config"}
    Przykład: brak pliku -> FileNotFoundError
    """
    return torch.load(path, map_location="cpu", weights_only=True)


def load_gpt(path: Path, device: str | torch.device) -> GPT:
    """Odtwarza GPT z checkpointu (model_config + wagi), na device, w trybie eval().

    Przykład: load_gpt(p, "cpu")(idx) == logity zapisanego modelu
    Przykład: checkpoint modelu bez config (np. nn.Embedding) -> ValueError
    """
    checkpoint = load_checkpoint(path)
    model_config = checkpoint["model_config"]
    if model_config is None:
        raise ValueError(f"checkpoint {path} nie zawiera model_config, nie da się odtworzyć GPT")
    model = GPT(GPTConfig(**model_config))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def _append_log(log_path: Path | None, record: dict[str, float | int]) -> None:
    """Dopisuje jedną linię JSON do pliku logu (gdy log_path podany)."""
    if log_path is None:
        return
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def train(
    model: nn.Module,
    data: torch.Tensor,
    config: TrainConfig,
    device: str | torch.device,
    on_log: Callable[[int, float], None] | None = None,
    *,
    val_data: torch.Tensor | None = None,
    on_eval: Callable[[int, float], None] | None = None,
    checkpoint_path: Path | None = None,
    resume_from: Path | None = None,
    log_path: Path | None = None,
    until_step: int | None = None,
    mask: torch.Tensor | None = None,
    val_mask: torch.Tensor | None = None,
) -> list[float]:
    """Trenuje model krokami start..end-1 i zwraca stratę każdego wykonanego kroku.

    model: już na device; data, val_data: tensory 1D id tokenów na CPU.
    on_log(step, loss): dla step % log_every == 0 i dla step == max_steps - 1 (kroki od 0).
    on_eval(step, val_loss): po kroku, gdy val_data podane, eval_every > 0
        i ((step + 1) % eval_every == 0 albo step == max_steps - 1).
    resume_from: start = krok z checkpointu, odtworzone model, optymalizator i generator.
    until_step: koniec = min(until_step, max_steps); None -> max_steps.
    checkpoint_path: zapis co checkpoint_every kroków i na końcu wywołania.
    mask: maska 1D dla data, batche z get_masked_batch; val_mask: maska 1D dla val_data.
    Przykład: max_steps=5, log_every=2 -> lista 5 strat, on_log wołana dla kroków 0, 2, 4
    Przykład: max_steps=4, log_every=2 -> on_log wołana dla kroków 0, 2, 3
    Przykład: max_steps=0 -> [], on_log nie wołana
    Przykład: straty train(until_step=k, checkpoint_path=p) + train(resume_from=p)
        == straty train() bez przerwy (CPU, ten sam config)
    Przykład: mask None, val_mask None -> dokładnie jak dotąd
    Przykład: mask o innej długości niż data -> ValueError przed pierwszym krokiem
    Przykład: val_mask podane bez val_data -> ValueError
    """
    if until_step is not None and until_step < 0:
        raise ValueError(f"until_step musi być >= 0, jest {until_step}")
    if mask is not None and mask.shape != data.shape:
        raise ValueError(f"mask {tuple(mask.shape)} nie pasuje do data {tuple(data.shape)}")
    if val_mask is not None:
        if val_data is None:
            raise ValueError("val_mask wymaga val_data")
        if val_mask.shape != val_data.shape:
            raise ValueError(
                f"val_mask {tuple(val_mask.shape)} nie pasuje do val_data {tuple(val_data.shape)}"
            )
    model.train()
    optimizer = make_optimizer(model, config)
    generator = torch.Generator().manual_seed(config.seed)
    start = 0
    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        generator.set_state(checkpoint["generator"])
        start = checkpoint["step"]
    end = config.max_steps if until_step is None else min(until_step, config.max_steps)
    use_bf16 = config.bf16 and torch.device(device).type == "cuda"
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if resume_from is None:
            log_path.write_text("", encoding="utf-8")
    losses: list[float] = []
    last_step = config.max_steps - 1
    for step in range(start, end):
        lr = lr_at(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if mask is None:
            x, y = get_batch(data, config.batch_size, config.block_size, generator, device)
            loss = train_step(model, optimizer, x, y, grad_clip=config.grad_clip, bf16=use_bf16)
        else:
            x, y, m = get_masked_batch(
                data, mask, config.batch_size, config.block_size, generator, device
            )
            loss = train_step(
                model, optimizer, x, y, grad_clip=config.grad_clip, bf16=use_bf16, mask=m
            )
        losses.append(loss)
        if step % config.log_every == 0 or step == last_step:
            if on_log is not None:
                on_log(step, loss)
            _append_log(log_path, {"step": step, "train_loss": loss, "lr": lr})
        if (
            val_data is not None
            and config.eval_every > 0
            and ((step + 1) % config.eval_every == 0 or step == last_step)
        ):
            val = estimate_loss(
                model,
                val_data,
                config.batch_size,
                config.block_size,
                device,
                max_batches=config.eval_batches,
                mask=val_mask,
            )
            if on_eval is not None:
                on_eval(step, val)
            _append_log(log_path, {"step": step, "val_loss": val})
            model.train()
        if (
            checkpoint_path is not None
            and config.checkpoint_every > 0
            and (step + 1) % config.checkpoint_every == 0
        ):
            save_checkpoint(checkpoint_path, model, optimizer, step + 1, generator, config)
    if checkpoint_path is not None and end > start:
        save_checkpoint(checkpoint_path, model, optimizer, end, generator, config)
    return losses
