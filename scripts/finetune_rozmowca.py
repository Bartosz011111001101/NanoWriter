"""Zadanie 09: douczanie rozmówcy na rozmowach (strata tylko na wypowiedziach B).

Użycie (z korzenia repo):
    uv run python scripts/finetune_rozmowca.py [--init-from data/rozmowca/ckpt-wstepny/gpt.pt]
        [--data-dir data/rozmowca/tokeny-rozmowy] [--out-dir data/rozmowca/ckpt-rozmowy]
        [--steps 300] [--device cuda] [--resume] [--until-step N]
"""

import argparse
import math
import sys
import time
from pathlib import Path

from nanowriter.dataset import load_mask, load_meta, load_tokens
from nanowriter.train import TrainConfig, estimate_loss, load_gpt, train


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń; argv=None oznacza sys.argv[1:].

    Przykład: parse_args([]) -> init_from=Path("data/rozmowca/ckpt-wstepny/gpt.pt"),
        data_dir=Path("data/rozmowca/tokeny-rozmowy"), out_dir=Path("data/rozmowca/ckpt-rozmowy"),
        steps=300, batch_size=16, lr=0.0003, min_lr=0.00003, warmup=20, weight_decay=0.1,
        grad_clip=1.0, eval_every=25, eval_batches=20, checkpoint_every=50, log_every=10,
        seed=0, device="cuda", no_bf16=False, resume=False, until_step=None
    Przykład: parse_args(["--device", "cpu", "--resume"]) -> device="cpu", resume=True
    """
    parser = argparse.ArgumentParser(
        description="Douczanie modelu GPT na rozmowach (strata tylko na wypowiedziach B)."
    )
    parser.add_argument(
        "--init-from", dest="init_from", type=Path,
        default=Path("data/rozmowca/ckpt-wstepny/gpt.pt"),
        help="checkpoint treningu wstępnego, z którego startuje douczanie",
    )
    parser.add_argument(
        "--data-dir", dest="data_dir", type=Path,
        default=Path("data/rozmowca/tokeny-rozmowy"),
        help="katalog z meta.json, train.bin, train_mask.bin, val.bin i val_mask.bin",
    )
    parser.add_argument(
        "--out-dir", dest="out_dir", type=Path,
        default=Path("data/rozmowca/ckpt-rozmowy"),
        help="katalog na gpt.pt i train_log.jsonl (nie może być katalogiem init-from)",
    )
    parser.add_argument("--steps", type=int, default=300, help="liczba kroków douczania")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=16,
                        help="rozmiar batcha")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="maksymalny learning rate")
    parser.add_argument("--min-lr", dest="min_lr", type=float, default=3e-5,
                        help="minimalny learning rate na końcu cosinusa")
    parser.add_argument("--warmup", type=int, default=20,
                        help="liczba kroków liniowego rozgrzewania LR")
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.1,
                        help="weight decay AdamW")
    parser.add_argument("--grad-clip", dest="grad_clip", type=float, default=1.0,
                        help="maksymalna norma gradientów (0 wyłącza)")
    parser.add_argument("--eval-every", dest="eval_every", type=int, default=25,
                        help="co ile kroków liczyć stratę walidacyjną")
    parser.add_argument("--eval-batches", dest="eval_batches", type=int, default=20,
                        help="liczba batchy walidacji w trakcie douczania")
    parser.add_argument("--checkpoint-every", dest="checkpoint_every", type=int, default=50,
                        help="co ile kroków zapisywać checkpoint (0 wyłącza)")
    parser.add_argument("--log-every", dest="log_every", type=int, default=10,
                        help="co ile kroków wypisywać stratę treningową")
    parser.add_argument("--seed", type=int, default=0, help="seed losowości")
    parser.add_argument("--device", type=str, default="cuda",
                        help="urządzenie, np. cuda albo cpu")
    parser.add_argument("--no-bf16", dest="no_bf16", action="store_true",
                        help="wyłącza bfloat16 na GPU")
    parser.add_argument("--resume", action="store_true",
                        help="wznawia douczanie z out-dir/gpt.pt")
    parser.add_argument("--until-step", dest="until_step", type=int, default=None,
                        help="zatrzymuje douczanie przed tym krokiem")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> tuple[float, float]:
    """Douczanie modelu z checkpointu na rozmowach; zwraca (strata val przed, strata val po).

    Strata walidacyjna jest liczona tylko na tokenach z maską 1, na całym zbiorze val.
    Przykład: main(["--init-from", p, "--data-dir", d, "--out-dir", o, "--device", "cpu",
        "--steps", "60", ...]) -> (przed, po) z po < przed, pliki o/"gpt.pt" i
        o/"train_log.jsonl", plik p bez zmian
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    start = time.perf_counter()

    ckpt = args.out_dir / "gpt.pt"
    if ckpt.resolve() == args.init_from.resolve():
        raise ValueError(
            f"nie wolno nadpisać checkpointu treningu wstępnego: {args.init_from}"
        )

    meta = load_meta(args.data_dir / "meta.json")
    train_data = load_tokens(args.data_dir / "train.bin")
    train_mask = load_mask(args.data_dir / "train_mask.bin")
    val_data = load_tokens(args.data_dir / "val.bin")
    val_mask = load_mask(args.data_dir / "val_mask.bin")
    print(
        f"dane: train {len(train_data)} tokenow (maska {int(train_mask.sum())}), "
        f"val {len(val_data)} tokenow (maska {int(val_mask.sum())}), "
        f"slownik {meta.vocab_size}"
    )

    model = load_gpt(args.init_from, args.device)
    if model.config.vocab_size != meta.vocab_size:
        raise ValueError(
            f"vocab_size modelu ({model.config.vocab_size}) nie zgadza się ze słownikiem "
            f"danych ({meta.vocab_size})"
        )
    block_size = model.config.block_size
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"model: {n_params} parametrow z {args.init_from}, "
        f"urzadzenie {args.device}, kontekst {block_size}"
    )

    before = estimate_loss(
        model, val_data, args.batch_size, block_size, args.device, mask=val_mask
    )
    print(f"val przed douczaniem: strata {before:.4f}")

    config = TrainConfig(
        batch_size=args.batch_size,
        block_size=block_size,
        max_steps=args.steps,
        learning_rate=args.lr,
        seed=args.seed,
        log_every=args.log_every,
        weight_decay=args.weight_decay,
        beta1=0.9,
        beta2=0.95,
        warmup_steps=args.warmup,
        lr_schedule="cosine",
        min_lr=args.min_lr,
        grad_clip=args.grad_clip,
        bf16=not args.no_bf16,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        checkpoint_every=args.checkpoint_every,
    )

    def on_log(step: int, loss: float) -> None:
        """Wypisuje stratę treningową dla kroku step."""
        print(f"krok {step:5d}  strata {loss:.4f}")

    def on_eval(step: int, val: float) -> None:
        """Wypisuje stratę walidacyjną i perplexity dla kroku step."""
        print(f"krok {step:5d}  val {val:.4f}  perplexity {math.exp(val):.2f}")

    train(
        model,
        train_data,
        config,
        args.device,
        on_log=on_log,
        val_data=val_data,
        on_eval=on_eval,
        checkpoint_path=ckpt,
        resume_from=ckpt if args.resume else None,
        log_path=args.out_dir / "train_log.jsonl",
        until_step=args.until_step,
        mask=train_mask,
        val_mask=val_mask,
    )

    after = estimate_loss(
        model, val_data, args.batch_size, block_size, args.device, mask=val_mask
    )
    print(f"val po douczaniu: strata {after:.4f}, perplexity {math.exp(after):.2f}")
    print(f"checkpoint -> {ckpt}")
    print(f"czas: {time.perf_counter() - start:.1f} s")
    return before, after


if __name__ == "__main__":
    main()
