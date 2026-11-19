"""Etap 5: trening transformera GPT na plikach .bin z tokenami.

Użycie (z korzenia repo):
    uv run python scripts/train_gpt.py [--steps 5000] [--device cuda] [--resume]
        [--until-step N] [--data-dir data/tokens] [--out-dir data/checkpoints]
"""

import argparse
import sys
import time
from pathlib import Path

import torch

from nanowriter.dataset import load_meta, load_tokens
from nanowriter.model import GPT, GPTConfig
from nanowriter.train import TrainConfig, bits_per_char, estimate_loss, train


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń; argv=None oznacza sys.argv[1:].

    Przykład: parse_args([]) -> data_dir=Path("data/tokens"), out_dir=Path("data/checkpoints"),
        steps=5000, batch_size=64, block_size=256, n_layer=4, n_head=4, d_model=128,
        lr=0.002, min_lr=0.0002, warmup=200, weight_decay=0.1, grad_clip=1.0,
        eval_every=250, eval_batches=20, checkpoint_every=500, log_every=100, seed=0,
        device="cuda", no_bf16=False, resume=False, until_step=None
    Przykład: parse_args(["--device", "cpu", "--resume"]) -> device="cpu", resume=True
    """
    parser = argparse.ArgumentParser(description="Trening transformera GPT na tokenach .bin.")
    parser.add_argument(
        "--data-dir", dest="data_dir", type=Path, default=Path("data/tokens"),
        help="katalog z meta.json, train.bin i val.bin",
    )
    parser.add_argument(
        "--out-dir", dest="out_dir", type=Path, default=Path("data/checkpoints"),
        help="katalog na gpt.pt i train_log.jsonl",
    )
    parser.add_argument("--steps", type=int, default=5000, help="liczba kroków treningu")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=64,
                        help="rozmiar batcha")
    parser.add_argument("--block-size", dest="block_size", type=int, default=256,
                        help="długość kontekstu")
    parser.add_argument("--n-layer", dest="n_layer", type=int, default=4,
                        help="liczba bloków transformera")
    parser.add_argument("--n-head", dest="n_head", type=int, default=4,
                        help="liczba głowic uwagi")
    parser.add_argument("--d-model", dest="d_model", type=int, default=128,
                        help="szerokość modelu")
    parser.add_argument("--lr", type=float, default=2e-3,
                        help="maksymalny learning rate")
    parser.add_argument("--min-lr", dest="min_lr", type=float, default=2e-4,
                        help="minimalny learning rate na końcu cosinusa")
    parser.add_argument("--warmup", type=int, default=200,
                        help="liczba kroków liniowego rozgrzewania LR")
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.1,
                        help="weight decay AdamW")
    parser.add_argument("--grad-clip", dest="grad_clip", type=float, default=1.0,
                        help="maksymalna norma gradientów (0 wyłącza)")
    parser.add_argument("--eval-every", dest="eval_every", type=int, default=250,
                        help="co ile kroków liczyć stratę walidacyjną")
    parser.add_argument("--eval-batches", dest="eval_batches", type=int, default=20,
                        help="liczba batchy walidacji w trakcie treningu")
    parser.add_argument("--checkpoint-every", dest="checkpoint_every", type=int, default=500,
                        help="co ile kroków zapisywać checkpoint (0 wyłącza)")
    parser.add_argument("--log-every", dest="log_every", type=int, default=100,
                        help="co ile kroków wypisywać stratę treningową")
    parser.add_argument("--seed", type=int, default=0, help="seed losowości")
    parser.add_argument("--device", type=str, default="cuda",
                        help="urządzenie, np. cuda albo cpu")
    parser.add_argument("--no-bf16", dest="no_bf16", action="store_true",
                        help="wyłącza bfloat16 na GPU")
    parser.add_argument("--resume", action="store_true",
                        help="wznawia trening z out-dir/gpt.pt")
    parser.add_argument("--until-step", dest="until_step", type=int, default=None,
                        help="zatrzymuje trening przed tym krokiem")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> tuple[float, float]:
    """Trenuje GPT, wypisuje postęp i zwraca (strata val na całym zbiorze, bity na znak).

    Przykład: main(["--data-dir", d, "--out-dir", o, "--device", "cpu", "--steps", "20", ...])
        -> (strata, bity), pliki o/"gpt.pt" i o/"train_log.jsonl";
        bity / strata == val_tokens / (val_chars * ln 2)
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    start = time.perf_counter()

    meta = load_meta(args.data_dir / "meta.json")
    train_data = load_tokens(args.data_dir / "train.bin")
    val_data = load_tokens(args.data_dir / "val.bin")
    print(
        f"dane: train {len(train_data)} tokenow, val {len(val_data)} tokenow, "
        f"slownik {meta.vocab_size}"
    )

    torch.manual_seed(args.seed)
    model = GPT(
        GPTConfig(
            vocab_size=meta.vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            d_model=args.d_model,
        )
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params} parametrow, urzadzenie {args.device}")

    config = TrainConfig(
        batch_size=args.batch_size,
        block_size=args.block_size,
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
        """Wypisuje stratę walidacyjną i bity na znak dla kroku step."""
        bpc = bits_per_char(val, meta.val_tokens, meta.val_chars)
        print(f"krok {step:5d}  val {val:.4f}  bity/znak {bpc:.4f}")

    ckpt = args.out_dir / "gpt.pt"
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
    )

    final = estimate_loss(model, val_data, args.batch_size, args.block_size, args.device)
    bpc = bits_per_char(final, meta.val_tokens, meta.val_chars)
    print(f"val (calosc): strata {final:.4f}, bity/znak {bpc:.4f}")
    print(f"checkpoint -> {ckpt}")
    print(f"czas: {time.perf_counter() - start:.1f} s")
    return final, bpc


if __name__ == "__main__":
    main()
