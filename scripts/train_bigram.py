"""Etap 1: trening modelu bigramowego na jednej książce i generowanie tekstu.

Użycie (z korzenia repo):
    uv run python scripts/train_bigram.py [--text PLIK] [--steps 5000] [--device cpu]
"""

import argparse
import sys
from pathlib import Path

import torch

from nanowriter.data import download_book, load_text
from nanowriter.model import BigramLanguageModel
from nanowriter.sample import generate
from nanowriter.tokenizer import CharTokenizer
from nanowriter.train import TrainConfig, train


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń; argv=None oznacza sys.argv[1:].

    Przykład: parse_args([]) -> slug="lalka-tom-pierwszy", text=None, data_dir=Path("data/raw"),
        steps=5000, batch_size=32, block_size=8, lr=0.01, seed=0, log_every=500,
        prompt="Wokulski", length=500, device="cpu"
    Przykład: parse_args(["--text", "a.txt", "--steps", "50"]) -> text=Path("a.txt"), steps=50
    """
    parser = argparse.ArgumentParser(description="Trening modelu bigramowego i generowanie tekstu.")
    parser.add_argument(
        "--slug", type=str, default="lalka-tom-pierwszy", help="slug książki z Wolnych Lektur"
    )
    parser.add_argument(
        "--text", type=Path, default=None, help="lokalny plik tekstowy zamiast pobierania"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/raw"), help="katalog na pobrane książki"
    )
    parser.add_argument("--steps", type=int, default=5000, help="liczba kroków treningu")
    parser.add_argument("--batch-size", type=int, default=32, help="rozmiar batcha")
    parser.add_argument("--block-size", type=int, default=8, help="długość kontekstu w batchu")
    parser.add_argument("--lr", type=float, default=0.01, help="tempo uczenia (learning rate)")
    parser.add_argument("--seed", type=int, default=0, help="ziarno losowości")
    parser.add_argument("--log-every", type=int, default=500, help="co ile kroków wypisywać stratę")
    parser.add_argument(
        "--prompt", type=str, default="Wokulski", help="początek tekstu do generowania"
    )
    parser.add_argument("--length", type=int, default=500, help="liczba znaków do wygenerowania")
    parser.add_argument("--device", type=str, default="cpu", help="urządzenie, np. cpu albo cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> str:
    """Trenuje model bigramowy na tekście książki, wypisuje postęp i wygenerowany tekst.

    Zwraca wygenerowany tekst.

    Przykład: main(["--text", plik_z_ala_ma_kota, "--steps", "50", "--log-every", "25",
                    "--prompt", "ala", "--length", "30"])
        -> tekst o długości 33 zaczynający się od "ala"
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args(argv)

    if args.text is not None:
        text_path = args.text
    else:
        text_path = download_book(args.slug, args.data_dir)

    text = load_text(text_path)
    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.int64)

    torch.manual_seed(args.seed)
    model = BigramLanguageModel(tokenizer.vocab_size).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"znaki: {len(text)}, slownik: {tokenizer.vocab_size}, parametry: {n_params}")

    config = TrainConfig(
        batch_size=args.batch_size,
        block_size=args.block_size,
        max_steps=args.steps,
        learning_rate=args.lr,
        seed=args.seed,
        log_every=args.log_every,
    )

    def on_log(step: int, loss: float) -> None:
        print(f"krok {step:5d}  strata {loss:.4f}")

    train(model, data, config, args.device, on_log=on_log)

    model.eval()
    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.int64, device=args.device)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    out = generate(model, idx, args.length, generator=generator)
    result = tokenizer.decode(out[0].tolist())

    print("--- tekst ---")
    print(result)
    return result


if __name__ == "__main__":
    main()
