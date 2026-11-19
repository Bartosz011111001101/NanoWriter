"""Etap 6: generowanie tekstu z wytrenowanego modelu GPT.

Użycie (z korzenia repo):
    uv run python scripts/generate.py --prompt "Wieczorem" [--length 500] [--temperature 0.8]
        [--top-k 50] [--top-p 0.95] [--seed 0] [--device cuda] [--no-cache]
"""

import argparse
import sys
import time
from pathlib import Path

import torch

from nanowriter.sample import generate
from nanowriter.tokenizer import BPETokenizer
from nanowriter.train import load_gpt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń; argv=None oznacza sys.argv[1:].

    Przykład: parse_args([]) -> checkpoint=Path("data/checkpoints/gpt.pt"),
        tokenizer=Path("data/tokens/tokenizer.json"), prompt="Wieczorem", length=500,
        temperature=0.8, top_k=50, top_p=0.95, seed=0, device="cuda", no_cache=False
    Przykład: parse_args(["--top-k", "0", "--no-cache"]) -> top_k=0, no_cache=True
    """
    parser = argparse.ArgumentParser(description="Generowanie tekstu z wytrenowanego GPT.")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("data/checkpoints/gpt.pt"),
        help="checkpoint wytrenowanego modelu",
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("data/tokens/tokenizer.json"),
        help="słownik BPE zapisany przez prepare_tokens",
    )
    parser.add_argument("--prompt", type=str, default="Wieczorem",
                        help="początek tekstu, od którego model pisze dalej")
    parser.add_argument("--length", type=int, default=500,
                        help="liczba nowych tokenów do wygenerowania")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="temperatura próbkowania (0 = najbardziej prawdopodobny token)")
    parser.add_argument("--top-k", dest="top_k", type=int, default=50,
                        help="losuj tylko spośród k najbardziej prawdopodobnych (0 wyłącza)")
    parser.add_argument("--top-p", dest="top_p", type=float, default=0.95,
                        help="nucleus sampling o zadanej masie (1.0 wyłącza)")
    parser.add_argument("--seed", type=int, default=0, help="seed losowania")
    parser.add_argument("--device", type=str, default="cuda",
                        help="urządzenie, np. cuda albo cpu")
    parser.add_argument("--no-cache", dest="no_cache", action="store_true",
                        help="generuje bez KV cache")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> str:
    """Wczytuje model i słownik, generuje dalszy ciąg --prompt, wypisuje go i zwraca tekst.

    Przykład: main(["--checkpoint", c, "--tokenizer", t, "--prompt", "ala", "--length", "40",
                    "--device", "cpu"])
        -> tekst zaczynający się od "ala"; z "--no-cache" ten sam tekst
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)

    tokenizer = BPETokenizer.load(args.tokenizer)
    model = load_gpt(args.checkpoint, args.device)

    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.int64, device=args.device)
    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)

    top_k = None if args.top_k == 0 else args.top_k
    top_p = None if args.top_p == 1.0 else args.top_p

    start = time.perf_counter()
    out = generate(
        model,
        idx,
        args.length,
        generator=generator,
        block_size=model.config.block_size,
        temperature=args.temperature,
        top_k=top_k,
        top_p=top_p,
        use_cache=not args.no_cache,
    )
    elapsed = time.perf_counter() - start

    text = tokenizer.decode(out[0].tolist())
    print("--- tekst ---")
    print(text)
    cache = "nie" if args.no_cache else "tak"
    print(f"tokeny: {args.length}, czas: {elapsed:.2f} s, cache: {cache}")
    return text


if __name__ == "__main__":
    main()
