"""Etap 10: rozmowa z modelem w terminalu.

Użycie (z korzenia repo):
    uv run python scripts/chat.py --checkpoint <plik.pt> --tokenizer <tokenizer.json>
        [--max-turns 4] [--max-reply-tokens 64] [--temperature 0.7] [--top-k 40] [--top-p 0.9]
        [--seed 0] [--device cuda] [--no-cache]
Wyjście z rozmowy: linia /koniec albo Ctrl+C.
"""

import argparse
import sys
from pathlib import Path

import torch

from nanowriter.sample import chat_reply
from nanowriter.tokenizer import BPETokenizer
from nanowriter.train import load_gpt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń; argv=None oznacza sys.argv[1:].

    Przykład: parse_args(["--checkpoint", "m.pt", "--tokenizer", "t.json"])
        -> checkpoint=Path("m.pt"), tokenizer=Path("t.json"), max_turns=4, max_reply_tokens=64,
        temperature=0.7, top_k=40, top_p=0.9, seed=0, device="cuda", no_cache=False
    Przykład: parse_args([]) -> SystemExit (brakuje --checkpoint i --tokenizer)
    """
    parser = argparse.ArgumentParser(description="Rozmowa z wytrenowanym modelem w terminalu.")
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="checkpoint wytrenowanego modelu",
    )
    parser.add_argument(
        "--tokenizer", type=Path, required=True,
        help="słownik BPE zapisany przez prepare_tokens",
    )
    parser.add_argument("--max-turns", dest="max_turns", type=int, default=4,
                        help="ile ostatnich wymian pamięta model")
    parser.add_argument("--max-reply-tokens", dest="max_reply_tokens", type=int, default=64,
                        help="najwyżej tyle tokenów w jednej odpowiedzi")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="temperatura próbkowania (0 = najbardziej prawdopodobny token)")
    parser.add_argument("--top-k", dest="top_k", type=int, default=40,
                        help="losuj tylko spośród k najbardziej prawdopodobnych (0 wyłącza)")
    parser.add_argument("--top-p", dest="top_p", type=float, default=0.9,
                        help="nucleus sampling o zadanej masie (1.0 wyłącza)")
    parser.add_argument("--seed", type=int, default=0, help="seed losowania")
    parser.add_argument("--device", type=str, default="cuda",
                        help="urządzenie, np. cuda albo cpu")
    parser.add_argument("--no-cache", dest="no_cache", action="store_true",
                        help="generuje odpowiedzi bez KV cache")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[tuple[str, str]]:
    """Wczytuje model i słownik, prowadzi rozmowę w pętli i zwraca historię wymian.

    Przykład: wejście "cześć", potem "/koniec" -> wypisuje "B: <odpowiedź>" raz;
        zwraca [("cześć", "<odpowiedź>")]
    Przykład: pusta linia -> pomijana; EOFError albo KeyboardInterrupt -> koniec rozmowy
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    args = parse_args(argv)

    tokenizer = BPETokenizer.load(args.tokenizer)
    model = load_gpt(args.checkpoint, args.device)
    block_size = model.config.block_size

    if args.max_reply_tokens >= block_size:
        sys.exit(
            f"--max-reply-tokens ({args.max_reply_tokens}) musi być mniejsze "
            f"niż block_size modelu ({block_size})"
        )

    top_k = None if args.top_k == 0 else args.top_k
    top_p = None if args.top_p == 1.0 else args.top_p

    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)

    print("Rozmowa z modelem. Wpisz /koniec albo naciśnij Ctrl+C, żeby wyjść.")
    history: list[tuple[str, str]] = []
    while True:
        try:
            line = input("A: ")
        except (EOFError, KeyboardInterrupt):
            break
        text = line.strip()
        if text == "":
            continue
        if text == "/koniec":
            break
        reply = chat_reply(
            model,
            tokenizer,
            history,
            text,
            device=args.device,
            block_size=block_size,
            max_new_tokens=args.max_reply_tokens,
            max_turns=args.max_turns,
            generator=generator,
            temperature=args.temperature,
            top_k=top_k,
            top_p=top_p,
            use_cache=not args.no_cache,
        )
        if reply == "":
            print("B: (brak odpowiedzi)")
            continue
        print(f"B: {reply}")
        history.append((text, reply))
    return history


if __name__ == "__main__":
    main()
