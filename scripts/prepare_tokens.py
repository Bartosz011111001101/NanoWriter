"""Etap 3: trening tokenizera BPE i zapis danych jako pliki .bin z tokenami.

Użycie (z korzenia repo):
    uv run python scripts/prepare_tokens.py [--data-dir data] [--out-dir data/tokens]
        [--vocab-size 1024] [--sample-mb 8]
"""

import argparse
import sys
import time
from pathlib import Path

from nanowriter.data import load_text
from nanowriter.dataset import TOKEN_DTYPE, TokenMeta, encode_file, save_meta
from nanowriter.tokenizer import BPETokenizer, sample_lines


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń; argv=None oznacza sys.argv[1:].

    Przykład: parse_args([]) -> data_dir=Path("data"), out_dir=Path("data/tokens"),
        vocab_size=1024, sample_mb=8.0
    Przykład: parse_args(["--vocab-size", "300"]) -> vocab_size=300
    """
    parser = argparse.ArgumentParser(description="Trening BPE i zapis tokenów do plików .bin.")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="katalog z train.txt i val.txt"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/tokens"), help="katalog na wynikowe pliki"
    )
    parser.add_argument("--vocab-size", type=int, default=1024, help="rozmiar słownika BPE")
    parser.add_argument(
        "--sample-mb", type=float, default=8.0, help="ile milionów znaków próbki do treningu BPE"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> TokenMeta:
    """Uczy BPE na próbce train.txt, koduje train.txt i val.txt do .bin, zapisuje meta.json.

    Zwraca zapisane TokenMeta.
    Przykład: train.txt = "Ala ma kota, a kot ma Ale.\\n" * 200, val.txt = "Kot ma Ale.\\n" * 20,
        main(["--data-dir", d, "--out-dir", d + "/tokens", "--vocab-size", "300",
              "--sample-mb", "1"])
        -> TokenMeta(vocab_size=267, dtype="uint16", train_tokens=2000, train_chars=5400,
                     val_tokens=120, val_chars=240)
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args(argv)
    start = time.perf_counter()

    train_path = args.data_dir / "train.txt"
    val_path = args.data_dir / "val.txt"

    text = load_text(train_path)
    sample = sample_lines(text, int(args.sample_mb * 1_000_000))
    print(f"probka: {len(sample)} z {len(text)} znakow, trening BPE (vocab {args.vocab_size})")

    tokenizer = BPETokenizer.train(sample, args.vocab_size)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.out_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    print(f"slownik: {tokenizer.vocab_size} tokenow -> {tokenizer_path}")

    train_bin = args.out_dir / "train.bin"
    train_tokens, train_chars = encode_file(tokenizer, train_path, train_bin)
    print(
        f"train: {train_tokens} tokenow, {train_chars} znakow, "
        f"{train_chars / max(train_tokens, 1):.3f} znaku/token -> {train_bin}"
    )

    val_bin = args.out_dir / "val.bin"
    val_tokens, val_chars = encode_file(tokenizer, val_path, val_bin)
    print(
        f"val: {val_tokens} tokenow, {val_chars} znakow, "
        f"{val_chars / max(val_tokens, 1):.3f} znaku/token -> {val_bin}"
    )

    meta = TokenMeta(
        tokenizer.vocab_size, TOKEN_DTYPE, train_tokens, train_chars, val_tokens, val_chars
    )
    meta_path = args.out_dir / "meta.json"
    save_meta(meta, meta_path)
    print(f"meta -> {meta_path}")

    print(f"czas: {time.perf_counter() - start:.1f} s")
    return meta


if __name__ == "__main__":
    main()
