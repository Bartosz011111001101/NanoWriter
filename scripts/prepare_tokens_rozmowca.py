"""Zadanie 08: trening BPE rozmówcy i zapis tokenów korpusu oraz rozmów (z maską) jako .bin.

Użycie (z korzenia repo):
    uv run python scripts/prepare_tokens_rozmowca.py [--corpus-dir data/pretrain]
        [--dialogs-dir data/rozmowy] [--out-dir data/rozmowca] [--vocab-size 8192]
        [--sample-mb 8] [--dialog-sample-mb 4] [--block-mb 4]
"""

import argparse
import sys
import time
from pathlib import Path

from nanowriter.dataset import (
    TOKEN_DTYPE,
    TokenMeta,
    encode_dialogs_file,
    encode_file,
    load_mask,
    save_meta,
)
from nanowriter.tokenizer import BPETokenizer, sample_file


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń; argv=None oznacza sys.argv[1:].

    Przykład: parse_args([]) -> corpus_dir=Path("data/pretrain"),
        dialogs_dir=Path("data/rozmowy"), out_dir=Path("data/rozmowca"), vocab_size=8192,
        sample_mb=8.0, dialog_sample_mb=4.0, block_mb=4.0
    Przykład: parse_args(["--vocab-size", "300"]) -> vocab_size=300
    """
    parser = argparse.ArgumentParser(
        description="Trening BPE rozmówcy i zapis tokenów korpusu oraz rozmów (z maską)."
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/pretrain"),
        help="katalog z korpusem wstępnym: train.txt i val.txt",
    )
    parser.add_argument(
        "--dialogs-dir",
        type=Path,
        default=Path("data/rozmowy"),
        help="katalog z rozmowami: train.txt i val.txt (linie A: i B:)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/rozmowca"),
        help="katalog na wynikowe pliki",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=8192, help="rozmiar słownika BPE"
    )
    parser.add_argument(
        "--sample-mb",
        type=float,
        default=8.0,
        help="ile milionów znaków korpusu do próbki treningu BPE",
    )
    parser.add_argument(
        "--dialog-sample-mb",
        type=float,
        default=4.0,
        help="ile milionów znaków rozmów do próbki treningu BPE",
    )
    parser.add_argument(
        "--block-mb",
        type=float,
        default=4.0,
        help="rozmiar bloku kodowania korpusu w milionach znaków",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> tuple[TokenMeta, TokenMeta]:
    """Uczy BPE na mieszance korpusu i rozmów, koduje oba zestawy i zwraca ich TokenMeta.

    Zwraca (meta korpusu wstępnego, meta rozmów); oba zapisane też jako meta.json.
    Przykład: korpus "ala ma kota\\n" * 200 (train) i "kot ma ale\\n" * 20 (val), rozmowy
        "A: hej\\nB: ok tak\\n\\n" * 50 (train) i * 5 (val), --vocab-size 256
        -> (TokenMeta(256, "uint16", 2400, 2400, 220, 220),
            TokenMeta(256, "uint16", 900, 900, 90, 90)), z maską 400 jedynek w train rozmów
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args(argv)
    start = time.perf_counter()

    tokens_dir = args.out_dir / "tokeny-wstepne"
    dialogs_tokens_dir = args.out_dir / "tokeny-rozmowy"
    corpus_train = args.corpus_dir / "train.txt"
    corpus_val = args.corpus_dir / "val.txt"
    dialogs_train = args.dialogs_dir / "train.txt"
    dialogs_val = args.dialogs_dir / "val.txt"

    corpus_sample = sample_file(corpus_train, int(args.sample_mb * 1_000_000))
    dialog_sample = sample_file(dialogs_train, int(args.dialog_sample_mb * 1_000_000))
    sample = corpus_sample + "\n" + dialog_sample
    print(
        f"probka: {len(corpus_sample)} znakow z korpusu + {len(dialog_sample)} znakow "
        f"z rozmow, trening BPE (vocab {args.vocab_size})"
    )

    tokenizer = BPETokenizer.train(sample, args.vocab_size)
    tokens_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = tokens_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    print(f"slownik: {tokenizer.vocab_size} tokenow -> {tokenizer_path}")

    block_chars = int(args.block_mb * 1_000_000)
    corpus_inputs = {"train": corpus_train, "val": corpus_val}
    corpus_counts: dict[str, tuple[int, int]] = {}
    for name, text_path in corpus_inputs.items():
        tokens, chars = encode_file(
            tokenizer, text_path, tokens_dir / f"{name}.bin", block_chars=block_chars
        )
        corpus_counts[name] = (tokens, chars)
        print(
            f"korpus {name}: {tokens} tokenow, {chars} znakow, "
            f"{chars / max(tokens, 1):.3f} znaku/token"
        )
    corpus_meta = TokenMeta(
        tokenizer.vocab_size,
        TOKEN_DTYPE,
        corpus_counts["train"][0],
        corpus_counts["train"][1],
        corpus_counts["val"][0],
        corpus_counts["val"][1],
    )
    save_meta(corpus_meta, tokens_dir / "meta.json")

    dialog_inputs = {"train": dialogs_train, "val": dialogs_val}
    dialog_counts: dict[str, tuple[int, int]] = {}
    for name, text_path in dialog_inputs.items():
        tokens, chars = encode_dialogs_file(
            tokenizer,
            text_path,
            dialogs_tokens_dir / f"{name}.bin",
            dialogs_tokens_dir / f"{name}_mask.bin",
        )
        dialog_counts[name] = (tokens, chars)
        masked = int(load_mask(dialogs_tokens_dir / f"{name}_mask.bin").sum())
        print(
            f"rozmowy {name}: {tokens} tokenow, z maska {masked} "
            f"({100 * masked / max(tokens, 1):.1f}%), {chars} znakow"
        )
    dialogs_meta = TokenMeta(
        tokenizer.vocab_size,
        TOKEN_DTYPE,
        dialog_counts["train"][0],
        dialog_counts["train"][1],
        dialog_counts["val"][0],
        dialog_counts["val"][1],
    )
    save_meta(dialogs_meta, dialogs_tokens_dir / "meta.json")

    print(f"czas: {time.perf_counter() - start:.1f} s")
    return corpus_meta, dialogs_meta


if __name__ == "__main__":
    main()
