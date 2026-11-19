"""Etap 2: pobranie prozy z Wolnych Lektur, czyszczenie i podział train/val po książkach.

Użycie (z korzenia repo):
    uv run python scripts/prepare_data.py [--target-mb 28] [--max-download-mb 200] [--pause 0.5]
"""

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from nanowriter.data import (
    PROSE_EPOCHS,
    LimitedFetch,
    build_corpus,
    list_books,
    select_books,
    split_corpus,
)

if TYPE_CHECKING:
    from nanowriter.data import SplitResult


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń; argv=None oznacza sys.argv[1:].

    Przykład: parse_args([]) -> data_dir=Path("data"), epochs=list(PROSE_EPOCHS), target_mb=28.0,
        max_download_mb=200.0, pause=0.5, seed=0, val_fraction=0.1, translations=False
    Przykład: parse_args(["--epochs", "pozytywizm", "--translations"])
        -> epochs=["pozytywizm"], translations=True
    """
    parser = argparse.ArgumentParser(
        description="Pobranie prozy z Wolnych Lektur, czyszczenie i podział train/val."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="katalog danych (raw/, clean/, train.txt, val.txt)",
    )
    parser.add_argument(
        "--epochs", type=str, nargs="+", default=list(PROSE_EPOCHS),
        help="epoki z Wolnych Lektur",
    )
    parser.add_argument(
        "--target-mb", type=float, default=28.0,
        help="docelowy rozmiar czystego tekstu w MB",
    )
    parser.add_argument(
        "--max-download-mb", type=float, default=200.0,
        help="twardy limit pobierania w MB",
    )
    parser.add_argument(
        "--pause", type=float, default=0.5, help="pauza między pobraniami w sekundach"
    )
    parser.add_argument("--seed", type=int, default=0, help="ziarno wyboru książek i podziału")
    parser.add_argument(
        "--val-fraction", type=float, default=0.1, help="udział walidacji (w bajtach)"
    )
    parser.add_argument(
        "--translations", action=argparse.BooleanOptionalAction, default=False,
        help="czy brać przekłady",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> "SplitResult":
    """Buduje korpus z Wolnych Lektur, dzieli go na train/val, wypisuje postęp
    i zwraca wynik podziału.

    Przykład: main(["--data-dir", d, "--epochs", "pozytywizm", "--pause", "0"]) przy fałszywym
        _fetch_url z 4 nowelami a, b, c, d -> SplitResult(train=["a", "b", "d"], val=["c"], ...)
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args(argv)
    fetcher = LimitedFetch(int(args.max_download_mb * 1_000_000), pause=args.pause)

    entries = [
        entry for epoch in args.epochs for entry in list_books(epoch, fetch=fetcher)
    ]
    selected = select_books(entries, seed=args.seed)
    print(f"katalog: {len(entries)} pozycji, wybrane: {len(selected)}")

    def on_book(slug: str, status: str) -> None:
        print(f"{status:<12} {slug}")

    report = build_corpus(
        selected,
        args.data_dir / "raw",
        args.data_dir / "clean",
        int(args.target_mb * 1_000_000),
        include_translations=args.translations,
        fetch=fetcher,
        on_book=on_book,
    )
    print(
        f"ksiazki: {len(report.books)}, czysty tekst: {report.clean_bytes / 1e6:.1f} MB, "
        f"pobrano: {fetcher.used / 1e6:.1f} MB, pominiete: {len(report.skipped)}"
    )
    if report.limit_reached:
        print("uwaga: osiagnieto limit pobierania")

    result = split_corpus(
        args.data_dir / "clean",
        report.books,
        args.data_dir,
        args.val_fraction,
        args.seed,
    )
    print(
        f"train: {len(result.train)} ksiazek, {result.train_chars} znakow -> {result.train_path}"
    )
    print(f"val: {len(result.val)} ksiazek, {result.val_chars} znakow -> {result.val_path}")
    return result


if __name__ == "__main__":
    main()
