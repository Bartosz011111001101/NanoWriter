"""Rozmówca, krok 2: wczytanie, walidacja i podział rozmów z data/dialogi/surowe.

Użycie (z korzenia repo):
    uv run python scripts/prepare_dialogs.py [--src data/dialogi/surowe] [--out data/rozmowy]
"""

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from nanowriter.data import build_dialog_corpus

if TYPE_CHECKING:
    from nanowriter.data import DialogCorpusResult


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń; argv=None oznacza sys.argv[1:].

    Przykład: parse_args([]) -> src=Path("data/dialogi/surowe"), out=Path("data/rozmowy"),
        val_fraction=0.05, seed=0, min_lines=4, max_lines=16
    Przykład: parse_args(["--val-fraction", "0.1", "--max-lines", "20"])
        -> val_fraction=0.1, max_lines=20
    """
    parser = argparse.ArgumentParser(
        description="Wczytanie, walidacja i podział rozmów z data/dialogi/surowe."
    )
    parser.add_argument(
        "--src", type=Path, default=Path("data/dialogi/surowe"),
        help="katalog z rozmowami NNN.txt",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/rozmowy"),
        help="katalog wyjściowy (train.txt, val.txt, report.json)",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.05, help="udział walidacji (w rozmowach)"
    )
    parser.add_argument("--seed", type=int, default=0, help="ziarno podziału")
    parser.add_argument(
        "--min-lines", type=int, default=4, help="najmniejsza liczba linii rozmowy"
    )
    parser.add_argument(
        "--max-lines", type=int, default=16, help="największa liczba linii rozmowy"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> "DialogCorpusResult":
    """Buduje korpus rozmów, wypisuje podsumowanie i zwraca wynik.

    Przykład: main(["--src", s, "--out", o, "--val-fraction", "0.25"]) przy s z plikiem 001.txt
        z 4 różnymi poprawnymi rozmowami o tej samej długości -> DialogCorpusResult(train=3,
        val=1, ...) i pliki train.txt, val.txt, report.json w o
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args(argv)
    wynik = build_dialog_corpus(
        args.src, args.out, args.val_fraction, args.seed, args.min_lines, args.max_lines
    )
    report = wynik.report

    print(f"pliki: {report.files}")
    print(f"znalezione rozmowy: {report.found}")
    print(f"przyjete: {report.accepted}")
    print(f"odrzucone: {len(report.rejected)}")
    for zrodlo, powod in list(report.rejected.items())[:20]:
        print(f"  {zrodlo}: {powod}")
    reszta = len(report.rejected) - 20
    if reszta > 0:
        print(f"  ... i {reszta} kolejnych (patrz report.json)")
    print(f"duplikaty: {len(report.duplicates)}")
    print(f"train: {wynik.train} rozmow -> {wynik.train_path}")
    print(f"val: {wynik.val} rozmow -> {wynik.val_path}")
    return wynik


if __name__ == "__main__":
    main()
