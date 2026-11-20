"""Rozmówca, krok 1: pobranie początku napisów OPUS i budowa korpusu treningu wstępnego.

Użycie (z korzenia repo):
    uv run python scripts/prepare_pretrain.py --check-size
    uv run python scripts/prepare_pretrain.py [--max-download-mb 300] [--target-mb 600]
"""

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from nanowriter.data import (
    OPUS_SUBTITLES_PL_URL,
    build_pretrain_corpus,
    download_prefix,
    remote_size,
)

if TYPE_CHECKING:
    from nanowriter.data import PretrainReport


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń; argv=None oznacza sys.argv[1:].

    Przykład: parse_args([]) -> data_dir=Path("data"), url=OPUS_SUBTITLES_PL_URL,
        check_size=False, max_download_mb=300.0, target_mb=600.0, block_lines=20000,
        val_every=50, book_val_fraction=0.1, seed=0, allow_large=False
    Przykład: parse_args(["--check-size", "--target-mb", "10"]) -> check_size=True, target_mb=10.0
    Przykład: parse_args(["--max-download-mb", "2000"]) -> SystemExit (ponad 1000 MB wymaga
        --allow-large)
    Przykład: parse_args(["--max-download-mb", "2000", "--allow-large"]) -> max_download_mb=2000.0
    """
    parser = argparse.ArgumentParser(
        description="Rozmowca, krok 1: pobranie poczatku napisow OPUS i korpus treningu wstepnego."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="katalog danych (opus/, clean/, pretrain/)",
    )
    parser.add_argument(
        "--url", type=str, default=OPUS_SUBTITLES_PL_URL,
        help="adres pliku pl.txt.gz z napisami OPUS",
    )
    parser.add_argument(
        "--check-size", action="store_true",
        help="tylko podaj rozmiar pliku, bez pobierania",
    )
    parser.add_argument(
        "--max-download-mb", type=float, default=300.0,
        help="twardy limit pobierania w MB",
    )
    parser.add_argument(
        "--target-mb", type=float, default=600.0,
        help="docelowy rozmiar przyjetych napisow w MB",
    )
    parser.add_argument(
        "--block-lines", type=int, default=20_000, help="liczba linii w jednym bloku"
    )
    parser.add_argument(
        "--val-every", type=int, default=50, help="co ile blokow jeden idzie do walidacji"
    )
    parser.add_argument(
        "--book-val-fraction", type=float, default=0.1, help="udzial ksiazek walidacyjnych"
    )
    parser.add_argument("--seed", type=int, default=0, help="ziarno podzialu ksiazek")
    parser.add_argument(
        "--allow-large", action="store_true",
        help="zgoda na limit pobierania powyzej 1000 MB",
    )
    args = parser.parse_args(argv)
    if args.max_download_mb > 1000 and not args.allow_large:
        parser.error("pobranie powyzej 1000 MB wymaga --allow-large")
    return args


def main(argv: list[str] | None = None) -> "PretrainReport | None":
    """Podaje rozmiar pliku (--check-size) albo pobiera prefiks i buduje korpus; zwraca raport.

    Przykład: main(["--check-size"]) przy _head_url zwracającym 5_000_000 -> wypisuje
        "rozmiar pliku: 5 MB" i "limit pobierania: 300 MB", zwraca None, nic nie tworzy na dysku
    Przykład: main(["--data-dir", d]) przy podmienionym _open_url (strumień małego pliku .gz)
        i katalogu d/clean z co najmniej dwiema książkami -> tworzy d/opus/pl.txt.gz,
        d/pretrain/train.txt, val.txt, report.json i zwraca PretrainReport
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args(argv)
    if args.check_size:
        rozmiar = remote_size(args.url)
        print(f"rozmiar pliku: {rozmiar / 1e6:.0f} MB")
        print(f"limit pobierania: {args.max_download_mb:.0f} MB")
        return None

    gz = args.data_dir / "opus" / "pl.txt.gz"
    if gz.exists():
        print(f"plik istnieje, pomijam pobieranie: {gz}")
    else:
        pobrano = download_prefix(args.url, gz, int(args.max_download_mb * 1_000_000))
        print(f"pobrano: {pobrano / 1e6:.1f} MB -> {gz}")

    report = build_pretrain_corpus(
        gz,
        args.data_dir / "clean",
        args.data_dir / "pretrain",
        int(args.target_mb * 1_000_000),
        block_lines=args.block_lines,
        val_every=args.val_every,
        book_val_fraction=args.book_val_fraction,
        seed=args.seed,
    )
    print(f"przeczytano linii napisow: {report.lines_read}")
    print(
        f"napisy: train {report.train_lines} linii, val {report.val_lines} linii, "
        f"razem {report.subtitle_bytes / 1e6:.1f} MB"
    )
    print(f"ksiazki: train {len(report.train_books)}, val {len(report.val_books)}")
    print(
        f"pliki: train {report.train_bytes / 1e6:.1f} MB, "
        f"val {report.val_bytes / 1e6:.1f} MB -> {args.data_dir / 'pretrain'}"
    )
    if report.rejected:
        powody = sorted(report.rejected.items(), key=lambda para: (-para[1], para[0]))
        opis = ", ".join(f"{powod}={liczba}" for powod, liczba in powody)
        print(f"odrzucone: {opis}")
    else:
        print("odrzucone: brak")
    if not report.target_reached:
        print(
            "uwaga: nie osiagnieto celu --target-mb; zwieksz --max-download-mb "
            f"i usun plik {gz}, zeby pobrac wiecej"
        )
    return report


if __name__ == "__main__":
    main()
