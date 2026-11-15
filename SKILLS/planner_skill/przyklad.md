# Przykład: wypełniony prompt kodu

Ilustracja poziomu szczegółu, który wystarcza Haiku. Projekt i nazwy są zmyślone, liczy się tylko układ i dokładność sekcji. Ogólny szablon jest w `SKILL.md`.

# 02-eksport-csv

- **Trudność:** łatwe
- **Fala:** 2
- **Zależy od:** 01-eksporter-bazowy
- **Pliki (tylko te wolno zmieniać):** src/app/exporters.py
- **Kontrakt:** Docs/Kontrakty/exporters.md

## Cel
Eksport rekordów do pliku CSV, wybierany po nazwie formatu `"csv"`.

## Stan obecny
- **Pliki:** `src/app/exporters.py` istnieje (fala 1). Nowych plików brak.
- **Struktura `src/app/exporters.py`:**
  ```python
  from abc import ABC, abstractmethod
  from pathlib import Path

  class Exporter(ABC):
      extension: str
      def __init__(self, out_dir: Path) -> None: ...      # ustawia self.out_dir
      def target_path(self, name: str) -> Path: ...      # out_dir / f"{name}.{extension}"
      @abstractmethod
      def export(self, records: list[dict[str, str]], name: str) -> Path: ...

  class JsonExporter(Exporter): ...                     # extension = "json"

  EXPORTERS: dict[str, type[Exporter]] = {"json": JsonExporter}
  ```
- **Klasy:** `CsvExporter` dziedziczy po `Exporter`. Konstruktora nie nadpisuje, nadpisuje tylko `export`.
- **Do użycia:** `self.target_path(name) -> Path` z `Exporter`. Ścieżki nie składaj ręcznie.
- **Importy:** dopisz `import csv` na górze pliku. Wolno z: biblioteki standardowej. Nie wolno z: `app.cli`. Z `EXPORTERS` korzysta `src/app/cli.py` (`EXPORTERS[fmt](out_dir).export(records, name)`), więc nazwa i typ słownika zostają.
- **Współbieżność:** brak.
- **Dane i stan:** CSV w UTF-8, separator `,`, pierwszy wiersz to nagłówki w kolejności kluczy pierwszego rekordu.
- **Punkt wpięcia:** klasa pod `JsonExporter`, wpis `"csv": CsvExporter` w `EXPORTERS`.
- **Nie ruszać:** `Exporter`, `JsonExporter`.

## Ma być
```python
class CsvExporter(Exporter):
    """Zapisuje rekordy do pliku CSV w out_dir i zwraca jego ścieżkę.

    Przykład: export([{"a": "1", "b": "2"}], "x") -> out_dir/x.csv o treści "a,b\n1,2\n"
    Pusta lista -> pusty plik out_dir/x.csv
    """
    extension = "csv"

    def export(self, records: list[dict[str, str]], name: str) -> Path:
        ...

EXPORTERS: dict[str, type[Exporter]] = {"json": JsonExporter, "csv": CsvExporter}
```

## Kroki
1. Dopisz `import csv` do importów.
2. Pod `JsonExporter` dodaj klasę `CsvExporter` z `extension = "csv"`.
3. W `export`: `path = self.target_path(name)`, otwórz `path` w trybie `"w"` z `encoding="utf-8", newline=""`.
4. Gdy `records` jest pusta, zostaw plik pusty i zwróć `path`.
5. W przeciwnym razie: `csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")`, potem `writeheader()` i `writerows(records)`.
6. Zwróć `path`.
7. Dopisz `"csv": CsvExporter` do `EXPORTERS`.

## Uwaga
- **Moduł csv.** Na Windowsie puste linie między wierszami albo `\r\n` zamiast `\n` → plik otwarty bez `newline=""` i domyślny `lineterminator` → `open(..., newline="")` i `csv.writer(f, lineterminator="\n")`.
- **Puste wejście.** `IndexError` dla pustej listy → kod zakłada co najmniej jeden element → obsłuż pusty przypadek na początku funkcji, zgodnie z docstringiem.

## Gotowe, gdy
- `uv run python -c "import tempfile; from pathlib import Path; from app.exporters import EXPORTERS; p = EXPORTERS['csv'](Path(tempfile.mkdtemp())).export([{'a': '1', 'b': '2'}], 'x'); print(repr(p.read_text(encoding='utf-8')))"` wypisuje `'a,b\n1,2\n'`.

## Propozycja testów (dla testera)
- przykład z docstringu: treść pliku i zwrócona ścieżka `out_dir/x.csv`
- pusta lista daje pusty plik
- `EXPORTERS["csv"] is CsvExporter`, a `EXPORTERS["json"]` bez zmian
- wartość `"Zażółć"` zapisuje się i odczytuje w UTF-8 bez zmian
