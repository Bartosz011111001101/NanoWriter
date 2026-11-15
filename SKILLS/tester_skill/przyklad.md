# Przykład: specyfikacja, poprawka i weryfikacja

Ilustracja poziomu szczegółu. Projekt i nazwy są zmyślone (ten sam przykład co w `SKILLS/planner_skill/przyklad.md`), liczy się tylko układ i dokładność. Ogólne szablony są w `SKILL.md`.

## Specyfikacja testów

```markdown
# 02-eksport-csv (testy)

- **Trudność:** łatwe
- **Model:** sonnet
- **Plik testów (tylko ten wolno zmieniać):** tests/test_exporters_02_eksport_csv.py
- **Wspólne fixture'y:** nie zmieniaj
- **Prompt kodu:** Docs/Prompts/05-eksport/kod/02-eksport-csv.md
- **Kontrakt:** Docs/Kontrakty/exporters.md
- **Ocena propozycji plannera:** uzupełnione: dodano test, że JsonExporter został w rejestrze

## Przygotowanie
- Import: `from app.exporters import EXPORTERS, CsvExporter, JsonExporter`
- Obiekty: `CsvExporter` z katalogiem tymczasowym jako `out_dir`.
- Zasoby: katalog tymczasowy.

## Testy
### test_csv_export_example
- Przypadek: przykład z docstringu.
- `export([{"a": "1", "b": "2"}], "x")` → zwraca `out_dir / "x.csv"`, plik w UTF-8 ma treść dokładnie `"a,b\n1,2\n"`

### test_csv_export_empty
- Przypadek: pusta lista rekordów.
- `export([], "x")` → plik `out_dir / "x.csv"` istnieje i jest pusty

### test_csv_registered
- Przypadek: rejestr eksporterów.
- `EXPORTERS["csv"]` → `CsvExporter`, `EXPORTERS["json"]` → `JsonExporter` (bez zmian)
```

## Poprawka kodu

```markdown
# 02-eksport-csv, poprawka 1

- **Trudność:** łatwe
- **Pliki (tylko te wolno zmieniać):** src/app/exporters.py
- **Prompt pierwotny:** Docs/Prompts/05-eksport/kod/02-eksport-csv.md

## Błąd
- Test: `tests/test_exporters_02_eksport_csv.py::test_csv_export_empty`
- Wynik: `IndexError: list index out of range` dla `export([], "x")`
- Oczekiwane według docstringu: pusty plik
- Hipoteza (pułapka „Puste wejście”): `records[0]` jest czytane przed sprawdzeniem pustej listy.

## Stan obecny
- **Struktura:** `CsvExporter.export` tworzy `DictWriter` z `fieldnames=list(records[0])` w pierwszej linii po `open`.
- **Nie ruszać:** reszta pliku.

## Gotowe, gdy
- `uv run pytest tests/test_exporters_02_eksport_csv.py` przechodzi.
```

## Weryfikacja

```markdown
# Weryfikacja: zadanie 05, fala 2

## Runda 1
- **Testy:** `uv run pytest` → 14 passed, 2 failed
- **Sprawdzenie statyczne:** przechodzi

| Test | Przyczyna | Błąd w | Działanie |
| --- | --- | --- | --- |
| test_csv_export_empty | IndexError dla [] | kodzie | poprawki/02-eksport-csv-1.md |
| test_csv_registered | zły import | teście | poprawione, przechodzi |

- **Nowe pułapki:** brak
- **Problemy z kontraktem:** brak
- **Werdykt:** niegotowe
```
