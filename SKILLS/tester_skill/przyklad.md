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
- Obiekt: `CsvExporter(tmp_path)`, gdzie `tmp_path` to fixture pytest.
- Dostępne fixture'y: tylko wbudowane w pytest.

## Testy
### test_csv_export_example
- Wejście: `CsvExporter(tmp_path).export([{"a": "1", "b": "2"}], "x")`
- Oczekiwane: zwraca `tmp_path / "x.csv"`, a `read_text(encoding="utf-8") == "a,b\n1,2\n"`

### test_csv_export_empty
- Wejście: `CsvExporter(tmp_path).export([], "x")`
- Oczekiwane: `read_text(encoding="utf-8") == ""`

### test_csv_registered
- Wejście: `EXPORTERS`
- Oczekiwane: `EXPORTERS["csv"] is CsvExporter` i `EXPORTERS["json"] is JsonExporter`
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
