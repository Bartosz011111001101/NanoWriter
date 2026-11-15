---
name: tester_skill
description: Używaj, gdy trzeba ocenić propozycje testów od plannera i napisać specyfikacje testów dla wykonawców (tryb SPEC), albo uruchomić testy fali, rozstrzygnąć, czy błąd jest w kodzie, czy w teście, i napisać poprawki (tryb WERYFIKACJA).
---

# Testowanie

Tester projektuje testy i ocenia wyniki. Testów nie pisze sam, tylko odpala do tego agentów `wykonawca-testow`. Kodu nie zmienia nigdy.

## Na start czytasz

- `SKILLS/projekt/profil.md`: komendy testów, układ repozytorium, nazwy plików testów.
- `SKILLS/projekt/katalog-testow.md`: testy obowiązkowe dla tej dziedziny.
- Pliki pułapek wymienione w profilu (`SKILLS/projekt/pulapki/`).

## Pliki

```
czytasz:  Docs/Plan/NN-zadanie/plan.md
          Docs/Prompts/NN-zadanie/kod/PP-nazwa.md          szkielet i "Propozycja testów"
          Docs/Kontrakty/<modul>.md
piszesz:  Docs/Prompts/NN-zadanie/testy/PP-nazwa.md        specyfikacja testów
          Docs/Prompts/NN-zadanie/poprawki/PP-nazwa-K.md   poprawka kodu
          Docs/Postep/NN-zadanie/weryfikacja-fala-F.md
          SKILLS/projekt/pulapki/<technologia>.md          nowe pułapki
```

`Docs/` jest w `.gitignore`. Listuj przez `ls`, czytaj po pełnej ścieżce.

Każdy prompt kodu ma własny plik testów (nazwa według profilu), więc równolegli wykonawcy testów się nie nadpisują. Wspólne fixture'y zmienia w fali tylko jedna specyfikacja, oznaczona w nagłówku.

## Model wykonawcy testów

Testy nigdy nie idą na Haiku.

| Trudność promptu kodu | Model wykonawcy testów |
| --- | --- |
| łatwe | `sonnet` |
| średnie | `sonnet` |
| trudne | `opus` |

Poprawka tego samego pliku testów w kolejnej rundzie idzie na `opus`.

## Tryb SPEC

Dla każdego promptu kodu z fali F:

1. Oceń „Propozycję testów” plannera według listy niżej. Brakujące testy dopisz.
2. Zapisz specyfikację w `Docs/Prompts/NN-zadanie/testy/PP-nazwa.md` (szablon niżej).
3. Odpal agenta `wykonawca-testow` dla każdej specyfikacji, równolegle, z modelem według tabeli wyżej. Zlecenie: „Wykonaj `Docs/Prompts/NN-zadanie/testy/PP-nazwa.md`.”
4. Odpowiedź dla orkiestratora: lista plików testów i wyniki z `Docs/Postep/NN-zadanie/testy/`.

### Czy propozycja wystarcza

- Każdy niezmiennik z kontraktu modułu ma test.
- Każdy przykład z docstringu w szkielecie ma test.
- Przypadki brzegowe: puste wejście, jeden element, wartość na granicy zakresu.
- Typ zwracanej wartości, a także kształt i format struktur danych z kontraktu.
- Wyjątki, jeśli kontrakt je przewiduje.
- Klasy: podklasa działa tam, gdzie kod używa klasy bazowej, a rejestry i fabryki zwracają właściwą klasę.
- Pozycje z katalogu testów projektu, które dotyczą tego modułu.

### Szablon specyfikacji testów

Każdy test ma dokładne wejście i dokładny oczekiwany wynik. Wykonawca niczego nie wymyśla. Sekcja „Przygotowanie” mówi, jak zbudować obiekty, żeby wykonawca nie musiał czytać kodu.

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

## Tryb WERYFIKACJA

1. Użyj `superpowers:verification-before-completion`. Uruchom pełny zestaw testów komendą z profilu.
2. Dla każdego nieprzechodzącego testu rozstrzygnij, gdzie jest błąd:

| Sytuacja | Błąd w | Działanie |
| --- | --- | --- |
| Test sprawdza coś innego, niż mówi specyfikacja, albo ma literówkę lub zły import | teście | Odpal `wykonawca-testow` (model według tabeli „Model wykonawcy testów”) z poprawioną specyfikacją, uruchom testy ponownie. |
| Specyfikacja testu przeczy szkieletowi albo kontraktowi | specyfikacji | Popraw specyfikację, potem jak wyżej. |
| Kod łamie szkielet, docstring albo kontrakt | kodzie | Napisz `Docs/Prompts/NN-zadanie/poprawki/PP-nazwa-K.md`. Kodu nie poprawiasz. |
| Szkielet i kontrakt są ze sobą sprzeczne albo niejasne | kontrakcie | Zgłoś to orkiestratorowi. Rozstrzyga planner. |

3. Hipotezę przyczyny błędu w kodzie zacznij od plików pułapek z profilu.
4. **Nowa pułapka.** Jeśli przyczyna błędu w kodzie może się powtórzyć w innym miejscu, a w plikach pułapek jej nie ma, dopisz wpis do pliku właściwej technologii w formacie tego pliku. Błąd jednorazowy, np. literówka, nie trafia do pułapek.
5. Zapisz `Docs/Postep/NN-zadanie/weryfikacja-fala-F.md` (szablon niżej). Kolejne rundy dopisuj w tym samym pliku jako nową sekcję.

### Szablon poprawki kodu

Ten sam układ co prompt kodu (z „Stanem obecnym” opisującym kod po nieudanej rundzie), plus sekcja „Błąd” na początku:

````markdown
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
````

Wykonawca poprawki może uruchamiać testy, ale nie może ich zmieniać.

### Szablon weryfikacji

```markdown
# Weryfikacja: zadanie 05, fala 2

## Runda 1
- **Komenda:** `uv run pytest` → 14 passed, 2 failed

| Test | Przyczyna | Błąd w | Działanie |
| --- | --- | --- | --- |
| test_csv_export_empty | IndexError dla [] | kodzie | poprawki/02-eksport-csv-1.md |
| test_csv_registered | zły import | teście | poprawione, przechodzi |

- **Nowe pułapki:** brak | pulapki/python.md: „Puste wejście”
- **Problemy z kontraktem:** brak
- **Werdykt:** niegotowe
```

Odpowiedź dla orkiestratora: werdykt, ścieżki nowych plików w `poprawki/`, dopisane pułapki albo „brak”, problemy z kontraktem albo „brak”.
