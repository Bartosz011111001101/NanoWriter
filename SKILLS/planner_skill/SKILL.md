---
name: planner_skill
description: Używaj, gdy zadanie trzeba rozpisać na fale i prompty kodu dla wykonawców (Haiku, Sonnet, Opus), ocenić trudność promptu, opisać wykonawcy stan obecny kodu albo napisać lub zmienić kontrakt modułu.
---

# Planowanie

Prompt kodu daje wykonawcy całą wiedzę potrzebną do pracy: co jest teraz w kodzie, co ma powstać, gdzie to wpiąć i na co uważać. Wykonawca nie przeszukuje repozytorium i nie projektuje niczego, czego prompt mu wprost nie zleca. Dobrze rozpisany prompt zrobi Haiku, nawet jeśli to cała klasa z dziedziczeniem albo trudny algorytm.

## Na start czytasz

1. `SKILLS/projekt/profil.md`: stack, komendy, układ repozytorium, mapa modułów, etapy, decyzje użytkownika.
2. Pliki pułapek wymienione w profilu (`SKILLS/projekt/pulapki/`).
3. Plan wstępny z profilu, kontrakty z `Docs/Kontrakty/` i raporty z `Docs/Postep/` poprzednich zadań.
4. **Aktualny kod każdego pliku, który zadanie zmienia.** Z niego, a nie z pamięci ani z kontraktu, piszesz sekcję „Stan obecny”.

`Docs/` jest w `.gitignore`, więc Grep i Glob mogą go pomijać. Listuj przez `ls`, czytaj po pełnej ścieżce.

## Rozpoznanie kodu: gdzie to umieścić

Zanim rozpiszesz prompty, ustal dla każdej rzeczy, której zadanie potrzebuje (funkcja, klasa, stała, typ), czy już istnieje i gdzie ma żyć.

1. Uruchom „Mapę kodu” z profilu. Dostajesz wszystkie klasy, funkcje i metody publiczne projektu.
2. Dla każdej potrzebnej rzeczy uruchom „Szukanie istniejącego kodu” z profilu. Szukaj po nazwie i po 2–3 słowach opisujących, co robi (np. `path|ścieżk|katalog`).
3. Rozstrzygnij według tabeli. Bierz pierwszy pasujący wiersz:

| Co znalazłeś | Decyzja | W prompcie |
| --- | --- | --- |
| Istnieje i robi dokładnie to | Używasz istniejącego. | „Do użycia” z sygnaturą i importem. |
| Istnieje i robi prawie to | Rozszerzasz istniejące (nowy argument z wartością domyślną, nowa metoda). Nowej, podobnej funkcji nie tworzysz. | Prompt zmienia tę funkcję: „Było” i „Ma być”. |
| Potrzebują tego co najmniej dwa moduły | Jedno miejsce: najniższa warstwa, z której oba mogą importować, we wcześniejszej fali. | Wcześniejszy prompt to tworzy, późniejsze mają to w „Do użycia”. |
| Nie istnieje, pasuje do „Odpowiada za” któregoś modułu z profilu | Nowy kod w tym module. | Zwykły prompt. |
| Nie istnieje i nie pasuje do żadnego modułu | Nie tworzysz nowego modułu. | Wpis w „Otwartych pytaniach”: proponowany moduł i jego warstwa. |

4. Sprawdź kierunek każdego importu z warstwami z profilu. Import z wyższej warstwy oznacza, że kod jest w złym miejscu: przenieś wspólny kawałek niżej, zamiast importować w górę.
5. Nowa biblioteka zewnętrzna tylko wtedy, gdy nie wystarczy ani biblioteka standardowa, ani zależności już zainstalowane. Taka biblioteka zawsze trafia do „Otwartych pytań”.

## Co tworzysz

```
Docs/Plan/NN-zadanie/plan.md
Docs/Prompts/NN-zadanie/kod/PP-nazwa.md      jeden plik na kawałek pracy
Docs/Kontrakty/<modul>.md                    nowe albo zmienione kontrakty
```

Promptów testów nie piszesz, robi to tester. W każdym prompcie kodu zostawiasz sekcję „Propozycja testów”.

## Cięcie na prompty i fale

- Jeden prompt to jeden spójny kawałek: kilka funkcji albo klasa z metodami.
- **Fala** to prompty, które idą równolegle. Prompty w jednej fali:
  - nie zmieniają tych samych plików;
  - nie zależą od siebie (żaden nie używa tego, co pisze inny prompt tej samej fali).
- Prompt korzystający z wyniku wcześniejszego promptu idzie do późniejszej fali.
- **Jak najmniej fal.** Każda fala to pełna runda testów i weryfikacji. Układaj fale tak, żeby w każdej pracowało kilka modułów naraz, zamiast dawać jeden moduł na falę.
- Gdy jeden plik dostaje w zadaniu więcej niż 3 prompty, zaproponuj podział modułu na mniejsze pliki. Jeśli moduł ma już kod, wpisz to do „Otwartych pytań”.
- Najpierw najprostsza działająca wersja, optymalizacje w późniejszych falach.
- Ostatnia fala zawiera test bramki z profilu, jeśli zadanie zamyka etap.

## Trudność

Trudność zależy od zakresu zmian i od tego, czy umiesz rozwiązanie rozpisać z góry. Złożoność samego algorytmu jej nie podnosi. Sprawdzaj wiersze od góry i bierz pierwszy pasujący:

| Trudność | Model kodu | Warunek |
| --- | --- | --- |
| trudne | Opus | Nie umiesz rozpisać rozwiązania z góry, bo trzeba je zaprojektować albo zbadać, **albo** prompt zmienia więcej niż 5 plików. |
| średnie | Sonnet | Prompt zmienia 3–5 plików **albo** przerabia istniejącą logikę, z której korzystają inne moduły, i zmiany muszą się zgadzać w kilku miejscach. |
| łatwe | Haiku | Do 2 plików, a rozwiązanie jest rozpisane w całości w sekcjach „Stan obecny”, „Ma być” i „Kroki”. |

Testy nigdy nie idą na Haiku. Model wykonawcy testów dobiera tester (minimum Sonnet).

Wahasz się między łatwe a średnie? Rozpisz dokładniej „Stan obecny” i „Kroki” albo podziel prompt. „Trudne” to wyjątek.

## Szablon: prompt kodu

Sekcje zawsze w tej kolejności i zawsze wszystkie, niezależnie od trudności. Pole, które nie dotyczy promptu, ma wartość „brak”.

| Sekcja | Zawartość |
| --- | --- |
| Nagłówek | Trudność, Fala, Zależy od, Pliki (tylko te wolno zmieniać), Kontrakt |
| Cel | Jedno zdanie: co powstaje i po co. |
| Stan obecny | **Pliki** (które istnieją, które są nowe). **Struktura** każdego zmienianego pliku: importy, klasy, funkcje, sygnatury, zarys bez ciał. **Klasy:** po czym dziedziczy, konstruktor, pola, które metody nadpisać, co wywołać z `super()`. **Do użycia:** istniejące funkcje, klasy i stałe z całego projektu, które wykonawca ma wywołać zamiast pisać własne, z sygnaturą i importem (wynik „Rozpoznania kodu”). **Importy:** dokładne linie do dopisania, „wolno z” i „nie wolno z” (moduły według warstw) oraz kto korzysta z tego kodu. **Współbieżność:** workery, wątki, procesy, async; co działa gdzie, jak się komunikuje, start i stop. **Dane i stan:** formaty plików, konfiguracja, zmienne środowiska. **Punkt wpięcia:** gdzie dokładnie dopisać. **Nie ruszać:** co zostaje bez zmian. |
| Ma być | Szkielet docelowy: pełne sygnatury z typami, docstring z przykładem wejście → wyjście. Zmiana istniejącej funkcji: „Było” (zarys) i „Ma być” (pełny szkielet). |
| Kroki | Łatwe: każdy krok mechaniczny, w kolejności. Średnie: kroki dla każdej funkcji. Trudne: kolejność pracy i sposób sprawdzenia poprawności. |
| Uwaga | Wpisy z plików pułapek, które dotyczą tego promptu, skopiowane w całości. |
| Gotowe, gdy | Komenda sprawdzająca i oczekiwany wynik (zasady niżej). |
| Propozycja testów | Lista przypadków dla testera z dokładnymi wejściami i wynikami. |

**Prompt trudny.** W „Ma być” ustalasz granice: publiczne sygnatury, niezmienniki i to, co ma przejść. Środek oznaczasz jako „Do zaprojektowania przez wykonawcę: …”. Wykonawca opisuje swój projekt w raporcie, a Ty bierzesz go do promptów następnych fal.

**Gotowe, gdy.** Jedna linia, komenda z profilu. Działa w kilka sekund na CPU, bez pobierania danych i bez zapisu poza katalogiem tymczasowym. W kodzie wewnątrz `-c "..."` tylko ASCII i cudzysłowy pojedyncze. Nigdy nie wymaga testów, bo wykonawca kodu ich nie pisze.

### Przykład promptu

````markdown
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
````

## Szablon: plan.md

```markdown
# Zadanie NN: <nazwa>

## Cel
<jedno zdanie>. Gotowe, gdy: <coś do uruchomienia i zobaczenia>.

## Fale
| Fala | Prompt | Trudność | Pliki |
| --- | --- | --- | --- |
| 1 | 01-eksporter-bazowy | łatwe | src/app/exporters.py |
| 1 | 02-konfiguracja | łatwe | src/app/config.py |
| 2 | 03-eksport-csv | łatwe | src/app/exporters.py |

## Kontrakty
- utworzone/zmienione: Docs/Kontrakty/exporters.md

## Otwarte pytania
- <decyzja dla użytkownika> albo "brak"
```

## Szablon: kontrakt

Plik `Docs/Kontrakty/<modul>.md`, sekcje:

- **Odpowiedzialność**
- **Publiczne API:** sygnatury z typami, klasy z dziedziczeniem.
- **Struktury danych:** kształty, typy i formaty (np. tensor `idx: LongTensor[B, T]`, rekord `dict[str, str]`).
- **Pliki:** formaty na dysku.
- **Niezmienniki:** co tester ma sprawdzić.
- **Zmiany:** data i opis.

Szkielety w promptach muszą być zgodne z kontraktem. Każda zmiana kontraktu to wpis w „Zmiany”.

## Przed oddaniem planu

Sprawdź każdy punkt. Jeśli któryś nie przechodzi, popraw prompt:

- [ ] „Stan obecny” każdego promptu jest przepisany z aktualnego kodu albo ze szkieletu promptu wcześniejszej fali, jeśli kod jeszcze nie istnieje.
- [ ] Każda nowa funkcja i klasa przeszła „Rozpoznanie kodu”: nic podobnego nie istnieje, a jeśli istnieje, jest w „Do użycia” albo prompt je rozszerza.
- [ ] Żadna logika nie jest zaplanowana w dwóch promptach naraz.
- [ ] Każdy import w „Importach” idzie z wyższej warstwy do niższej.
- [ ] Każdy przykład w docstringu jest policzony ręcznie i się zgadza, łącznie z białymi znakami i typem wyniku.
- [ ] Żadne dwa prompty jednej fali nie mają wspólnego pliku w „Plikach”.
- [ ] Wszystko, czego prompt używa, powstaje we wcześniejszej fali albo już istnieje.
- [ ] Szkielety zgadzają się z kontraktem.
- [ ] Trudność wynika z tabeli, czyli liczby plików i tego, czy rozwiązanie jest rozpisane.
- [ ] Komenda „Gotowe, gdy” spełnia zasady.

## Odpowiedź dla orkiestratora

1. Ścieżka planu i lista fal (numer fali, nazwy promptów, trudności).
2. Utworzone i zmienione kontrakty.
3. Otwarte pytania albo „brak”.

## Częste błędy

| Błąd | Poprawka |
| --- | --- |
| „Stan obecny” napisany z pamięci albo z kontraktu | Przeczytaj plik i przepisz zarys z kodu. Wykonawca zablokuje się na pierwszej niezgodności. |
| Nowa funkcja pomocnicza, choć podobna już istnieje | Uruchom „Szukanie istniejącego kodu” i wpisz znalezioną do „Do użycia”. |
| Ta sama logika w dwóch promptach różnych modułów | Jeden wcześniejszy prompt w najniższej wspólnej warstwie, w pozostałych „Do użycia”. |
| Import z wyższej warstwy „bo tam już jest” | Przenieś ten kawałek do niższej warstwy w osobnym, wcześniejszym prompcie. |
| Szkielet bez typów albo bez przykładu w docstringu | Dopisz typy i przykład wejście → wyjście. |
| Klasa bez informacji, po czym dziedziczy i co wywołać z `super()` | Uzupełnij pole „Klasy” w „Stanie obecnym”. |
| Trudność podniesiona, bo algorytm jest trudny, choć kroki go opisują | Zostaw „łatwe”. Trudność to zakres i to, czy rozwiązanie jest rozpisane. |
| Jeden moduł na falę, przez co fal jest dużo | Połącz w jednej fali prompty z różnych modułów. |
| Dwa prompty jednej fali zmieniają ten sam plik | Przenieś jeden do następnej fali. |
| „Gotowe, gdy” wymaga testów albo trwa minuty | Daj jednolinijkową komendę z profilu, działającą w kilka sekund. |
| Decyzja użytkownika podjęta przez plannera | Przenieś ją do „Otwartych pytań”. |
