---
name: tester_skill
description: Używaj, gdy trzeba ocenić propozycje testów od plannera i napisać specyfikacje testów dla wykonawców (tryb SPEC), albo uruchomić testy fali, rozstrzygnąć, czy błąd jest w kodzie, czy w teście, i napisać poprawki (tryb WERYFIKACJA).
---

# Testowanie

Tester projektuje testy i ocenia wyniki. Testów nie pisze sam, tylko odpala do tego agentów `wykonawca-testow`. Kodu nie zmienia nigdy.

## Na start czytasz

- `SKILLS/projekt/profil.md`: komendy testów, układ repozytorium, nazwy plików testów.
- `SKILLS/projekt/konwencje.md`, sekcja „Testy”: czym jest wspólne przygotowanie testów i jakie zasady obowiązują w testach.
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

Każdy prompt kodu ma własny plik testów (nazwa według profilu), więc równolegli wykonawcy testów się nie nadpisują. Plik testów jest zawsze osobnym plikiem, innym niż pliki kodu, także w języku, w którym testy zwykle siedzą w pliku kodu. Wspólne przygotowanie testów (dane, obiekty, zasoby używane przez wiele plików testów) zmienia w fali tylko jedna specyfikacja, oznaczona w nagłówku.

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
- Każdy przykład z opisu funkcji w szkielecie ma test.
- Przypadki brzegowe: puste wejście, jeden element, wartość na granicy zakresu.
- Typ zwracanej wartości, a także kształt i format struktur danych z kontraktu.
- Błędy (wyjątki albo kody błędów), jeśli kontrakt je przewiduje.
- Typy: typ pochodny albo implementacja działa tam, gdzie kod używa typu bazowego albo interfejsu, a rejestry i fabryki zwracają właściwy typ.
- Każdy przypadek z „Kroków” promptu kodu (wejście albo warunek → efekt) ma test.
- Pozycje z katalogu testów projektu, które dotyczą tego modułu.

### Szablon specyfikacji testów

**Ty projektujesz test, wykonawca go implementuje.** Projekt to przypadek (co sprawdzasz), dokładne dane wejściowe i dokładny oczekiwany wynik. Implementacja, czyli asercje, przygotowanie danych, parametryzacja i sposób odczytu wyniku, należy do wykonawcy. W specyfikacji nie piszesz kodu testu. Każdy test zapisujesz jako „dane → oczekiwany wynik”, używając tylko nazw ze szkieletu promptu kodu. Sekcja „Przygotowanie” mówi, skąd wziąć importowane nazwy i jak zbudować obiekty, żeby wykonawca nie musiał czytać kodu. Wypełnione przykłady specyfikacji, poprawki i weryfikacji są w `SKILLS/tester_skill/przyklad.md`.

```markdown
# <PP-nazwa> (testy)

- **Trudność:** łatwe | średnie | trudne
- **Model:** sonnet | opus
- **Plik testów (tylko ten wolno zmieniać):** <ścieżka według profilu>
- **Wspólne przygotowanie testów:** nie zmieniaj | zmieniasz: <co dodać>
- **Prompt kodu:** Docs/Prompts/NN-zadanie/kod/<PP-nazwa>.md
- **Kontrakt:** Docs/Kontrakty/<modul>.md
- **Ocena propozycji plannera:** wystarcza | uzupełnione: <co dodano>

## Przygotowanie
- Import: <dokładne linie importów>
- Obiekty: <jaki obiekt i jak go utworzyć: konstruktor albo funkcja tworząca, z jakimi argumentami>
- Zasoby: <katalog tymczasowy, dane testowe, wspólne przygotowanie testów: nazwa i co daje> | brak

## Testy
### <nazwa_testu>
- Przypadek: <co sprawdza, jedno zdanie>
- <dane wejściowe> → <dokładny oczekiwany wynik, błąd albo warunek>
```

## Tryb WERYFIKACJA

1. Użyj `superpowers:verification-before-completion`. Uruchom pełny zestaw testów i „Sprawdzenie statyczne”, obie komendy z profilu. Werdykt „gotowe” tylko wtedy, gdy przechodzą testy i sprawdzenie statyczne plików fali F.
   - Późniejsze fale mogą już mieć kod albo testy w repo, bo kod nie czeka na weryfikację. Błąd w pliku, który nie należy do fali F (nie ma go w „Plikach” jej promptów ani wśród jej plików testów), nie wpływa na werdykt fali F. Wpisz go w weryfikacji jako „poza falą” i nie pisz do niego poprawki.
   - Plików testów i specyfikacji innych fal nie zmieniasz, bo może nad nimi pracować inny tester.
2. Dla każdego nieprzechodzącego testu i błędu sprawdzenia statycznego rozstrzygnij, gdzie jest błąd:

| Sytuacja | Błąd w | Działanie |
| --- | --- | --- |
| Test sprawdza coś innego, niż mówi specyfikacja, albo ma literówkę lub zły import | teście | Odpal `wykonawca-testow` (model według tabeli „Model wykonawcy testów”) z poprawioną specyfikacją, uruchom testy ponownie. |
| Specyfikacja testu przeczy szkieletowi albo kontraktowi | specyfikacji | Popraw specyfikację, potem jak wyżej. |
| Kod łamie szkielet, opis funkcji albo kontrakt | kodzie | Napisz `Docs/Prompts/NN-zadanie/poprawki/PP-nazwa-K.md`. Kodu nie poprawiasz. |
| Sprawdzenie statyczne zgłasza błąd w pliku kodu (np. złamana reguła zależności, cykl, podwójna definicja, skopiowany blok) | kodzie | Poprawka jak wyżej. Przy skopiowanym bloku w „Do użycia” wskaż oryginał. |
| Sprawdzenie statyczne zgłasza błąd w pliku testów | teście | Jak przy błędzie w teście. |
| Kod musiał złamać reguły zależności z profilu (np. warstwy), bo tak go zaplanowano | kontrakcie | Zgłoś to orkiestratorowi. Rozstrzyga planner. |
| Szkielet i kontrakt są ze sobą sprzeczne albo niejasne | kontrakcie | Zgłoś to orkiestratorowi. Rozstrzyga planner. |

3. Hipotezę przyczyny błędu w kodzie zacznij od plików pułapek z profilu.
4. **Nowa pułapka.** Jeśli przyczyna błędu w kodzie może się powtórzyć w innym miejscu, a w plikach pułapek jej nie ma, dopisz wpis do pliku właściwej technologii w formacie tego pliku. Błąd jednorazowy, np. literówka, nie trafia do pułapek. Format wpisu: `- **Obszar.** Objaw → przyczyna → jak uniknąć.` Gdy pliku tej technologii jeszcze nie ma, zgłoś to w odpowiedzi dla orkiestratora.
5. Zapisz `Docs/Postep/NN-zadanie/weryfikacja-fala-F.md` (szablon niżej). Kolejne rundy dopisuj w tym samym pliku jako nową sekcję.

### Szablon poprawki kodu

Ten sam układ co prompt kodu (z „Stanem obecnym” opisującym kod po nieudanej rundzie), plus sekcja „Błąd” na początku:

```markdown
# <PP-nazwa>, poprawka <K>

- **Trudność:** łatwe | średnie | trudne
- **Pliki (tylko te wolno zmieniać):** <ścieżki>
- **Prompt pierwotny:** Docs/Prompts/NN-zadanie/kod/<PP-nazwa>.md

## Błąd
- Test: <plik i nazwa testu, zapisane tak, jak podaje je narzędzie testów> | <błąd sprawdzenia statycznego: plik, linia, kod błędu>
- Wynik: <dokładny komunikat i wejście, przy którym wystąpił>
- Oczekiwane według opisu funkcji albo kontraktu: <co powinno być>
- Hipoteza (pułapka „<nazwa>” albo własna): <przyczyna>

## Stan obecny
- **Struktura:** <zarys fragmentu z błędem, taki jak jest teraz w kodzie>
- **Nie ruszać:** <co zostaje bez zmian>

## Gotowe, gdy
- Komenda „Jeden plik testów” z profilu, uruchomiona dla `<plik testów>`, przechodzi.
```

Wykonawca poprawki może uruchamiać testy, ale nie może ich zmieniać.

### Szablon weryfikacji

```markdown
# Weryfikacja: zadanie <NN>, fala <F>

## Runda <R>
- **Testy:** `<komenda>` → <ile testów przeszło, ile nie>
- **Sprawdzenie statyczne:** przechodzi | <liczba błędów, pliki>

| Test albo błąd statyczny | Przyczyna | Błąd w | Działanie |
| --- | --- | --- | --- |
| <nazwa> | <przyczyna> | kodzie / teście / specyfikacji / kontrakcie | <plik poprawki albo „poprawione, przechodzi”> |

- **Nowe pułapki:** brak | <plik pułapek>: „<nazwa wpisu>”
- **Problemy z kontraktem:** brak | <opis>
- **Werdykt:** gotowe | niegotowe
```

Odpowiedź dla orkiestratora: werdykt, ścieżki nowych plików w `poprawki/`, dopisane pułapki albo „brak”, problemy z kontraktem albo „brak”.
