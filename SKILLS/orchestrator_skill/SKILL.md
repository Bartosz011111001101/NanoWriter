---
name: orchestrator_skill
description: Używaj w głównej sesji projektu NanoWriter przy każdej prośbie użytkownika, gdy trzeba zdecydować, kto wykonuje pracę (planner, tester, implementacja), napisać zlecenie dla agenta albo zamknąć zadanie lub etap.
---

# Orkiestracja pracy w NanoWriter

## Co robić z prośbą

| Prośba | Działanie |
| --- | --- |
| Pytanie, wyjaśnienie | Odpowiedz sam, bez agentów. |
| Nowy etap z `wstępny_plan.md` albo nowa funkcja | Brakujące decyzje → zapytaj użytkownika. Potem **planner**. |
| Zadanie z gotowego planu | Cykl zadania (niżej). |
| Bug, test nie przechodzi | Sam: `superpowers:systematic-debugging`. Poprawkę robi implementacja, a **tester** ją potwierdza. |
| Zmiana kontraktu, który ma już kod | Zapytaj użytkownika. Potem **planner** aktualizuje kontrakt, a **tester** testy. |
| Drobna poprawka (literówka, komentarz, jedna linia) | Zrób sam, uruchom `uv run pytest`, commit. |

## Cykl zadania

1. **tester (RED)** pisze testy do zadania. Testy muszą nie przechodzić z właściwego powodu.
2. **Implementacja** (subagent `general-purpose`) pisze kod, aż testy przejdą.
3. **tester (weryfikacja)** uruchamia pełny `uv run pytest`.
4. Werdykt „gotowe”: commit na `main` i wpis do `Docs/Postep/`.
5. Werdykt „niegotowe”: odeślij implementacji raport testera bez zmian. Po drugiej nieudanej poprawce przerwij i zrób sam `superpowers:systematic-debugging`. Jeśli przyczyna leży w kontrakcie, zapytaj użytkownika.

Zadania wykonuj po kolei, nie równolegle, bo moduły LLM zależą od siebie (dane → tokenizer → model → trening).

## Szablony zleceń

Każde zlecenie to cztery punkty, w tej kolejności.

**Dla plannera:**
1. Cel: rozpisz etap N (nazwa) na zadania i kontrakty.
2. Czytaj: `Docs/Plan/wstępny_plan.md`, `Docs/Kontrakty/*`, ostatni wpis w `Docs/Postep/`.
3. Decyzje użytkownika: (lista, np. „język: polski, dane: Wolne Lektury”).
4. Gotowe, gdy: plan w `Docs/Plan/etap-N.md` i kontrakty dla nowych modułów.

**Dla testera (RED):**
1. Cel: testy do zadania „X” z `Docs/Plan/etap-N.md`.
2. Czytaj: plan (sekcja zadania) i `Docs/Kontrakty/<modul>.md`.
3. Możesz zmieniać: `tests/test_<modul>.py`, `tests/conftest.py`.
4. Gotowe, gdy: testy istnieją i nie przechodzą z właściwego powodu.

**Dla implementacji:**
1. Cel: zaimplementuj zadanie „X”, aż przejdą testy w `tests/test_<modul>.py`.
2. Czytaj: plan (sekcja zadania), `Docs/Kontrakty/<modul>.md`, testy.
3. Możesz zmieniać: `src/nanowriter/<modul>.py` (i tylko pliki wskazane w planie). Testów nie zmieniasz.
4. Gotowe, gdy: `uv run pytest tests/test_<modul>.py` przechodzi. Uruchamiaj przez `uv run`.

**Dla testera (weryfikacja):**
1. Cel: potwierdź zadanie „X”.
2. Czytaj: plan i kontrakt modułu.
3. Niczego nie zmieniasz.
4. Gotowe, gdy: raport z wynikiem pełnego `uv run pytest` i werdyktem.

## Kiedy pytać użytkownika

- Decyzje o produkcie: język, rodzaj tekstu, źródło danych, rozmiar modelu.
- Zmiana kontraktu, który ma już kod i testy.
- Przed treningiem dłuższym niż ~15 minut albo pobieraniem dużych danych (powyżej ~1 GB), bo użytkownik korzysta wtedy z komputera.
- Po dwóch nieudanych poprawkach tego samego zadania.

## Bramki etapów

Etap jest zamknięty dopiero, gdy przechodzi jego test bramki:

| Etap | Test bramki |
| --- | --- |
| 1 Bigram | Strata treningowa spada, `generate` zwraca tekst o zadanej długości. |
| 2 Dane | Podział train/val istnieje, zbiory się nie pokrywają, rozmiary zapisane w `Docs/Postep/`. |
| 3 Tokenizer | `decode(encode(x)) == x` dla polskich znaków, emoji i pustego tekstu, a pliki `.bin` mają dtype z kontraktu. |
| 4 Transformer | Kształty się zgadzają, test maski przyczynowej przechodzi, model przeucza się na jednym batchu. |
| 5 Trening | Wznowienie z checkpointu daje tę samą stratę co trening bez przerwy, a strata walidacyjna jest logowana. |
| 6 Generowanie | Z KV cache i bez niego wynik jest identyczny przy tym samym seedzie. |

Dla etapów 7–10 bramkę ustala planner w planie etapu.

## Commit

- Jeden commit na zadanie, na `main`, bez pushu.
- Wiadomość: `etap N: <co zrobiono>`, np. `etap 3: BPE encode/decode z testem roundtrip`.
- `Docs/` jest w `.gitignore` i nie trafia do commitów.

## Wpis w Docs/Postep/

Plik `Docs/Postep/RRRR-MM-DD.md`, jeden wpis na zadanie:

```markdown
### etap N / <nazwa zadania>
- **Wynik:** zrobione | częściowo | zablokowane
- **Testy:** `uv run pytest`: 14 passed
- **Commit:** abc1234
- **Dalej:** <następne zadanie albo opis blokady>
```
