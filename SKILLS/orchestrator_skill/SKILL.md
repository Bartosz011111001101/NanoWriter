---
name: orchestrator_skill
description: Używaj w głównej sesji projektu NanoWriter przy każdej prośbie użytkownika, gdy trzeba zdecydować, kogo odpalić (planner, tester, wykonawca-kodu), jakim modelem, w jakiej kolejności, albo zamknąć falę lub zadanie.
---

# Orkiestracja pracy w NanoWriter

Orkiestrator nie pisze kodu, testów ani promptów. Odpala agentów, pilnuje kolejności i commituje.

## Mapa Docs/

`NN-zadanie` to folder jednego zadania (np. `03-tokenizer`). `PP-nazwa` to jeden prompt (np. `01-pary`). Ten sam `PP-nazwa` w każdym folderze dotyczy tego samego kawałka pracy, więc czytasz zawsze tylko jeden plik.

```
Docs/
  Plan/NN-zadanie/plan.md                        planner: cel, fale, trudności
  Prompts/NN-zadanie/kod/PP-nazwa.md             planner: szkielet + kroki dla wykonawcy kodu
  Prompts/NN-zadanie/testy/PP-nazwa.md           tester: specyfikacja testów
  Prompts/NN-zadanie/poprawki/PP-nazwa-K.md      tester: poprawka kodu nr K
  Postep/NN-zadanie/kod/PP-nazwa[-K].md          wykonawca-kodu: co zrobił
  Postep/NN-zadanie/testy/PP-nazwa.md            wykonawca-testow: co zrobił
  Postep/NN-zadanie/weryfikacja-fala-F.md        tester: wynik testów i werdykt
  Kontrakty/<modul>.md                           planner: jeden plik na moduł, bez folderów zadań
```

`Docs/` jest w `.gitignore`, więc Grep i Glob mogą go pomijać. Listuj przez `ls`, czytaj po pełnej ścieżce.

## Co robić z prośbą

| Prośba | Działanie |
| --- | --- |
| Pytanie, wyjaśnienie | Odpowiedz sam, bez agentów. |
| Nowe zadanie (etap z `Plan/00-wstepny-plan/wstępny_plan.md` albo funkcja) | Brakujące decyzje użytkownika → zapytaj. Potem cykl zadania. |
| Bug poza trwającym zadaniem | Sam `superpowers:systematic-debugging`, potem małe zadanie dla plannera. |
| Zmiana kontraktu, który ma już kod | Zapytaj użytkownika, potem planner. |
| Drobna poprawka (literówka, komentarz) | Zrób sam, `uv run pytest`, commit. |

## Cykl zadania

1. **planner**: „Zaplanuj zadanie `NN-zadanie`. Cel: … Decyzje użytkownika: …”. Wraca lista fal.
2. **Dla każdej fali F, po kolei:**
   1. Przed falą F > 1 przeczytaj `Postep/NN-zadanie/kod/*` z poprzednich fal. Jeśli ktoś zgłosił odstępstwo od szkieletu albo uwagi dla następnych fal, najpierw odpal plannera z prośbą o aktualizację promptów fali F.
   2. **Równolegle, w jednej wiadomości:**
      - **tester**: „Tryb SPEC. Zadanie `NN-zadanie`, fala F.” Tester pisze specyfikacje i sam odpala swoich wykonawców testów.
      - **wykonawca-kodu**, po jednym na każdy prompt fali: „Wykonaj `Docs/Prompts/NN-zadanie/kod/PP-nazwa.md`.” Model według tabeli niżej.
   3. Gdy wszyscy wrócą, **tester**: „Tryb WERYFIKACJA. Zadanie `NN-zadanie`, fala F, runda R.”
   4. Werdykt **gotowe**: commit (niżej), następna fala.
   5. Werdykt **niegotowe**: dla każdego nowego pliku w `Prompts/NN-zadanie/poprawki/` odpal wykonawcę-kodu, równolegle, jeśli pliki się nie pokrywają. Potem znowu punkt 3 z R+1.
3. Po ostatniej fali: `superpowers:verification-before-completion` i krótkie podsumowanie dla użytkownika.

Wykonawca kodu nigdy nie dostaje promptu testów ani nie zmienia `tests/`. Testy powstają wyłącznie u testera.

## Trudność → model

| Trudność w nagłówku promptu | Model |
| --- | --- |
| łatwe | `haiku` |
| średnie | `sonnet` |
| trudne | `opus` |

**Eskalacja poprawek:** każda kolejna runda poprawki tego samego `PP-nazwa` podnosi model o jeden poziom (haiku → sonnet → opus). Gdy nie wyjdzie na Opusie, zatrzymaj zadanie i zapytaj użytkownika.

## Kiedy pytać użytkownika

- Decyzje o produkcie: język, rodzaj tekstu, źródło danych, rozmiar modelu.
- Zmiana kontraktu, który ma już kod i testy.
- Przed treningiem dłuższym niż ~15 minut albo pobieraniem więcej niż ~1 GB danych, bo użytkownik korzysta wtedy z komputera.
- Gdy poprawka nie wychodzi nawet na Opusie.
- Gdy planner zwróci „Otwarte pytania” inne niż „brak”.

## Bramki etapów

Zadanie odpowiadające etapowi jest zamknięte dopiero, gdy przechodzi test bramki. Planner umieszcza go w ostatniej fali.

| Etap | Test bramki |
| --- | --- |
| 1 Bigram | Strata treningowa spada, `generate` zwraca tekst o zadanej długości. |
| 2 Dane | Podział train/val istnieje, zbiory się nie pokrywają. |
| 3 Tokenizer | `decode(encode(x)) == x` dla polskich znaków, emoji i pustego tekstu, a pliki `.bin` mają dtype z kontraktu. |
| 4 Transformer | Kształty się zgadzają, test maski przyczynowej przechodzi, model przeucza się na jednym batchu. |
| 5 Trening | Wznowienie z checkpointu daje tę samą stratę co trening bez przerwy. |
| 6 Generowanie | Z KV cache i bez niego wynik jest identyczny przy tym samym seedzie. |

Dla etapów 7–10 bramkę ustala planner.

## Commit

- Jeden commit na falę z werdyktem „gotowe”, na `main`, bez pushu.
- Wiadomość: `zadanie NN fala F: <co zrobiono>`, np. `zadanie 03 fala 1: zliczanie par i merge BPE`.
- `Docs/` nie trafia do commitów.
