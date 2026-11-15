---
name: orchestrator_skill
description: Używaj w głównej sesji przy każdej prośbie użytkownika, gdy trzeba zdecydować, kogo odpalić (planner, tester, wykonawca-kodu), jakim modelem, w jakiej kolejności, albo zamknąć falę lub zadanie.
---

# Orkiestracja pracy

Orkiestrator nie pisze kodu, testów ani promptów. Odpala agentów, pilnuje kolejności i commituje. Projekt (stack, komendy, etapy, bramki, decyzje użytkownika) opisuje `SKILLS/projekt/profil.md`.

## Mapa Docs/

`NN-zadanie` to folder jednego zadania (np. `03-tokenizer`). `PP-nazwa` to jeden prompt (np. `01-pary`). Ten sam `PP-nazwa` w każdym folderze dotyczy tego samego kawałka pracy, więc czytasz zawsze tylko jeden plik.

```
Docs/
  Plan/NN-zadanie/plan.md                        planner: cel, fale, trudności
  Prompts/NN-zadanie/kod/PP-nazwa.md             planner: stan obecny, szkielet, kroki
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
| Nowe zadanie (etap z planu wstępnego albo funkcja) | Brakujące decyzje użytkownika → zapytaj. Potem cykl zadania. |
| Bug poza trwającym zadaniem | Sam `superpowers:systematic-debugging`, potem małe zadanie dla plannera. |
| Zmiana kontraktu, który ma już kod | Zapytaj użytkownika, potem planner. |
| Drobna poprawka (literówka, komentarz) | Zrób sam, uruchom testy, commit. |

## Cykl zadania

1. **planner**: „Zaplanuj zadanie `NN-zadanie`. Cel: … Decyzje użytkownika: …”. Wraca lista fal.
2. **Dla każdej fali F, po kolei:**
   1. Przed falą F > 1 przeczytaj `Postep/NN-zadanie/kod/*` z poprzednich fal. Jeśli ktoś zgłosił odstępstwo od szkieletu, uwagi dla następnych fal albo opisał „Projekt” (prompt trudny), najpierw odpal plannera z prośbą o aktualizację promptów fali F.
   2. **Równolegle, w jednej wiadomości:**
      - **tester**: „Tryb SPEC. Zadanie `NN-zadanie`, fala F.” Tester pisze specyfikacje i sam odpala wykonawców testów.
      - **wykonawca-kodu**, po jednym na każdy prompt fali: „Wykonaj `Docs/Prompts/NN-zadanie/kod/PP-nazwa.md`.” Model według tabeli niżej.
   3. Wykonawca zgłosił „zablokowane”: odpal plannera z raportem, żeby poprawił prompt, potem jeszcze raz wykonawcę z tym samym modelem.
   4. Gdy wszyscy wrócą, **tester**: „Tryb WERYFIKACJA. Zadanie `NN-zadanie`, fala F, runda R.”
   5. Werdykt **gotowe** (testy i sprawdzenie statyczne przechodzą): commit (niżej), następna fala.
   6. Werdykt **niegotowe**: dla każdego nowego pliku w `Prompts/NN-zadanie/poprawki/` odpal wykonawcę kodu, równolegle, jeśli pliki się nie pokrywają. Potem znowu punkt 4 z R+1.
3. Po ostatniej fali: `superpowers:verification-before-completion` i krótkie podsumowanie dla użytkownika.

Wykonawca kodu nigdy nie dostaje promptu testów ani nie zmienia testów. Testy powstają wyłącznie u testera.

## Trudność → model

| Trudność w nagłówku promptu | Wykonawca kodu (odpalasz Ty) | Wykonawca testów (odpala tester) |
| --- | --- | --- |
| łatwe | `haiku` | `sonnet` |
| średnie | `sonnet` | `sonnet` |
| trudne | `opus` | `opus` |

Testy nigdy nie idą na Haiku.

**Eskalacja poprawek:** każda kolejna runda poprawki tego samego `PP-nazwa` podnosi model wykonawcy kodu o jeden poziom (haiku → sonnet → opus). Gdy nie wyjdzie na Opusie, zatrzymaj zadanie i zapytaj użytkownika.

## Kiedy pytać użytkownika

- Decyzje użytkownika wymienione w profilu.
- Operacje z sekcji „Ograniczenia” w profilu (np. długi trening, duże pobieranie).
- Zmiana kontraktu, który ma już kod i testy.
- Gdy poprawka nie wychodzi nawet na Opusie.
- Gdy planner zwróci „Otwarte pytania” inne niż „brak”.

## Bramki etapów

Zadanie odpowiadające etapowi jest zamknięte dopiero wtedy, gdy przechodzi test bramki z profilu. Planner umieszcza go w ostatniej fali.

## Commit

- Jeden commit na falę z werdyktem „gotowe”, na `main`, bez pushu.
- Do commita trafiają też pułapki dopisane przez testera w `SKILLS/projekt/pulapki/`.
- Wiadomość: `zadanie NN fala F: <co zrobiono>`, np. `zadanie 03 fala 1: zliczanie par i merge BPE`.
- `Docs/` nie trafia do commitów.
