---
name: orchestrator_skill
description: Używaj w głównej sesji przy każdej prośbie użytkownika, gdy trzeba zdecydować, kogo odpalić (planner, tester, wykonawca-kodu), jakim modelem, w jakiej kolejności, albo zamknąć falę lub zadanie.
---

# Orkiestracja pracy

Orkiestrator nie pisze kodu, testów ani promptów. Odpala agentów, pilnuje kolejności i commituje. Projekt (stack, komendy, etapy, bramki, decyzje użytkownika) opisuje `SKILLS/projekt/profil.md`.

## Mapa Docs/

`NN-zadanie` to folder jednego zadania (np. `05-eksport`). `PP-nazwa` to jeden prompt (np. `02-eksport-csv`). Ten sam `PP-nazwa` w każdym folderze dotyczy tego samego kawałka pracy, więc czytasz zawsze tylko jeden plik.

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
- Operacje z sekcji „Ograniczenia” w profilu.
- Zmiana kontraktu, który ma już kod i testy.
- Gdy poprawka nie wychodzi nawet na Opusie.
- Gdy planner zwróci „Otwarte pytania” inne niż „brak”.

## Bramki etapów

Zadanie odpowiadające etapowi jest zamknięte dopiero wtedy, gdy przechodzi test bramki z profilu. Planner umieszcza go w ostatniej fali.

## Przeniesienie do nowego projektu

| Co | Jak |
| --- | --- |
| `.claude/agents/`, `SKILLS/*_skill/`, `.claude/settings.json` (plugin superpowers) | kopiujesz bez zmian |
| `SKILLS/projekt/pulapki/` | kopiujesz pliki pasujących technologii, brakujące technologie zaczynasz od pustego pliku w tym samym formacie |
| `SKILLS/projekt/profil.md`, `konwencje.md`, `katalog-testow.md` | piszesz od nowa według `SKILLS/orchestrator_skill/szablon-profilu.md`. Każda sekcja i każda komenda z szablonu musi istnieć albo mieć wartość „brak”. |
| Narzędzia sprawdzenia statycznego | instalujesz narzędzia, które łapią: nieużywany kod i podwójne definicje, cykle zależności, skopiowane bloki, a w projekcie z warstwami także import z wyższej warstwy. Komendy i pliki konfiguracji wpisujesz do profilu. Przykład dla Pythona: ruff, import-linter, pylint. |
| `CLAUDE.md` | kopiujesz bez zmian. Zawiera trzy importy: `@.claude/agents/orchestrator.md`, `@SKILLS/orchestrator_skill/SKILL.md`, `@SKILLS/projekt/profil.md` |
| `.gitignore` | wpis `Docs/` |
| Platforma | Claude Code z modelami `haiku`, `sonnet`, `opus`, z pluginem superpowers (katalog pluginów musi być dodany na komputerze) i z możliwością odpalania agentów przez agenta. Po zmianie nazw modeli poprawiasz tabele w `orchestrator_skill`, `planner_skill`, `tester_skill` i pole `model:` w agentach `wykonawca-kodu` i `wykonawca-testow`. Instrukcje są po polsku. |

## Commit

- Jeden commit na falę z werdyktem „gotowe”, na gałęzi z sekcji „Git” profilu. Push tylko wtedy, gdy ta sekcja na to pozwala.
- Do commita trafiają też pułapki dopisane przez testera w `SKILLS/projekt/pulapki/`.
- Wiadomość: `zadanie NN fala F: <co zrobiono>`, np. `zadanie 05 fala 2: eksport do CSV`.
- `Docs/` nie trafia do commitów.
