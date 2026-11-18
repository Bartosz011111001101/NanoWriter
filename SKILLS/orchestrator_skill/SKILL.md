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

Kod nie czeka na testy ani na numer zadania. Prompt rusza, gdy to, z czego korzysta, jest napisane. Testy, weryfikacja i poprawki dochodzą w tle i wracają do wykonawcy, który pisał dany kod. Wszystko, co może iść równolegle, odpalasz w jednej wiadomości.

1. **Planowanie wszystkich zadań naraz.** Gdy znasz listę zadań (projekt, plan wstępny, kilka etapów do zrobienia), planujesz je wszystkie od razu, a nie zadanie po zadaniu:
   - jeden **planner** dostaje wszystkie zadania, które jeszcze nie mają planu: „Zaplanuj naraz zadania `NN-a`, `NN-b`, … Cele: … Decyzje użytkownika: …”. Jeden agent pilnuje, żeby kontrakty się nie gryzły;
   - drugi planner może iść równolegle tylko dla zadań, które zmieniają inne kontrakty niż pierwszy. W zleceniu wymień, których plików kontraktów i których folderów `Docs/` nie rusza;
   - wraca lista promptów z kolumnami „Zależy od” (pełne nazwy `NN-zadanie/PP-nazwa`, także między zadaniami) i „Czeka na weryfikację”.
2. **Kiedy prompt rusza.** Prompt kodu odpalasz, gdy spełnia wszystkie trzy warunki:
   - każdy prompt z jego „Zależy od” ma kod „zrobione”, a jeśli jego fala ma „Czeka na weryfikację: tak”, to werdykt „gotowe” fali tych promptów;
   - żaden pracujący agent nie zmienia jego plików i do jego plików nie czeka niewykonana poprawka;
   - jeśli prompt korzysta z kodu, którego wykonawca zgłosił odstępstwo od szkieletu, uwagi dla następnych fal albo „Projekt” (prompt trudny), planner najpierw zaktualizował ten prompt.
   Numer zadania i fali nie wstrzymuje promptu. Prompty różnych zadań idą równolegle.
3. **Start prompów fali F zadania NN**, w jednej wiadomości z innymi gotowymi promptami:
   - **tester**: „Tryb SPEC. Zadanie `NN-zadanie`, fala F.” (raz na falę, przy pierwszym jej prompcie). Tester pisze specyfikacje i sam odpala wykonawców testów;
   - **wykonawca-kodu**, po jednym na prompt: „Wykonaj `Docs/Prompts/NN-zadanie/kod/PP-nazwa.md`.” Model według tabeli niżej;
   - zapisz w `Docs/Postep/NN-zadanie/agenci.md` wiersz `PP-nazwa | model | id agenta`. Id jest potrzebne do poprawek, a po streszczeniu rozmowy znika z kontekstu.
4. Po każdym powrocie wykonawcy sprawdź punkt 2 dla wszystkich czekających promptów i odpal te, które mogą ruszyć.
5. Wykonawca zgłosił „zablokowane”: odpal plannera z raportem, żeby poprawił prompt, potem jeszcze raz wykonawcę z tym samym modelem.
6. **Weryfikacja fali F** rusza, gdy kod i testy fali F są gotowe (wrócili wykonawcy kodu fali F i tester SPEC fali F): **tester**: „Tryb WERYFIKACJA. Zadanie `NN-zadanie`, fala F, runda R.” Weryfikacje różnych fal mogą iść równolegle.
7. Werdykt **gotowe**: commit (niżej).
8. Werdykt **niegotowe**: każdy nowy plik `Prompts/NN-zadanie/poprawki/PP-nazwa-K.md` wraca do wykonawcy, który pisał `PP-nazwa` (eskalacja niżej). Poprawki różnych plików idą równolegle. Potem znowu punkt 6 z R+1.
9. **Jeden plik naraz.** Dwa agenty nigdy nie zmieniają tego samego pliku w tym samym czasie. Porównaj „Pliki” poprawki z „Plikami” promptów, które właśnie pracują. Przy kolizji poprawka czeka, aż tamten wykonawca wróci.
10. Po ostatniej fali ostatniego zadania: `superpowers:verification-before-completion` i krótkie podsumowanie dla użytkownika.

Wykonawca kodu nigdy nie dostaje promptu testów ani nie zmienia testów. Testy powstają wyłącznie u testera.

### Poprawka do tego samego wykonawcy

Pierwszą poprawkę danego `PP-nazwa` wysyłasz do wykonawcy, który pisał ten kod: narzędzie `SendMessage` (jeśli jest odroczone, załaduj je przez `ToolSearch` z `select:SendMessage`), adresat to id z `agenci.md`, treść: „Wykonaj poprawkę `Docs/Prompts/NN-zadanie/poprawki/PP-nazwa-K.md`.” Wykonawca ma już w kontekście prompt i swój kod, więc poprawia szybciej. Gdy id jest nieznane albo agent nie odpowiada, odpal nowego wykonawcę z tym samym modelem.

## Trudność → model

| Trudność w nagłówku promptu | Wykonawca kodu (odpalasz Ty) | Wykonawca testów (odpala tester) |
| --- | --- | --- |
| łatwe | `haiku` | `sonnet` |
| średnie | `sonnet` | `sonnet` |
| trudne | `opus` | `opus` |

Testy nigdy nie idą na Haiku.

**Eskalacja poprawek** dla tego samego `PP-nazwa`:
- poprawka 1: ten sam wykonawca, który pisał kod (`SendMessage`, ten sam model);
- poprawka 2 i kolejne: nowy wykonawca, model o jeden poziom wyżej niż w poprzedniej rundzie (haiku → sonnet → opus). Dopisz go do `agenci.md`.

Gdy nie wyjdzie na Opusie, zatrzymaj zadanie i zapytaj użytkownika.

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
- Fale jednego zadania commitujesz po kolei: fala F dopiero po commicie fali F−1. Falę commitujesz dopiero wtedy, gdy są w commitach wszystkie prompty z „Zależy od” jej promptów, także z innych zadań.
- Do commita fali F dodajesz (`git add <ścieżki>`) tylko pliki z „Plików” jej promptów i poprawek oraz jej pliki testów. Pliki późniejszych fal zostają poza commitem.
- Jeśli plik fali F zmieniła już późniejsza, jeszcze niezweryfikowana fala, wstrzymaj commit fali F i zrób jeden wspólny commit obu fal po werdykcie „gotowe” tej późniejszej: `zadanie NN fale F–G: <co zrobiono>`.
- Do commita trafiają też pułapki dopisane przez testera w `SKILLS/projekt/pulapki/`.
- Wiadomość: `zadanie NN fala F: <co zrobiono>`, np. `zadanie 05 fala 2: eksport do CSV`.
- `Docs/` nie trafia do commitów.
