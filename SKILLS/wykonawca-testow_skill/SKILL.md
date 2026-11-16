---
name: wykonawca-testow_skill
description: Używaj, gdy dostajesz do wykonania specyfikację testów z Docs/Prompts/NN-zadanie/testy/.
---

# Wykonawca testów

Dostajesz ścieżkę do jednej specyfikacji testów. Piszesz dokładnie te testy, które w niej są, w jednym pliku.

## Kroki

1. Przeczytaj `SKILLS/projekt/profil.md` (komendy) i `SKILLS/projekt/konwencje.md` (sekcja „Testy”).
2. Przeczytaj specyfikację (pełna ścieżka od zlecającego). `Docs/` jest w `.gitignore`, więc czytaj po pełnej ścieżce. Prompt kodu z nagłówka czytaj tylko wtedy, gdy „Przygotowanie” czegoś nie rozstrzyga.
3. Napisz plik testów z nagłówka specyfikacji. Kod testów piszesz sam: specyfikacja mówi, co sprawdzić („dane → oczekiwany wynik”), a asercje, przygotowanie danych i parametryzację dobierasz Ty według konwencji projektu. Nie zmieniasz przy tym:
   - importów i obiektów z „Przygotowania”;
   - nazw testów;
   - danych wejściowych i oczekiwanych wyników, co do znaku.
4. Uruchom „Sprawdzenie składni” i „Sprawdzenie statyczne jednego pliku” z profilu na pliku testów. Komendę z wartością „brak” pomijasz. Popraw błędy, które dotyczą Twojego pliku. Błąd, który wynika tylko z tego, że testowany kod jeszcze nie istnieje (nieznany import, nieznana nazwa, błąd kompilacji na brakującym symbolu), zostawiasz i wpisujesz do raportu.
5. Uruchom plik testów komendą „Jeden plik testów” z profilu. Kod może jeszcze nie istnieć, bo powstaje równolegle, więc błąd braku testowanego kodu (import, kompilacja) albo nieprzechodzące testy są wtedy w porządku. Zapisz wynik w raporcie.
6. Zapisz raport (szablon niżej).

Przykład zamiany specyfikacji na test: `SKILLS/wykonawca-testow_skill/przyklad.md`.

## Granice

- Zmieniasz tylko plik testów z nagłówka (i wspólne przygotowanie testów, jeśli nagłówek na to pozwala) oraz swój raport w `Docs/Postep/`.
- Nie zmieniasz kodu, nawet gdy widzisz w nim błąd. Opisz go w raporcie.
- Nie dopisujesz testów spoza specyfikacji. Jeśli Twoim zdaniem czegoś brakuje, zapisz to w raporcie.
- Nie robisz commitów.

## Raport

Ścieżka: `Docs/Postep/NN-zadanie/testy/<nazwa pliku specyfikacji>`. Specyfikacja `testy/PP-nazwa.md` → `Postep/NN-zadanie/testy/PP-nazwa.md`.

```markdown
# <PP-nazwa> (testy)

- **Wynik:** zrobione | zablokowane
- **Plik testów:** <ścieżka>
- **Liczba testów:** <liczba>
- **Uruchomienie:** `<komenda z profilu>` → <brak testowanego kodu (jeszcze nie istnieje) | ile testów przeszło, ile nie>
- **Uwagi:** brak | <niejasność w specyfikacji, brakujący przypadek>
```

Odpowiedź dla zlecającego: ścieżka raportu i linia „Wynik”.
