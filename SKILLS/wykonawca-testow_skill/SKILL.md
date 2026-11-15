---
name: wykonawca-testow_skill
description: Używaj, gdy dostajesz do wykonania specyfikację testów z Docs/Prompts/NN-zadanie/testy/.
---

# Wykonawca testów

Dostajesz ścieżkę do jednej specyfikacji testów. Piszesz dokładnie te testy, które w niej są, w jednym pliku.

## Kroki

1. Przeczytaj `SKILLS/projekt/profil.md` (komendy) i `SKILLS/projekt/konwencje.md` (sekcja „Testy”).
2. Przeczytaj specyfikację (pełna ścieżka od zlecającego). `Docs/` jest w `.gitignore`, więc czytaj po pełnej ścieżce. Prompt kodu z nagłówka czytaj tylko wtedy, gdy „Przygotowanie” czegoś nie rozstrzyga.
3. Napisz plik testów z nagłówka specyfikacji:
   - importy i budowa obiektów dokładnie jak w „Przygotowaniu”;
   - nazwy testów dokładnie takie jak w specyfikacji;
   - wejścia i oczekiwane wyniki dokładnie takie jak w specyfikacji.
4. Sprawdź składnię, a potem uruchom „Sprawdzenie statyczne jednego pliku” na pliku testów. Obie komendy są w profilu. Popraw zgłoszone błędy w swoim pliku.
5. Uruchom plik testów komendą z profilu. Kod może jeszcze nie istnieć, bo powstaje równolegle, więc błąd importu albo nieprzechodzące testy są wtedy w porządku. Zapisz wynik w raporcie.
6. Zapisz raport (szablon niżej).

Przykład zamiany specyfikacji na test: `SKILLS/wykonawca-testow_skill/przyklad.md`.

## Granice

- Zmieniasz tylko plik testów z nagłówka (i wspólne fixture'y, jeśli nagłówek na to pozwala) oraz swój raport w `Docs/Postep/`.
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
- **Uruchomienie:** `<komenda z profilu>` → <błąd importu (kod jeszcze nie istnieje) | liczba passed, failed>
- **Uwagi:** brak | <niejasność w specyfikacji, brakujący przypadek>
```

Odpowiedź dla zlecającego: ścieżka raportu i linia „Wynik”.
