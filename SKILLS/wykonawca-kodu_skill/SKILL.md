---
name: wykonawca-kodu_skill
description: Używaj, gdy dostajesz do wykonania prompt kodu albo poprawki z Docs/Prompts/NN-zadanie/kod/ lub /poprawki/ w projekcie NanoWriter.
---

# Wykonawca kodu w NanoWriter

Dostajesz ścieżkę do jednego promptu. Robisz dokładnie to, co w nim jest, i zapisujesz, co zrobiłeś.

## Kroki

1. Przeczytaj prompt (pełna ścieżka od orkiestratora) i kontrakt wskazany w nagłówku. `Docs/` jest w `.gitignore`, więc czytaj po pełnej ścieżce.
2. Przeczytaj pliki z listy „Pliki”, jeśli już istnieją. Nie usuwaj kodu, którego prompt nie dotyczy.
3. Zaimplementuj szkielet:
   - nazwy, argumenty, typy i zwracane wartości bez zmian;
   - docstringi zostają tak, jak są w szkielecie;
   - przykłady z docstringów muszą działać.
4. Wykonaj sprawdzenie z „Gotowe, gdy”. Komendy Pythona uruchamiaj przez `uv run`.
5. Zapisz raport (szablon niżej).

## Granice

- Zmieniasz tylko pliki z listy „Pliki” i swój raport w `Docs/Postep/`.
- Nie tworzysz ani nie zmieniasz niczego w `tests/`. Testy pisze kto inny.
- Przy poprawce możesz uruchomić `uv run pytest <plik>`, ale testów nie zmieniasz.
- Nie dodajesz bibliotek (`uv add`), jeśli prompt tego nie mówi.
- Nie robisz commitów.
- Szkielet nie da się zaimplementować tak, jak jest opisany (sprzeczny przykład, brak potrzebnej funkcji, niejasny typ)? Nie wymyślaj własnej wersji. Zapisz raport z wynikiem „zablokowane” i opisz problem.

## Raport

Ścieżka: `Docs/Postep/NN-zadanie/kod/<nazwa pliku promptu>`, np. prompt `kod/01-pary.md` → `Postep/NN-zadanie/kod/01-pary.md`, poprawka `poprawki/01-pary-1.md` → `Postep/NN-zadanie/kod/01-pary-1.md`.

```markdown
# 01-pary

- **Wynik:** zrobione | zablokowane
- **Zmienione pliki:** src/nanowriter/tokenizer.py
- **Sprawdzenie:** `uv run python -c "..."` → `{(1, 2): 2, (2, 1): 1}`
- **Odstępstwa od szkieletu:** brak | <co i dlaczego>
- **Uwagi dla następnych fal:** brak | <np. funkcja X przyjmuje już też pustą listę>
```

Odpowiedź dla zlecającego: ścieżka raportu i linia „Wynik”.
