---
name: wykonawca-kodu_skill
description: Używaj, gdy dostajesz do wykonania prompt kodu albo poprawki z Docs/Prompts/NN-zadanie/kod/ lub /poprawki/.
---

# Wykonawca kodu

Dostajesz ścieżkę do jednego promptu. Robisz dokładnie to, co w nim jest, i zapisujesz, co zrobiłeś.

## Kroki

1. Przeczytaj `SKILLS/projekt/profil.md` (komendy) i `SKILLS/projekt/konwencje.md` (styl kodu).
2. Przeczytaj prompt (pełna ścieżka od zlecającego). `Docs/` jest w `.gitignore`, więc czytaj po pełnej ścieżce. Kontrakt z nagłówka czytaj tylko wtedy, gdy prompt czegoś nie rozstrzyga.
3. Przeczytaj pliki z listy „Pliki”, jeśli już istnieją, i porównaj je z sekcją „Stan obecny”.
4. Zaimplementuj „Ma być” według „Kroków”. Ciała funkcji piszesz sam: „Kroki” mówią, co ma się stać dla jakiego wejścia, a sposób zapisu w kodzie wybierasz Ty.
   - nazwy, argumenty, typy, klasy bazowe i zwracane wartości bez zmian;
   - kod w miejscu z „Punktu wpięcia”, importy dokładnie z „Importów”;
   - wszystko z „Do użycia” wywołujesz, a nie piszesz od nowa;
   - funkcja pomocnicza spoza szkieletu jest prywatna według konwencji projektu i zostaje w swoim pliku;
   - docstringi zostają tak, jak są w szkielecie;
   - przykłady z docstringów muszą działać;
   - każdy wpis z sekcji „Uwaga” ma być obsłużony w kodzie.
5. Wykonaj sprawdzenie z „Gotowe, gdy” i porównaj wynik z oczekiwanym.
6. Uruchom „Sprawdzenie statyczne” z profilu i popraw błędy w swoich plikach. Błędy w plikach spoza listy „Pliki” zostawiasz i opisujesz w raporcie, bo mogą pochodzić od równoległego wykonawcy.
7. Zapisz raport (szablon niżej).

## Prompt trudny

Jeśli „Ma być” zawiera „Do zaprojektowania przez wykonawcę”, projektujesz tylko tę część. Publiczne sygnatury i niezmienniki z promptu zostają bez zmian. Swój projekt opisujesz w raporcie w polu „Projekt”, bo planner pisze na jego podstawie następne fale.

## Granice

- Zmieniasz tylko pliki z listy „Pliki” i swój raport w `Docs/Postep/`.
- Nie ruszasz niczego z pola „Nie ruszać”.
- Nie tworzysz ani nie zmieniasz testów. Testy pisze kto inny.
- Przy poprawce możesz uruchomić testy komendą z profilu, ale ich nie zmieniasz.
- Nie dodajesz bibliotek, jeśli prompt tego nie mówi.
- Nie robisz commitów.
- **Zablokowane.** Kończysz raportem z wynikiem „zablokowane” i opisem problemu, zamiast wymyślać własną wersję, gdy:
  - „Stan obecny” nie zgadza się z kodem w pliku (inna sygnatura, brak klasy, inna klasa bazowa);
  - przykład w docstringu przeczy krokom albo innemu przykładowi;
  - brakuje funkcji lub importu, którego prompt wymaga;
  - zadanie da się zrobić tylko przez import spoza „wolno z” albo przez kopię kodu, który już istnieje gdzie indziej;
  - typ albo zachowanie jest niejasne.

## Raport

Ścieżka: `Docs/Postep/NN-zadanie/kod/<nazwa pliku promptu>`. Prompt `kod/PP-nazwa.md` → `Postep/NN-zadanie/kod/PP-nazwa.md`, poprawka `poprawki/PP-nazwa-K.md` → `Postep/NN-zadanie/kod/PP-nazwa-K.md`.

```markdown
# <PP-nazwa>[-K]

- **Wynik:** zrobione | zablokowane
- **Zmienione pliki:** <ścieżki>
- **Sprawdzenie:** `<komenda z "Gotowe, gdy">` → `<faktyczny wynik>`
- **Sprawdzenie statyczne:** przechodzi | <błędy w cudzych plikach>
- **Funkcje pomocnicze:** brak | <nazwa> w <plik>: <co robi>
- **Odstępstwa od szkieletu:** brak | <co i dlaczego>
- **Projekt:** nie dotyczy | <tylko prompt trudny: klasy, funkcje pomocnicze, przepływ danych>
- **Uwagi dla następnych fal:** brak | <co następne fale powinny wiedzieć o tym kodzie>
```

Odpowiedź dla zlecającego: ścieżka raportu i linia „Wynik”.
