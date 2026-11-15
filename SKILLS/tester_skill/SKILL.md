---
name: tester_skill
description: Używaj, gdy w NanoWriter trzeba ocenić propozycje testów od plannera i napisać specyfikacje testów dla wykonawców (tryb SPEC), albo uruchomić testy fali, rozstrzygnąć, czy błąd jest w kodzie, czy w teście, i napisać poprawki (tryb WERYFIKACJA).
---

# Testowanie w NanoWriter

Tester projektuje testy i ocenia wyniki. Testów nie pisze sam, tylko odpala do tego agentów `wykonawca-testow`. Kodu w `src/` nie zmienia nigdy.

## Pliki

```
czytasz:  Docs/Plan/NN-zadanie/plan.md
          Docs/Prompts/NN-zadanie/kod/PP-nazwa.md     szkielet i "Propozycja testów"
          Docs/Kontrakty/<modul>.md
piszesz:  Docs/Prompts/NN-zadanie/testy/PP-nazwa.md   specyfikacja testów
          Docs/Prompts/NN-zadanie/poprawki/PP-nazwa-K.md   poprawka kodu
          Docs/Postep/NN-zadanie/weryfikacja-fala-F.md
```

`Docs/` jest w `.gitignore`. Listuj przez `ls`, czytaj po pełnej ścieżce.

Plik testów jednego promptu: `tests/test_<modul>_<PP>_<nazwa>.py` (podkreślniki zamiast myślników, np. `tests/test_tokenizer_01_pary.py`). Każdy prompt ma własny plik, więc równolegli wykonawcy testów się nie nadpisują. Wspólne fixture'y w `tests/conftest.py` zmienia w fali tylko jeden prompt testów, oznaczony w nagłówku.

## Tryb SPEC

Dla każdego promptu kodu z fali F:

1. Oceń „Propozycję testów” plannera według listy niżej. Brakujące testy dopisz.
2. Zapisz specyfikację w `Docs/Prompts/NN-zadanie/testy/PP-nazwa.md` (szablon niżej).
3. Odpal agenta `wykonawca-testow` dla każdej specyfikacji, równolegle, model według trudności (łatwe → `haiku`, średnie → `sonnet`, trudne → `opus`). Zlecenie: „Wykonaj `Docs/Prompts/NN-zadanie/testy/PP-nazwa.md`.”
4. Odpowiedź dla orkiestratora: lista plików testów i wyniki z `Docs/Postep/NN-zadanie/testy/`.

### Czy propozycja wystarcza

- Każdy niezmiennik z kontraktu modułu ma test.
- Każdy przykład z docstringu w szkielecie ma test.
- Przypadki brzegowe: pusty wejściowy tekst lub lista, jeden element, polskie znaki i emoji, `B=1` i `T=1` dla tensorów.
- Kształty i dtype każdego zwracanego tensora.
- Wyjątki, jeśli kontrakt je przewiduje (`pytest.raises`).
- Dla modułów modelu i treningu: pozycje z katalogu niżej.

### Szablon specyfikacji testów

Każdy test ma dokładne wejście i dokładny oczekiwany wynik. Wykonawca niczego nie wymyśla.

```markdown
# 01-pary (testy)

- **Trudność:** łatwe
- **Plik testów (tylko ten wolno zmieniać):** tests/test_tokenizer_01_pary.py
- **conftest.py:** nie zmieniaj
- **Prompt kodu:** Docs/Prompts/03-tokenizer/kod/01-pary.md
- **Kontrakt:** Docs/Kontrakty/tokenizer.md
- **Ocena propozycji plannera:** uzupełnione: dodano test na typ zwracanej wartości

## Import
`from nanowriter.tokenizer import get_pair_counts`

## Testy
### test_get_pair_counts_example
- Wejście: `get_pair_counts([1, 2, 1, 2])`
- Oczekiwane: `== {(1, 2): 2, (2, 1): 1}`

### test_get_pair_counts_short_inputs
- Wejście: `get_pair_counts([])` i `get_pair_counts([7])`
- Oczekiwane: oba `== {}`

### test_get_pair_counts_repeated
- Wejście: `get_pair_counts([5, 5, 5, 5])`
- Oczekiwane: `== {(5, 5): 3}`
```

## Tryb WERYFIKACJA

1. Użyj `superpowers:verification-before-completion`. Uruchom pełny zestaw: `uv run pytest`.
2. Dla każdego nieprzechodzącego testu rozstrzygnij, gdzie jest błąd:

| Sytuacja | Błąd w | Działanie |
| --- | --- | --- |
| Test sprawdza coś innego, niż mówi specyfikacja, albo ma literówkę lub zły import | teście | Odpal `wykonawca-testow` z poprawioną specyfikacją, uruchom testy ponownie. |
| Specyfikacja testu przeczy szkieletowi albo kontraktowi | specyfikacji | Popraw specyfikację, potem jak wyżej. |
| Kod łamie szkielet, docstring albo kontrakt | kodzie | Napisz `Docs/Prompts/NN-zadanie/poprawki/PP-nazwa-K.md`. Kodu nie poprawiasz. |
| Szkielet i kontrakt są ze sobą sprzeczne albo niejasne | kontrakcie | Zgłoś to orkiestratorowi. Rozstrzyga planner. |

3. Zapisz `Docs/Postep/NN-zadanie/weryfikacja-fala-F.md` (szablon niżej). Kolejne rundy dopisuj w tym samym pliku jako nową sekcję.

### Szablon poprawki kodu

Ten sam układ co prompt kodu, plus sekcja „Błąd” na początku:

````markdown
# 01-pary, poprawka 1

- **Trudność:** łatwe
- **Pliki (tylko te wolno zmieniać):** src/nanowriter/tokenizer.py
- **Prompt pierwotny:** Docs/Prompts/03-tokenizer/kod/01-pary.md

## Błąd
- Test: `tests/test_tokenizer_01_pary.py::test_get_pair_counts_short_inputs`
- Wynik: `IndexError: list index out of range` dla `get_pair_counts([])`
- Oczekiwane według docstringu: `{}`
- Hipoteza: pętla zakłada co najmniej jeden element.

## Gotowe, gdy
- `uv run pytest tests/test_tokenizer_01_pary.py` przechodzi.
````

Wykonawca poprawki może uruchamiać testy, ale nie może ich zmieniać.

### Szablon weryfikacji

```markdown
# Weryfikacja: zadanie 03, fala 1

## Runda 1
- **Komenda:** `uv run pytest` → 14 passed, 2 failed

| Test | Przyczyna | Błąd w | Działanie |
| --- | --- | --- | --- |
| test_get_pair_counts_short_inputs | IndexError dla [] | kodzie | poprawki/01-pary-1.md |
| test_split_ratio | zły import | teście | poprawione, przechodzi |

- **Problemy z kontraktem:** brak
- **Werdykt:** niegotowe
```

Odpowiedź dla orkiestratora: werdykt, ścieżki nowych plików w `poprawki/`, problemy z kontraktem albo „brak”.

## Katalog testów dla LLM

| Moduł | Test | Co sprawdza |
| --- | --- | --- |
| `tokenizer` | roundtrip | `decode(encode(x)) == x` dla `""`, `"Zażółć gęślą jaźń"`, emoji, `"\n\t"` |
| `tokenizer` | zakres id | każde id spełnia `0 <= id < vocab_size` |
| `tokenizer` | save/load | po `load` wynik `encode` jest identyczny |
| `dataset` | format `.bin` | dtype i długość zgodne z kontraktem |
| `dataset` | przesunięcie | `y[:, :-1] == x[:, 1:]` |
| `model` | kształty | `logits.shape == (B, T, vocab_size)` |
| `model` | maska przyczynowa | zmiana tokenu na pozycji `t` nie zmienia wyjść na pozycjach `< t` |
| `model` | strata startowa | po inicjalizacji strata ≈ `ln(vocab_size)` (±10%) |
| `model` | gradienty | po `backward` każdy parametr ma niezerowy `grad` |
| `model` | przeuczenie | na jednym batchu strata spada poniżej ~0,1 w kilkuset krokach |
| `train` | checkpoint | po zapisie i odczycie model daje identyczne `logits` |
| `train` | wznowienie | trening N kroków = N/2 + wznowienie + N/2 |
| `sample` | długość | `generate` zwraca dokładnie zadaną liczbę nowych tokenów |
| `sample` | powtarzalność | ten sam seed daje ten sam tekst |
| `sample` | KV cache | wynik z cache i bez cache jest identyczny |

## Diagnoza typowych porażek

Wpisz ją jako „Hipoteza” w poprawce.

| Objaw | Najczęstsza przyczyna |
| --- | --- |
| Strata nie spada wcale | brak `opt.zero_grad()`, `opt.step()` przed `backward`, LR = 0, parametry poza optymalizatorem |
| Strata `NaN` | za duży LR, maska z `-inf` w całym wierszu, przepełnienie w fp16 (użyj bf16) |
| Test maski przyczynowej nie przechodzi | maska `triu` zamiast `tril` albo maska nałożona po softmaksie |
| Strata startowa dużo wyższa niż `ln(V)` | za duża inicjalizacja wag wyjściowych |
| Nie przeucza się na jednym batchu | błąd w przesunięciu `y`, zły kształt w `cross_entropy`, zamrożone wagi |
| Roundtrip psuje polskie znaki | tokenizer działa na znakach zamiast na bajtach UTF-8 albo dekoduje token po tokenie |
| Wyniki różnią się między uruchomieniami | brak seeda, `model.train()` z dropoutem w teście porównującym |
