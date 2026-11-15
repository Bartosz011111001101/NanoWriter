---
name: planner_skill
description: Używaj, gdy zadanie NanoWriter trzeba rozpisać na fale i prompty ze szkieletami funkcji dla wykonawców (Haiku, Sonnet, Opus), ocenić trudność promptu, oszacować rozmiar modelu, danych albo pamięci GPU, albo napisać lub zmienić kontrakt modułu.
---

# Planowanie w NanoWriter

Wykonawcami są głównie Haiku i Sonnet. Prompt ma być tak kompletny, żeby wykonawca niczego nie wymyślał: nazwy funkcji, argumenty, typy i zwracane wartości daje szkielet.

## Co tworzysz

```
Docs/Plan/NN-zadanie/plan.md
Docs/Prompts/NN-zadanie/kod/PP-nazwa.md      jeden plik na kawałek pracy
Docs/Kontrakty/<modul>.md                    nowe albo zmienione kontrakty
```

`Docs/` jest w `.gitignore`. Listuj przez `ls`, czytaj po pełnej ścieżce. Promptów testów nie piszesz, robi to tester. Ty w każdym prompcie kodu zostawiasz sekcję „Propozycja testów”.

## Mapa modułów

Każdy moduł to plik `src/nanowriter/<modul>.py` i kontrakt `Docs/Kontrakty/<modul>.md`.

| Moduł | Odpowiada za | Etap |
| --- | --- | --- |
| `data` | pobranie i czyszczenie tekstu, podział train/val | 2 |
| `tokenizer` | trening BPE, `encode`/`decode`, zapis słownika | 1 (znakowy), 3 (BPE) |
| `dataset` | pliki `.bin` z tokenami, losowanie batchy `(x, y)` | 3 |
| `model` | transformer: konfiguracja, `forward`, inicjalizacja wag | 1 (bigram), 4 |
| `train` | pętla treningu, optymalizator, harmonogram LR, checkpointy | 1, 5 |
| `sample` | generowanie: temperatura, top-k, top-p, KV cache | 1, 6 |
| `evaluate` | strata walidacyjna, perplexity, stały zestaw promptów | 7 |

Skrypty w `scripts/` tylko wywołują moduły, bez własnej logiki.

## Cięcie na prompty i fale

- Jeden prompt to jedna do trzech funkcji albo jedna mała klasa.
- **Fala** to prompty, które mogą iść równolegle. Prompty w jednej fali:
  - nie zmieniają tych samych plików;
  - nie zależą od siebie (żaden nie używa funkcji, którą pisze inny w tej samej fali).
- Prompt korzystający z funkcji z wcześniejszego promptu idzie do późniejszej fali.
- Gdy dwa niezależne prompty muszą zmienić ten sam plik, daj je do różnych fal.
- Najpierw najprostsza działająca wersja, optymalizacje (`torch.compile`, bf16, KV cache) w późniejszych falach.
- Ostatnia fala zadania zawiera test bramki z `SKILLS/orchestrator_skill/SKILL.md`, jeśli zadanie zamyka etap.

## Trudność

| Trudność | Model | Kiedy | Przykłady |
| --- | --- | --- | --- |
| łatwe | Haiku | Szkielet kompletny, ciało wynika wprost z kroków, bez decyzji projektowych. | zliczanie par BPE, podział train/val, zapis tablicy do `.bin` |
| średnie | Sonnet | Kilka współpracujących funkcji albo klasa, operacje na kształtach tensorów, drobne decyzje wewnątrz szkieletu. | multi-head attention, pętla treningu z ewaluacją, top-k/top-p |
| trudne | Opus | Łatwo o cichy błąd w poprawności albo wydajności. | KV cache zgodny z wersją bez cache, wznowienie treningu ze stanem RNG, szybki trening BPE na dużym korpusie |

Wahasz się między łatwe a średnie? Podziel prompt na mniejsze, zamiast podnosić trudność. „Trudne” to wyjątek.

## Szablon: plan.md

```markdown
# Zadanie NN: <nazwa>

## Cel
<jedno zdanie>. Gotowe, gdy: <coś do uruchomienia i zobaczenia>.

## Fale
| Fala | Prompt | Trudność | Pliki |
| --- | --- | --- | --- |
| 1 | 01-pary | łatwe | src/nanowriter/tokenizer.py |
| 1 | 02-podzial | łatwe | src/nanowriter/data.py |
| 2 | 03-merge | średnie | src/nanowriter/tokenizer.py |

## Kontrakty
- utworzone/zmienione: Docs/Kontrakty/tokenizer.md

## Otwarte pytania
- <decyzja dla użytkownika> albo "brak"
```

## Szablon: prompt kodu

Sekcje w tej kolejności. Każda funkcja w szkielecie ma pełną sygnaturę z typami i docstring z przykładem wejście → wyjście.

````markdown
# 01-pary

- **Trudność:** łatwe
- **Fala:** 1
- **Zależy od:** brak
- **Pliki (tylko te wolno zmieniać):** src/nanowriter/tokenizer.py
- **Kontrakt:** Docs/Kontrakty/tokenizer.md

## Cel
Zliczanie sąsiednich par id, potrzebne do wyboru następnego merge'a BPE.

## Szkielet
```python
def get_pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    """Zlicza wystąpienia sąsiednich par id.

    Przykład: [1, 2, 1, 2] -> {(1, 2): 2, (2, 1): 1}
    Pusta lista albo jeden element -> {}
    """
```

## Kroki
1. Utwórz pusty słownik.
2. Przejdź po parach `zip(ids, ids[1:])` i zwiększaj licznik każdej pary.
3. Zwróć słownik.

## Gotowe, gdy
- `uv run python -c "from nanowriter.tokenizer import get_pair_counts; print(get_pair_counts([1,2,1,2]))"` wypisuje `{(1, 2): 2, (2, 1): 1}`.

## Propozycja testów (dla testera)
- przykład z docstringu
- pusta lista i jeden element dają `{}`
- para powtórzona trzy razy pod rząd: `[5, 5, 5, 5]` -> `{(5, 5): 3}`
````

## Szablon: kontrakt

Plik `Docs/Kontrakty/<modul>.md`, sekcje: Odpowiedzialność, Publiczne API (sygnatury z typami), Tensory (kształt i dtype, np. `idx: LongTensor[B, T]`), Pliki (formaty), Niezmienniki (co tester ma sprawdzić), Zmiany (data + opis). Szkielety w promptach muszą być zgodne z kontraktem. Zmiana kontraktu to wpis w „Zmiany”.

## Liczby do planowania

- **Parametry transformera** ≈ `12 · n_layer · d_model²` plus embedding `vocab_size · d_model` (raz, gdy wagi wyjścia są współdzielone). Przykład: 6 warstw, `d_model=384`, vocab 8192 → ≈ 14M.
- **Dane (Chinchilla):** ok. 20 tokenów na parametr, czyli 14M parametrów → ~280M tokenów.
- **Pamięć GPU na wagi i AdamW (fp32):** ok. 16 bajtów na parametr. Reszta to aktywacje, które rosną z `batch · kontekst · d_model · n_layer`. Gdy brakuje pamięci: mniejszy batch i akumulacja gradientów.
- **Start etapu 4:** 4 warstwy, 4 głowice, `d_model=256`, kontekst 256.

## Odpowiedź dla orkiestratora

1. Ścieżka planu i lista fal (numer fali i nazwy promptów).
2. Utworzone i zmienione kontrakty.
3. Otwarte pytania albo „brak”.

## Częste błędy

| Błąd | Poprawka |
| --- | --- |
| Szkielet bez typów albo bez przykładu w docstringu | Haiku zacznie zgadywać. Dopisz typy i przykład wejście → wyjście. |
| Dwa prompty jednej fali zmieniają ten sam plik | Przenieś jeden do następnej fali. |
| Prompt używa funkcji z promptu tej samej fali | Przenieś go do następnej fali. |
| „Gotowe, gdy” wymaga testów | Wykonawca kodu nie pisze testów. Daj komendę `uv run python -c ...`. |
| Decyzja użytkownika podjęta przez plannera | Przenieś ją do „Otwarte pytania”. |
