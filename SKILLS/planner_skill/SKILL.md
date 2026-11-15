---
name: planner_skill
description: Używaj, gdy etap lub funkcję NanoWriter trzeba rozpisać na zadania, oszacować rozmiar modelu, danych albo pamięci GPU, albo gdy trzeba napisać lub zmienić kontrakt modułu.
---

# Planowanie w NanoWriter

## Mapa modułów

Każdy moduł to jeden plik `src/nanowriter/<modul>.py` i jeden kontrakt `Docs/Kontrakty/<modul>.md`.

| Moduł | Odpowiada za | Etap |
| --- | --- | --- |
| `data` | pobranie i czyszczenie tekstu, podział train/val | 2 |
| `tokenizer` | trening BPE, `encode`/`decode`, zapis słownika | 1 (znakowy), 3 (BPE) |
| `dataset` | pliki `.bin` z tokenami, losowanie batchy `(x, y)` | 3 |
| `model` | transformer: konfiguracja, `forward`, inicjalizacja wag | 1 (bigram), 4 |
| `train` | pętla treningu, optymalizator, harmonogram LR, checkpointy | 1, 5 |
| `sample` | generowanie: temperatura, top-k, top-p, KV cache | 1, 6 |
| `evaluate` | strata walidacyjna, perplexity, stały zestaw promptów | 7 |

Skrypty uruchamiające moduły trafiają do `scripts/`. Nie zawierają logiki, tylko ją wywołują.

## Jak ciąć etap na zadania

- Najpierw kontrakt, potem najprostsza działająca wersja, potem ulepszenia. Przykład: ręczna uwaga przed `scaled_dot_product_attention`, pozycje uczone przed RoPE.
- Jedno zadanie to jedna zmiana zachowania, którą da się sprawdzić jednym albo kilkoma testami. To mniej więcej godzina pracy.
- Każde zadanie wskazuje: pliki do zmiany, kontrakt, którego dotyczy, i test potwierdzający wykonanie.
- Optymalizacje (`torch.compile`, bf16, KV cache) to osobne zadania, zawsze po wersji, która już działa i ma testy.
- Ostatnie zadanie etapu to test bramki z `SKILLS/orchestrator_skill/SKILL.md`.

## Szablon planu etapu

Plik `Docs/Plan/etap-N.md`:

```markdown
# Etap N: <nazwa>

## Cel
<jedno zdanie> Gotowe, gdy: <coś do uruchomienia i zobaczenia>.

## Zadania
### N.1 <nazwa>
- Pliki: src/nanowriter/<modul>.py, tests/test_<modul>.py
- Kontrakt: Docs/Kontrakty/<modul>.md (sekcja ...)
- Test: <co dokładnie sprawdza>

## Kolejność
N.1 → N.2 → N.3 (N.4 zależy od N.2)

## Otwarte pytania
- <decyzja dla użytkownika> albo "brak"
```

## Szablon kontraktu

Plik `Docs/Kontrakty/<modul>.md`, sekcje w tej kolejności: Odpowiedzialność, Publiczne API, Tensory, Pliki, Niezmienniki, Zmiany. Przykład:

```markdown
# Kontrakt: tokenizer

## Odpowiedzialność
Zamienia tekst na listę id tokenów i z powrotem, bez strat.

## Publiczne API
- `class Tokenizer`
  - `train(text: str, vocab_size: int) -> None`
  - `encode(text: str) -> list[int]`
  - `decode(ids: list[int]) -> str`
  - `save(path: Path) -> None`, `@classmethod load(path: Path) -> Tokenizer`
  - `vocab_size: int`

## Tensory
Brak (zwraca listy int, tensory tworzy `dataset`).

## Pliki
- `tokenizer.json`: UTF-8 JSON, `{"merges": [[a, b], ...], "vocab_size": int}`.

## Niezmienniki
- `decode(encode(x)) == x` dla każdego `str`, także pustego, z polskimi znakami i emoji.
- Każde id spełnia `0 <= id < vocab_size`.
- Po `save` i `load` `encode` daje identyczny wynik.

## Zmiany
- RRRR-MM-DD: pierwsza wersja.
```

Zmiana istniejącego kontraktu to wpis w sekcji „Zmiany” z datą. W odpowiedzi dla orkiestratora wymień też testy, które trzeba zaktualizować.

## Liczby do planowania

- **Parametry transformera** ≈ `12 · n_layer · d_model²` (uwaga `4d²` + MLP `8d²` na warstwę) plus embedding `vocab_size · d_model` (raz, jeśli wagi wyjścia są współdzielone).
  - Przykład: 6 warstw, `d_model=384`, vocab 8192 → ok. 10,6M + 3,1M ≈ 14M.
- **Dane (Chinchilla):** ok. 20 tokenów na parametr. Model 14M parametrów potrzebuje ~280M tokenów. Przy mniejszej ilości danych zaplanuj mniejszy model albo kilka epok i zwróć uwagę na przeuczenie.
- **Pamięć GPU na wagi i optymalizator (AdamW, fp32):** ok. 16 bajtów na parametr. 14M parametrów to ~0,22 GB. Resztę pamięci zajmują aktywacje, które rosną z `batch · kontekst · d_model · n_layer`. Gdy pamięci brakuje, zmniejsz batch i włącz akumulację gradientów.
- **Konfiguracja startowa (etap 4):** 4 warstwy, 4 głowice, `d_model=256`, kontekst 256.

## Częste błędy

| Błąd | Poprawka |
| --- | --- |
| Zadanie „napisz model” | Podziel na: embedding, uwaga, blok, cały model, test przeuczenia. |
| Kontrakt bez kształtów tensorów | Każdy tensor ma kształt i dtype, np. `idx: LongTensor[B, T]`. |
| Zadanie bez testu | Dopisz, co test sprawdza. Jeśli nie da się tego sprawdzić, zadanie jest źle wycięte. |
| Optymalizacja przed działającą wersją | Przenieś ją na koniec etapu. |
| Decyzja użytkownika podjęta przez plannera | Przenieś ją do „Otwarte pytania”. |
