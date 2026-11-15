# Profil projektu: NanoWriter

Ten folder (`SKILLS/projekt/`) to jedyne miejsce z wiedzą o konkretnym projekcie. Skille agentów są ogólne i biorą stąd stack, komendy, moduły i etapy. W nowym projekcie podmieniasz `profil.md`, `konwencje.md`, `katalog-testow.md` i wybierasz pliki z `pulapki/`.

## Projekt

Mały LLM (decoder-only transformer) budowany od zera. Cel: model pisze płynny tekst w jednym stylu.

## Stack i sprzęt

- Python 3.13, uv, PyTorch (CUDA 12.8), pytest.
- GPU RTX 5060 Ti. System Windows 11, agenci uruchamiają komendy w Git Bash.
- Pakiet importuje się jako `nanowriter`.

## Komendy

| Co | Komenda |
| --- | --- |
| Uruchomienie Pythona | `uv run python ...` |
| Szybkie sprawdzenie | `uv run python -c "..."` |
| Wszystkie testy | `uv run pytest` |
| Jeden plik testów | `uv run pytest <plik>` |
| Sprawdzenie składni | `uv run python -m py_compile <plik>` |
| Sprawdzenie statyczne jednego pliku | `uv run ruff check <plik>` |
| Sprawdzenie statyczne | `uv run ruff check src tests && PYTHONUTF8=1 uv run lint-imports && uv run pylint src` |
| Mapa kodu | blok niżej |
| Szukanie istniejącego kodu | blok niżej |
| Nowa biblioteka | `uv add <pakiet>` (tylko gdy prompt tak mówi) |

Mapa kodu (klasy, funkcje i metody publiczne) oraz szukanie istniejącego kodu po słowach:

```bash
grep -rnE "^(class|def|async def) |^    (async )?def [a-z]" src/
grep -rniE "slowo1|slowo2" src/ Docs/Kontrakty/
```

`PYTHONUTF8=1` jest konieczne: bez niego `lint-imports` na Windowsie wywraca się z `UnicodeEncodeError`, gdy wyjście jest przekierowane. Sprawdzenie statyczne łapie nieużywane importy, podwójne definicje (ruff), import z wyższej warstwy i cykle (import-linter), a także skopiowane bloki kodu (pylint). Konfiguracja jest w `pyproject.toml`.

## Układ repozytorium

| Co | Gdzie |
| --- | --- |
| Kod modułu | `src/nanowriter/<modul>.py` |
| Testy jednego promptu | `tests/test_<modul>_<PP>_<nazwa>.py`, podkreślniki zamiast myślników, np. `tests/test_tokenizer_01_pary.py` |
| Wspólne fixture'y | `tests/conftest.py` |
| Skrypty (tylko wywołują moduły) | `scripts/` |
| Dane (poza gitem) | `data/` |
| Plan wstępny projektu | `Docs/Plan/00-wstepny-plan/wstępny_plan.md` |

## Pułapki obowiązujące w projekcie

`SKILLS/projekt/pulapki/python.md`, `SKILLS/projekt/pulapki/pytorch.md`

## Mapa modułów

Każdy moduł to plik kodu i kontrakt `Docs/Kontrakty/<modul>.md`.

| Moduł | Odpowiada za | Etap |
| --- | --- | --- |
| `data` | pobranie i czyszczenie tekstu, podział train/val | 2 |
| `tokenizer` | trening BPE, `encode`/`decode`, zapis słownika | 1 (znakowy), 3 (BPE) |
| `dataset` | pliki `.bin` z tokenami, losowanie batchy `(x, y)` | 3 |
| `model` | transformer: konfiguracja, `forward`, inicjalizacja wag | 1 (bigram), 4 |
| `train` | pętla treningu, optymalizator, harmonogram LR, checkpointy | 1, 5 |
| `sample` | generowanie: temperatura, top-k, top-p, KV cache | 1, 6 |
| `evaluate` | strata walidacyjna, perplexity, stały zestaw promptów | 7 |

## Warstwy

Moduł importuje tylko z warstw niższych od siebie. Kolejność od najwyższej do najniższej:

`evaluate → sample → train → model → dataset → tokenizer → data`

Skrypty w `scripts/` mogą importować z każdej warstwy. Reguła jest wymuszona przez import-linter (`[tool.importlinter]` w `pyproject.toml`). Nowy moduł wymaga wpisu tutaj i w `pyproject.toml`, więc to decyzja użytkownika.

## Etapy i testy bramki

Zadanie odpowiadające etapowi jest zamknięte dopiero wtedy, gdy przechodzi test bramki. Planner umieszcza go w ostatniej fali.

| Etap | Test bramki |
| --- | --- |
| 1 Bigram | Strata treningowa spada, `generate` zwraca tekst o zadanej długości. |
| 2 Dane | Podział train/val istnieje, zbiory się nie pokrywają. |
| 3 Tokenizer | `decode(encode(x)) == x` dla polskich znaków, emoji i pustego tekstu, a pliki `.bin` mają dtype z kontraktu. |
| 4 Transformer | Kształty się zgadzają, test maski przyczynowej przechodzi, model przeucza się na jednym batchu. |
| 5 Trening | Wznowienie z checkpointu daje tę samą stratę co trening bez przerwy. |
| 6 Generowanie | Z KV cache i bez niego wynik jest identyczny przy tym samym seedzie. |

Dla etapów 7–10 bramkę ustala planner.

## Decyzje użytkownika

Agenci ich nie podejmują, tylko pytają: język i rodzaj tekstu, źródło danych, rozmiar modelu.

## Ograniczenia

Użytkownik korzysta z komputera w trakcie pracy agentów. Zapytaj przed:
- treningiem dłuższym niż ~15 minut;
- pobieraniem więcej niż ~1 GB danych.

## Liczby do planowania

- **Parametry transformera** ≈ `12 · n_layer · d_model²` plus embedding `vocab_size · d_model` (raz, gdy wagi wyjścia są współdzielone). Przykład: 6 warstw, `d_model=384`, vocab 8192 → ≈ 14M.
- **Dane (Chinchilla):** ok. 20 tokenów na parametr, czyli 14M parametrów → ~280M tokenów.
- **Pamięć GPU na wagi i AdamW (fp32):** ok. 16 bajtów na parametr. Reszta to aktywacje, które rosną z `batch · kontekst · d_model · n_layer`. Gdy brakuje pamięci: mniejszy batch i akumulacja gradientów.
- **Start etapu 4:** 4 warstwy, 4 głowice, `d_model=256`, kontekst 256.
