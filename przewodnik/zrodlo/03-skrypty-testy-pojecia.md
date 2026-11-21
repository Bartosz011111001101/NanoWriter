
# 4. Skrypty: wejścia, wyjścia i decyzje

| Skrypt | Wejście | Co wywołuje | Wyjście |
| --- | --- | --- | --- |
| `scripts/prepare_data.py` | argumenty CLI; domyślnie API Wolnych Lektur | katalog, filtr, pobieranie, korpus i podział z `data.py` | `data/raw/`, `data/clean/`, `train.txt`, `val.txt`, `split.json` |
| `scripts/prepare_tokens.py` | `train.txt`, `val.txt`, parametry słownika i próbki | `BPETokenizer`, `encode_file`, zapis meta z `dataset.py` | tokenizer JSON, dwa pliki `.bin`, `meta.json` |
| `scripts/train_gpt.py` | `train.bin`, `val.bin`, `meta.json` i parametry | `GPT`, `TrainConfig`, `train`, `estimate_loss` | checkpoint i log treningu |
| `scripts/generate.py` | checkpoint, BPE JSON, prompt i parametry próbkowania | `load_gpt`, `BPETokenizer`, `generate` | tekst wypisany w terminalu |
| `scripts/train_bigram.py` | plik tekstowy albo książka z API, prompt | `CharTokenizer`, `BigramLanguageModel`, wspólne `train`/`generate` | tekst wygenerowany w terminalu |

Skrypty nie uruchamiają swojej głównej funkcji przy samym imporcie. `main()` jest wywoływane dopiero przy uruchomieniu pliku jako programu. To pozwala importować `parse_args` i `main` w testach.

> **Po co skrypty osobno od modułów?** Moduł ma odpowiadać na pytanie „jak to policzyć”, a skrypt na pytanie „skąd wziąć pliki i gdzie zapisać wynik”. Dzięki temu ten sam `train()` trenuje bigram, model 1M i rozmówcę 10M. Zmieniają się tylko skrypty, które go wołają.

## 4.1 Uruchomienie end-to-end

Instalacja/odtworzenie zależności:

```bash
uv sync
```

Następnie z katalogu repozytorium:

```bash
uv run python scripts/prepare_data.py
uv run python scripts/prepare_tokens.py --vocab-size 1024
uv run python scripts/train_gpt.py --device cuda --steps 100
uv run python scripts/generate.py --prompt "Wieczorem"
```

Pierwsza komenda potrzebuje sieci i pobiera teksty. Pomiń ją, jeśli nie chcesz teraz pobierać danych. `--steps 100` w przykładzie skraca trening demonstracyjny. Domyślne 5000 kroków może trwać znacznie dłużej. Do pracy bez GPU w poleceniach treningu i generowania podaj `--device cpu`. Wszystkie kroki wymagają zgodnych artefaktów utworzonych przez wcześniejszy etap.

Pomoc i dostępne parametry:

```bash
uv run python scripts/prepare_data.py --help
uv run python scripts/prepare_tokens.py --help
uv run python scripts/train_gpt.py --help
uv run python scripts/generate.py --help
```

Przykład punktu odniesienia:

```bash
uv run python scripts/train_bigram.py --text data/train.txt --steps 100
```

> **Sprawdź sam: ten sam model, różne ustawienia.** Uruchom trzy razy z tym samym seedem i różną temperaturą. Zobaczysz na żywo to, co wykres w rozdziale 8.8 pokazuje na liczbach:
>
> ```bash
> uv run python scripts/generate.py --prompt "Wieczorem" --seed 1 --temperature 0.5
> uv run python scripts/generate.py --prompt "Wieczorem" --seed 1 --temperature 1.0
> uv run python scripts/generate.py --prompt "Wieczorem" --seed 1 --temperature 1.5
> ```

# 5. Artefakty — co znajduje się na dysku

```text
data/
├── raw/                    pobrane teksty źródłowe
├── clean/                  pojedyncze oczyszczone książki
├── train.txt               tekst treningowy
├── val.txt                 tekst walidacyjny
├── split.json              książki w obu częściach i parametry podziału
├── tokens/
│   ├── tokenizer.json      reguły kodowania BPE
│   ├── train.bin           id treningowe uint16
│   ├── val.bin             id walidacyjne uint16
│   └── meta.json           słownik, dtype, liczby tokenów/znaków
├── checkpoints/
│   ├── gpt.pt              model, optymalizator, krok, RNG i konfiguracje
│   └── train_log.jsonl     rekordy strat i learning rate
└── rozmowca/ …             osobny zestaw plików modelu 10M (rozdział 9)
```

Te pliki łączy kontrakt kompatybilności:

- `train.bin` i `val.bin` muszą być zakodowane tym samym słownikiem, który jest w `tokenizer.json`;
- rozmiar słownika w `meta.json` ustawia `vocab_size` GPT;
- generowanie musi ładować tokenizer i checkpoint używane razem podczas treningu;
- do wznowienia służy checkpoint, nie sam plik wag pozbawiony stanu optymalizatora.

`data/`, `.bin` oraz checkpointy są ignorowane przez Git. Pobieranie i trening zachodzą lokalnie, a wygenerowane wyniki nie są automatycznie publikowane.

> **Po co zapisywać pośrednie pliki, zamiast liczyć wszystko w jednym przebiegu?** Każdy etap trwa inaczej: pobranie kilka minut, tokenizacja kilka minut, trening kilkadziesiąt. Gdy zmieniasz tylko parametry treningu, nie chcesz od nowa pobierać książek i trenować tokenizera. Pliki pośrednie są punktami, od których można zacząć ponownie.

> **Jak zajrzeć do `train_log.jsonl`.** To zwykły tekst, jeden rekord JSON na linię. Pierwsza linia Twojego modelu 1M to `{"step": 0, "train_loss": 6.946, "lr": 1e-05}`, ostatnia walidacja: `{"step": 4999, "val_loss": 3.112}`. Ten plik rysuje wykres w rozdziale 8.7.

# 6. Testy — jak wykorzystać je do nauki

Uruchomienie całości:

```bash
uv run pytest
```

Testy są przykładową dokumentacją zachowania: pokazują oczekiwane kształty, przypadki brzegowe, deterministyczność i zależności etapów. Wiele testów podstawia kontrolowane odpowiedzi zamiast łączyć się z API. Nie trzeba pobierać korpusu, żeby sprawdzić większość kodu.

| Plik testów | Zakres |
| --- | --- |
| `test_data_01_pobranie_ksiazki.py` | URL-e, pobieranie, UTF-8, końce linii, pliki istniejące i błędne odpowiedzi. |
| `test_data_01_katalog_i_czyszczenie.py` | Katalog, selekcja gatunków, normalizacja i czyszczenie. |
| `test_data_02_korpus_i_podzial.py` | Raport korpusu, limity, duplikaty oraz rozłączny train/val. |
| `test_tokenizer_02_tokenizer_znakowy.py` | Słownik znakowy, zakres id, błędne znaki i round-trip. |
| `test_tokenizer_01_bpe_tokenizer.py` | Trening BPE, deterministyczne scalenia, kodowanie i plik słownika. |
| `test_dataset_02_pliki_bin.py` | Batch, uint16, pliki binarne, meta i walidacja granic. |
| `test_model_03_model_bigram.py` | Wymiary bigramu, liczba parametrów, predykcje i gradienty. |
| `test_model_01_gpt.py` | Konfiguracja, maska, kształty, skład bloków, cache i uczenie na małym batchu. |
| `test_train_04_batch_i_trening.py` | Zachowanie batchowania i podstawowych elementów treningu. |
| `test_train_01_trening_pelny.py` | Harmonogram, optymalizator, ewaluacja, checkpoint i powtarzalność treningu. |
| `test_sample_01_kv_cache_i_probkowanie.py` | Filtry próbkowania i porównanie cache z generowaniem bez cache. |
| `test_sample_05_generate.py` | Długość, kontekst, seed, tryb modelu i nieprawidłowe wejścia `generate`. |
| `test_prepare_tokens_03_skrypt_tokenow.py` | Integracja skryptu tokenizacji i format wszystkich wyników. |
| `test_train_bigram_06_skrypt_bigram.py` | CLI bigramu, spadek straty i przykłady generowania. |
| `test_train_gpt_02_skrypt_treningu.py` | CLI GPT, logi, checkpointy i równoważność wznowienia. |
| `test_generate_02_skrypt_generowania.py` | Połączenie plików modelu/tokenizera z generowaniem CLI. |
| `test_train_01_strata_z_maska.py`, `test_dataset_02_dane_z_maska.py` | Strata liczona tylko tam, gdzie maska = 1 (rozmówca). |
| `test_chat_02_skrypt_czatu.py`, `test_finetune_rozmowca_02_skrypt_douczania.py` | Czat i douczanie rozmówcy. |

Przy nauce przydatny porządek to uruchomić najpierw testy tokenizerów, potem dataset, model, trening i na końcu testy skryptów. Test modułu odpowiada na pytanie „czy ta część działa?”, a test skryptu: „czy części poprawnie współpracują?”.

> **Najciekawsze testy do przeczytania przy nauce.** W `test_model_01_gpt.py` przeczytaj `test_causal_mask_hides_future`: zmienia token na pozycji 10 i sprawdza, że wyniki na pozycjach 0–9 się nie zmieniły, a od 10 w górę tak. To jest definicja „model nie widzi przyszłości” zapisana w kodzie. Drugi to `test_overfits_single_batch`: model przez 300 kroków uczy się jednego batcha i ma zejść ze straty powyżej 3 poniżej 0,1. Jeśli tego nie potrafi, w modelu jest błąd, a nie za mało danych.

# 7. Pojęcia i nazwy używane w kodzie

| Pojęcie | Znaczenie w tym projekcie |
| --- | --- |
| Korpus | Zbiór tekstów używany do uczenia i oceny modelu. |
| Train / trening | Dane, na których aktualizowane są wagi. |
| Validation / val | Oddzielne dane do pomiaru jakości bez aktualizacji wag. |
| Przeciek danych | Sytuacja, gdy informacja z walidacji trafia do treningu i wynik walidacji staje się zbyt optymistyczny. |
| Tokenizer | Reguły zamiany tekstu w id i id w tekst. |
| Token | Jednostka słownika: bajt, znak albo wyuczony fragment tekstu. |
| Vocabulary / vocab | Słownik tokenizera. `vocab_size` to liczba możliwych id. |
| BPE | Algorytm tworzący tokeny przez scalanie częstych par. |
| ID / token id | Liczba całkowita reprezentująca token. |
| Prompt | Tekst początkowy podany generatorowi. |
| Kontekst / block size | Maksymalna liczba pozycji, które model bierze pod uwagę naraz. |
| Sekwencja | Uporządkowane id tokenów, np. jedno zdanie po tokenizacji. |
| Batch | Kilka przykładów przetwarzanych razem w jednym kroku optymalizacji. |
| Tensor | Wielowymiarowa tablica PyTorch. Opis kształtu mówi o jej wymiarach. |
| Shape / kształt | Liczby wymiarów i ich rozmiary, np. `[B,T,V]`. |
| Embedding | Wyuczony wektor opisujący token albo jego pozycję. |
| Parametr / waga | Wartość sieci zmieniana podczas treningu. |
| Logit | Surowy wynik dla możliwej klasy/tokena, jeszcze nie prawdopodobieństwo. |
| Softmax | Przekształcenie logitów w dodatnie wartości sumujące się do 1. |
| Loss / strata | Liczba opisująca błąd przewidywań względem poprawnych celów. |
| Gradient | Pochodna wskazująca, jak zmiana parametru zmienia stratę. |
| Backpropagation | Obliczenie gradientów od straty wstecz przez warstwy modelu. |
| Optymalizator | Algorytm aktualizujący parametry na podstawie gradientów, tu AdamW. |
| Learning rate (LR) | Skala kroku, o jaki optymalizator zmienia parametry. |
| Warmup | Stopniowe zwiększenie LR na początku treningu. |
| Weight decay | Regularyzacja ograniczająca nadmierny wzrost niektórych wag. |
| Gradient clipping | Ograniczenie normy gradientu, aby pojedynczy krok nie był za duży. |
| Epoch | Jedno przejście przez cały zbiór treningowy. Ten projekt trenuje krokami losowanych batchy. |
| Bigram | Model zależny bezpośrednio od jednego poprzedniego tokenu. |
| Transformer | Architektura sieci używająca uwagi do łączenia informacji z pozycji sekwencji. |
| Decoder-only | Transformer używany do przewidywania dalszego ciągu z wcześniejszego kontekstu. |
| Attention / uwaga | Ważona agregacja informacji z innych pozycji sekwencji. |
| Causal mask | Maska blokująca dostęp do przyszłych pozycji. |
| Głowica uwagi | Jedna równoległa przestrzeń obliczania uwagi. |
| Q / K / V | Query, key i value: wejściowe projekcje używane przez uwagę. |
| MLP | Mała sieć w pełni połączona stosowana osobno w każdej pozycji. |
| GELU | Funkcja nieliniowa w MLP. |
| Residual connection | Dodanie wejścia podwarstwy do jej wyniku. |
| LayerNorm | Normalizacja wektora cech, pomagająca stabilizować uczenie. |
| Pre-norm | Układ, w którym LayerNorm jest przed uwagą i MLP. |
| Weight tying | Współdzielenie wag embeddingu tokenów i warstwy wyjściowej. |
| Checkpoint | Zapis stanu treningu potrzebny do odtworzenia modelu lub wznowienia pracy. |
| Seed | Ziarno RNG. Umożliwia powtarzalne losowania, gdy reszta warunków jest taka sama. |
| Autoregresja | Generowanie następnego tokenu na podstawie już dostępnych tokenów. |
| Teacher forcing | Podawanie poprawnej historii z danych podczas uczenia przewidywania następnego tokenu. |
| Overfitting | Dopasowanie do train bez poprawy na nowych danych. |
| Perplexity | e do potęgi średniej straty. Mówi, „między iloma tokenami model się waha”. Model 1M: e^3,11 ≈ 22. |
| Bits per character | Strata wyrażona orientacyjnie w bitach na znak, z uwzględnieniem liczby tokenów na znak. |
| Temperatura | Skala zmieniająca ostrość rozkładu podczas losowania. |
| Top-k / top-p | Ograniczenia zbioru tokenów, z których wolno losować następny token. |
| KV cache | Pamięć kluczy i wartości uwagi wcześniejszych pozycji podczas generowania. |
| Pretrening | Pierwszy, długi trening na dużym, ogólnym korpusie. Model uczy się języka. |
| Douczanie (fine-tuning) | Krótki trening już wytrenowanego modelu na małym, wybranym zbiorze. Model uczy się stylu lub zadania. |
| Maska straty | Zera i jedynki mówiące, które pozycje liczą się do straty. Przy douczaniu rozmówcy: tylko odpowiedzi B. |
| Reguła Chinchilla | Orientacyjnie ok. 20 tokenów treningowych na parametr modelu. |
| CUDA | Technologia uruchamiania obliczeń PyTorch na zgodnym GPU NVIDIA. |
| bfloat16 | 16-bitowy format zmiennoprzecinkowy używany opcjonalnie do obliczeń na GPU. |
| Little-endian | Kolejność bajtów, w której mniej znaczący bajt liczby jest zapisany jako pierwszy. |
| CLI | Interfejs wiersza poleceń: argumenty przekazywane skryptom po nazwie programu. |
