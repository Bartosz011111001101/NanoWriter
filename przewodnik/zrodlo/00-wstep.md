---
title: "NanoWriter — przewodnik po kodzie i podstawach LLM"
subtitle: "Materiał do nauki: co robi projekt, jak łączą się pliki i dlaczego"
lang: pl
---

# Jak korzystać z tego przewodnika

Ten tekst objaśnia bieżącą implementację NanoWriter, a nie ogólną specyfikację każdego nowoczesnego LLM. Najpierw opisuje pliki i przebieg programu, potem pojęcia potrzebne do zrozumienia kodu, a na końcu składa te elementy w teorię działania modelu autoregresyjnego. Nowy rozdział 9 opisuje drugi model, rozmówcę z ok. 10 mln parametrów.

Zakładamy podstawową znajomość Pythona. Nie trzeba wcześniej znać PyTorch, rachunku macierzowego ani uczenia maszynowego. Wzory są uproszczone do poziomu potrzebnego, by zrozumieć przepływ danych.

Wszystkie ścieżki są względne wobec katalogu głównego repozytorium. Przykładowe komendy uruchamia się właśnie z tego katalogu.

**Co nowego w tej wersji.** Przy ważnych miejscach są cztery rodzaje ramek:

> **Po co?** Jaki problem rozwiązuje ten kawałek kodu.

> **Bez tego…** Co konkretnie by się zepsuło, gdyby go nie było.

> **Twój model.** Prawdziwe liczby z modelu wytrenowanego w tym repozytorium (`data/checkpoints/gpt.pt`, ok. 1 mln parametrów, oraz rozmówca `data/rozmowca/`, ok. 10 mln).

> **Sprawdź sam.** Krótka komenda, która pokazuje opisane zjawisko na Twoim komputerze.

Obrazki zrobiono z prawdziwych modeli skryptami z folderu `przewodnik/` (np. `uv run python -m przewodnik.bpe`). Te same skrypty zapisują też animowane wersje w `przewodnik/wyniki/*.gif`. PDF ich nie odtwarza, ale można je otworzyć w przeglądarce.

**Najlepsza kolejność czytania przy pierwszym podejściu:** rozdział 1 (obraz całości), potem od razu rozdział 8 (teoria, krok po kroku), a dopiero potem rozdział 3 (moduły). Teoria podaje sens, a moduły pokazują, gdzie ten sens siedzi w kodzie.

# 1. Obraz całości

NanoWriter realizuje pełny, choć mały, pipeline językowy:

```text
tekst książek
  → czyszczenie i podział na train/val
  → tokenizer: tekst na identyfikatory
  → dataset: pliki tokenów i przykłady (x, y)
  → model: przewidywanie kolejnego tokenu
  → trening: poprawianie wag na podstawie błędu
  → generowanie: wielokrotne wybieranie kolejnego tokenu
```

Cały LLM sprowadza się do jednej umiejętności: **zgadnij następny kawałek tekstu**. Wszystko inne (tokenizer, uwaga, trening, próbkowanie) albo przygotowuje dane do tego zgadywania, albo uczy model lepiej zgadywać, albo zamienia zgadywanie w pisanie tekstu.

> **Po co cały pipeline?** Sieć neuronowa umie tylko mnożyć liczby. Nie rozumie liter. Każdy etap zamienia problem „napisz tekst” na coś, co da się policzyć: tekst → liczby → prawdopodobieństwa → błąd → poprawka wag.

W repozytorium są dwa modele:

- **Bigram** jest małym punktem odniesienia. Bieżący znak/token wskazuje, jakie tokeny mogą wystąpić zaraz po nim. Nie analizuje długiej historii.
- **GPT** jest decoder-only Transformerem. Każda pozycja może korzystać z wcześniejszego kontekstu, a kolejne bloki przetwarzają reprezentacje tekstu.

> **Po co bigram, skoro jest GPT?** Żeby mieć z czym porównać. Bigram po „k” zawsze zgaduje tak samo, niezależnie od tego, czy wcześniej było „Wieczorem stary ”, czy „Ala ma ”. Różnica między stratą bigramu a stratą GPT pokazuje, ile daje patrzenie na dłuższy kontekst.

Model nie jest gotowym asystentem i nie weryfikuje prawdziwości wypowiedzi. Repozytorium służy przede wszystkim do nauki, uruchamiania małych eksperymentów i obserwowania poszczególnych etapów.

## 1.1 Pięć programów, pięć zadań

```text
scripts/prepare_data.py   : źródło → oczyszczone train.txt / val.txt
scripts/prepare_tokens.py : tekst → tokenizer.json, train.bin, val.bin, meta.json
scripts/train_gpt.py      : tokeny → GPT → checkpoint i log treningu
scripts/generate.py       : checkpoint + tokenizer + prompt → tekst
scripts/train_bigram.py   : tekst → prosty model znakowy → tekst
```

Pierwsze cztery programy tworzą główną ścieżkę GPT. `train_bigram.py` jest alternatywnym, prostszym doświadczeniem i nie potrzebuje etapu BPE. Skrypty rozmówcy (rozdział 9) to osobna ścieżka obok tej.

> **Twój model (1M).** Po przejściu całej ścieżki masz: 9 813 598 tokenów treningowych z 22,75 mln znaków książek, model z 957 184 parametrami i stratę walidacyjną 3,11 po 5000 krokach.

# 2. Mapa repozytorium

```text
src/nanowriter/       logika biblioteki: dane, tokenizacja, model, trening, generowanie
scripts/              programy CLI, które łączą funkcje biblioteki
tests/                testy modułów i testy integracyjne skryptów
przewodnik/           skrypty obrazków do tego przewodnika
data/                 lokalne teksty, tokeny i checkpointy (nie są śledzone przez Git)
pyproject.toml        metadane, zależności i konfiguracja narzędzi
uv.lock               przypięte rozwiązane wersje zależności
README.md             krótszy przewodnik po projekcie
.gitignore            wykluczenia danych i wyników treningu
```

`src/nanowriter/` nie uruchamia sam z siebie pobierania danych ani treningu. Zawiera funkcje i klasy, które mogą być wywoływane przez różne skrypty oraz testy. Skrypty rozpoznają argumenty wiersza poleceń, ustawiają domyślne ścieżki, wyświetlają postęp i delegują pracę do modułów.

## 2.1 Zależności między plikami

Pierwszy diagram pokazuje przebieg pracy i pliki wynikowe:

```text
prepare_data.py
  └── data.py
      ├── data/raw/<slug>.txt
      ├── data/clean/<slug>.txt
      └── data/train.txt, data/val.txt, data/split.json
prepare_tokens.py
  ├── data.py       (wczytanie tekstu)
  ├── tokenizer.py  (uczenie i zapis BPE)
  └── dataset.py    (kodowanie do .bin i metadane)
      └── data/tokens/{tokenizer.json, train.bin, val.bin, meta.json}
train_gpt.py
  ├── dataset.py    (wczytanie binów i meta.json)
  ├── model.py      (utworzenie GPT)
  └── train.py      (optymalizator, trening, walidacja i zapis)
      └── data/checkpoints/{gpt.pt, train_log.jsonl}
generate.py
  ├── tokenizer.py  (wczytanie BPE i kodowanie promptu)
  ├── train.py      (odtworzenie GPT przez load_gpt)
  └── sample.py     (wybór następnych tokenów)
train_bigram.py
  ├── data.py, tokenizer.py, model.py
  └── train.py, sample.py
```

Zależności pomiędzy bibliotekami są celowo węższe:

```text
train.py ───→ dataset.py ───→ data.py
   │              └ - - - -→ tokenizer.py (tylko adnotacje typów)
   └──────────→ model.py
sample.py ───→ model.py
```

`dataset.py` potrzebuje z `data.py` funkcji `load_text`. Nazwy klas tokenizerów importuje tylko pod `TYPE_CHECKING`, aby opisać typy parametrów bez runtime'owego importu. `train.py` używa `get_batch` z `dataset.py` oraz klas GPT z `model.py`. `sample.py` zna z `model.py` klasę `KVCache`. Skrypty mogą importować kilka modułów, bo to one są miejscem składania aplikacji.

> **Po co ta reguła „tylko w dół”?** Moduły tworzą warstwy: `evaluate → sample → train → model → dataset → tokenizer → data`. Wyższy moduł może importować niższy, nigdy odwrotnie. Pilnuje tego narzędzie `lint-imports`.
>
> **Bez tego…** `model.py` mógłby zaimportować `train.py`, a `train.py` z kolei `model.py`. Taki cykl importów działa do pierwszej zmiany, a potem kończy się błędem `ImportError` w losowym miejscu. Do tego testu modelu nie dałoby się uruchomić bez wczytania pętli treningowej.

## 2.2 Dlaczego taki podział?

- **Jedna odpowiedzialność:** funkcja czyszcząca tekst nie musi znać sieci neuronowej.
- **Testowalność:** można testować BPE bez GPU i testować trening na sztucznym, małym tensorze bez pobierania książek.
- **Wielokrotne użycie:** `train.py` trenuje zarówno bigram, jak i GPT, bo przyjmuje dowolny `torch.nn.Module` z odpowiednim `forward`. Ta sama pętla trenuje też rozmówcę 10M.
- **Trwałe artefakty:** tokeny, słownik i checkpoint można sprawdzić, zachować i użyć ponownie bez pobierania tych samych książek.
- **Zgodność etapów:** tokenizer użyty do kodowania musi być tym samym tokenizerem, który później dekoduje wygenerowane identyfikatory.

> **Bez tego (zgodność etapów)…** Gdy wygenerujesz tekst modelem 10M, ale zdekodujesz go tokenizerem 1M, dostaniesz bełkot. Id 700 w jednym słowniku to zupełnie inny kawałek tekstu niż id 700 w drugim. Model nie zgłosi błędu, bo dla niego to tylko liczby.
