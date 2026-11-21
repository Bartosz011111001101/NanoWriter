
## 3.4 `src/nanowriter/model.py` — modele i tensorowe obliczenia

### 3.4.1 Model bigramowy

`BigramLanguageModel` jest tablicą parametrów o rozmiarze `[V, V]`, gdzie V to rozmiar słownika. Wiersz wybrany przez id bieżącego tokenu zawiera logity wszystkich możliwych następnych tokenów. Nie ma osobnej pamięci całej sekwencji: dwa konteksty kończące się tym samym tokenem dadzą ten sam rozkład.

> **Jak to sobie wyobrazić.** To dosłownie tabela „po literze X zwykle jest litera Y”. Wiersz „k” mówi: po „k” często jest „a”, „o”, „i”, rzadko „ż”. Trening wypełnia tę tabelę liczbami. Nic więcej tam nie ma.

### 3.4.2 `GPTConfig`

Konfiguracja bazowa `GPTConfig()` ma `vocab_size=1024`, `block_size=256`, `n_layer=4`, `n_head=4` i `d_model=128`. Skrypt `train_gpt.py` pobiera faktyczny rozmiar słownika z `meta.json`, a pozostałe wartości można podać w CLI. `d_model` musi dzielić się przez liczbę głowic.

> **Twój model (1M).** Właśnie ta konfiguracja daje **957 184 parametry**. Gdzie one siedzą, pokazuje wykres w rozdziale 9: ok. 14% to embedding tokenów, 28% uwaga, 55% MLP. Najwięcej „wiedzy” mieści się więc w MLP, a nie w uwadze.

### 3.4.3 Przepływ kształtów przez GPT

Dla batcha o kształcie `[B, T]`:

```text
ids                 [B, T]
token embedding     [B, T, d_model]
pozycje             [T, d_model]  (dodane do embeddingu tokenów)
wynik GPT           [B, T, vocab_size]
```

Wewnątrz uwagi `d_head = d_model / n_head`. Projekcja QKV tworzy query, key i value, dzieli je na głowice i przestawia wymiary, aby każda głowica mogła liczyć własną uwagę.

**Na liczbach Twojego modelu,** dla jednego zdania przykładowego (B=1, T=16):

```text
ids                 [1, 16]          16 liczb całkowitych, np. 87, 256, 266, ...
token embedding     [1, 16, 128]     każdy token → 128 liczb
po 4 blokach        [1, 16, 128]     ten sam kształt, inna treść
logity              [1, 16, 1024]    dla każdej z 16 pozycji: ocena każdego z 1024 tokenów
```

> **Po co kształt się nie zmienia w blokach?** Każdy blok dodaje coś do wektora, zamiast go zastępować (połączenie rezydualne, rozdział 8.4). Wejście i wyjście bloku muszą mieć ten sam kształt, żeby dało się je dodać. Dzięki temu bloki można układać jeden na drugim w dowolnej liczbie: 4 w modelu 1M, 6 w modelu 10M.

### 3.4.4 Blok transformera

`CausalSelfAttention` buduje Q/K/V, liczy wynik uwagi, stosuje maskę przyczynową i projekcję wyjściową. Maska jest trójkątem dolnym: pozycja t widzi wyłącznie pozycje 0..t. Nie zapisuje się do `state_dict`, bo można ją odtworzyć z konfiguracji.

W kodzie (`model.py`, metoda `CausalSelfAttention.forward`) to trzy linie:

```python
scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
masked = scores.masked_fill(~self.mask[start : start + t, :t_all], float("-inf"))
weights = F.softmax(masked, dim=-1)
```

Wpisanie `-inf` w miejsca przyszłych pozycji sprawia, że po softmaksie dostają one wagę dokładnie 0, bo e^(−∞) = 0. Prawdziwe mapy uwagi Twojego modelu są w rozdziale 8.3.

`MLP` ma wymiary `d_model → 4*d_model → d_model`, z GELU pomiędzy projekcjami. `Block` stosuje układ pre-norm i połączenia rezydualne:

```text
x = x + attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

`GPT` dodaje embedding pozycji do embeddingu tokenu, przeprowadza sekwencję przez `n_layer` bloków, normalizuje i produkuje logity. Waga `lm_head` jest tą samą macierzą co embedding tokenów. Weight tying oszczędza parametry i wiąże reprezentację tokenu wejściowego z klasyfikatorem tokenów wyjściowych.

> **Po co weight tying, na liczbach.** Embedding tokenów w modelu 1M to 1024 × 128 = 131 072 parametry. Osobna warstwa wyjściowa miałaby drugie tyle. Współdzielenie oszczędza więc 12% całego modelu. W modelu 10M oszczędność to 2,6 mln parametrów (26%). Przy okazji: kiedyś policzyłem 1 088 256 parametrów modelu 1M, bo zsumowałem `state_dict`, który pokazuje tę samą macierz dwa razy. Prawidłowa liczba to 957 184.

### 3.4.5 `KVCache`

Cache przechowuje key i value dla każdej warstwy i pozycji już przetworzonej w danym generowaniu. Podczas kolejnego kroku dopisuje dane nowego tokenu, zamiast przeliczać wszystkie wcześniejsze pozycje. `reset` czyści stan. Cache nie jest potrzebny w zwykłym treningu i nie jest częścią wag zapisanych w checkpoincie.

> **Bez tego…** Żeby napisać 500 tokenów, model bez cache przelicza najpierw 1 pozycję, potem 2, potem 3… aż do 256 (więcej nie mieści się w kontekście). To ok. 100 tys. przeliczeń pozycji zamiast 500. Wynik jest identyczny, tylko wolniejszy (rozdział 8.9).

## 3.5 `src/nanowriter/train.py` — jak model się uczy

### 3.5.1 Strata

Model zwraca logity `[B,T,V]`. `compute_loss` spłaszcza pozycje do `B*T` i liczy cross-entropy dla id z `y`. Strata jest skalarem. Backpropagacja oblicza pochodne straty względem parametrów.

`compute_loss` przyjmuje też opcjonalną maskę `[B,T]` z zerami i jedynkami. Wtedy liczy się tylko tam, gdzie maska ma 1. Model 1M jej nie używa. Rozmówca 10M używa jej przy douczaniu, żeby uczyć się tylko odpowiedzi, a nie pytań (rozdział 9.4).

> **Twój model (1M), pierwszy krok treningu.** Strata w kroku 0 wyniosła **6,946**. To nie przypadek: model z losowymi wagami daje wszystkim 1024 tokenom mniej więcej po równo, czyli 1/1024 każdemu. Strata to −ln(1/1024) = ln(1024) = **6,931**. Rozmówca 10M zaczął od 9,07, a ln(8192) = 9,01. Jeśli strata na starcie nie jest bliska ln(vocab_size), coś jest źle z inicjalizacją.

### 3.5.2 Optymalizator i harmonogram

`make_optimizer` tworzy AdamW. Parametry macierzowe o co najmniej dwóch wymiarach otrzymują weight decay. Wektory i skalarne parametry go nie otrzymują. `lr_at` oblicza learning rate: może używać rozgrzewki, potem wartości stałej lub harmonogramu cosinusowego z dolną granicą `min_lr`.

> **Po co rozgrzewka (warmup)?** Na samym początku wagi są losowe i gradienty bywają duże i chaotyczne. AdamW nie zebrał jeszcze statystyk, które normalnie łagodzą kroki. Pełny LR od kroku 0 potrafi wtedy „wystrzelić” wagi w złe miejsce, z którego model długo wraca. Twój model 1M zwiększał LR od 0,00001 do 0,002 przez pierwsze 200 kroków.

> **Po co spadek cosinusowy?** Na początku duże kroki szybko schodzą w dół. Pod koniec model jest blisko dobrego miejsca i duże kroki tylko by nad nim przeskakiwały. Twój model 1M kończył na 0,0002, dziesięć razy mniej niż szczyt.

`train_step`:

1. zeruje poprzednie gradienty;
2. wykonuje forward pass (opcjonalnie autocast bfloat16 na CUDA);
3. liczy stratę;
4. wykonuje backward;
5. opcjonalnie przycina normę gradientu;
6. robi krok AdamW.

> **Bez kroku 1 (zerowania)…** PyTorch dodaje nowe gradienty do starych, zamiast je zastępować. Po 10 krokach gradient byłby sumą 10 batchy, coraz większy, i trening by się rozjechał.

### 3.5.3 Trening, ewaluacja i wznowienie

`train` organizuje wiele kroków: losuje batch, ustawia LR, wykonuje `train_step`, wywołuje callbacki i zapisuje JSONL. `estimate_loss` przechodzi deterministycznie po niepokrywających się oknach zbioru i tymczasowo ustawia model w `eval()`. Po zakończeniu przywraca poprzedni tryb.

**Jak model uczy się pisać.** Poniższy wykres to krótki trening od zera tej samej konfiguracji co model 1M (800 kroków, ok. 18 s na GPU). Obok: tekst, który model pisze po „Wieczorem stary kot” na końcu treningu.

![Krótki trening od zera (800 kroków). Przerywana linia to strata losowego zgadywania, ln(1024). Animacja z próbkami tekstu co ok. 40 kroków: `przewodnik/wyniki/uczenie.gif`.](wyniki/uczenie.png)

Próbki z tego samego treningu, zawsze po „Wieczorem stary kot”:

| Krok | Strata | Co model pisze |
| --- | --- | --- |
| 0 | 6,93 | `mi raz raz�lołochoąaryromiałro F in ściąsł nie mnie� To rę rę…` |
| 241 | 4,63 | `i orgichło.` / `— A ja to nie mogła z drzegać się zastąpił się wygładak.` |
| 521 | 4,30 | `i cujudzi.` / `— Wycieka, to paniesz mnie…` / `— Wyjam, wyjaciem zwykam…` |
| 800 | 4,07 | `i cóżku, merestroncieria…` / `— Pan Kajtusiadungda nie zwykrekre.` / `— Dostanusz się zobaczyć.` |

Kolejność, w jakiej model się uczy, jest zawsze podobna: najpierw częste litery i spacje, potem polskie końcówki i myślniki dialogów, potem krótkie prawdziwe słowa („nie”, „się”, „Pan”), a gramatyka całych zdań przychodzi na końcu. Pełny trening modelu 1M (5000 kroków) zszedł do straty 3,00 na train i 3,11 na val.

> **Sprawdź sam.** `uv run python -m przewodnik.uczenie` powtórzy ten trening (ok. 20 s na GPU) i zapisze animację.

`save_checkpoint` zapisuje model, optymalizator, numer kroku, stan generatora batchy, konfigurację treningu i (jeśli model ją ma) konfigurację GPT. Zapisuje do pliku tymczasowego, a potem atomowo podmienia plik docelowy. `load_checkpoint` ładuje dane na CPU. `load_gpt` odtwarza architekturę z `model_config`, wczytuje wagi i przenosi model na wskazane urządzenie.

Wznowienie kontynuuje nie tylko od wag, ale też od stanu optymalizatora i RNG. To ważne: bez stanu AdamW i generatora kolejne aktualizacje/batche mogłyby różnić się od treningu bez przerwy.

> **Bez stanu AdamW…** AdamW pamięta dla każdego parametru średnią ostatnich gradientów i ich wielkości. Po wznowieniu bez tej pamięci pierwsze kroki byłyby jak bez rozgrzewki: gwałtowne. Na wykresie straty zwykle widać wtedy skok w górę zaraz po wznowieniu. Test `test_train_gpt_02_skrypt_treningu.py` sprawdza, że u Ciebie tak nie jest: wznowiony trening daje tę samą stratę co trening bez przerwy.

### 3.5.4 Wartości klasy a wartości CLI

`TrainConfig()` w module ma wartości ogólne, m.in. batch 32 i block 8. `train_gpt.py` tworzy własną konfigurację, domyślnie batch 64, block 256, 5000 kroków, LR 0.002, warmup 200, cosine, ewaluacja co 250 kroków, checkpoint co 500 kroków i log co 100 kroków. Przy czytaniu zachowania programu sprawdź, czy wartość pochodzi z klasy, czy z argumentów konkretnego skryptu.

> **Twój model (1M), ile tekstu przeczytał.** 5000 kroków × 64 fragmenty × 256 tokenów = 81,9 mln tokenów. Zbiór treningowy ma 9,8 mln tokenów, więc model „przeczytał” każdą książkę średnio **8,3 razy**. To dużo jak na tak mały model. Strata val (3,11) jest jednak tylko trochę wyższa od train (3,00), więc model jeszcze nie zaczął się uczyć tekstów na pamięć.

## 3.6 `src/nanowriter/sample.py` — od logitów do nowego tekstu

`generate` przyjmuje początkowe id `idx` o kształcie `[B,T]`. Dla każdego nowego tokenu:

1. ogranicza wejście do kontekstu, jeśli podano `block_size`;
2. oblicza logity ostatniej pozycji;
3. stosuje wybór zachłanny albo filtry próbkowania;
4. dopisuje wybrane id do sekwencji.

Temperatura zero wybiera najwyższy logit. Dodatnia temperatura dzieli logity przed softmaxem. Top-k odrzuca tokeny poza k najwyższymi. Top-p sortuje kandydatów i zachowuje najmniejszy prefiks o zadanej sumie prawdopodobieństwa. W tej implementacji stosowany jest najpierw filtr top-k, a potem top-p.

> **Po co losować, zamiast brać zawsze najlepszy token?** Zachłanny wybór jest przewidywalny do bólu: Twój model po „Wieczorem” na pierwszym miejscu stawia przecinek (14%), więc wybór zachłanny zawsze napisze „Wieczorem,”. Małe modele przy takim wyborze szybko wpadają w pętlę i powtarzają te same frazy. Losowanie daje różnorodność, a temperatura i top-k pilnują, żeby nie losować bzdur z ogona rozkładu (rozdział 8.8, z wykresem).

Przy `use_cache=True` pierwszy krok buduje cache dla promptu. Następne kroki przekazują zwykle tylko nowy token. Jeśli kontekst osiąga `block_size`, cache jest resetowany i odtwarzany z bieżącego okna kontekstu. `max_new_tokens=0` zwraca kopię wejścia.
