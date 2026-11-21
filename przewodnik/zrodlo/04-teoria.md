
# 8. Teoria: jak działa LLM krok po kroku

Ta część łączy powyższe pojęcia. Nie każdy LLM ma dokładnie tę samą architekturę. Opis dotyczy w szczególności małego decoder-only GPT z NanoWriter. Każdy podrozdział kończy się wskazaniem, gdzie dana rzecz siedzi w kodzie.

## 8.1 Cel: przewidywać następny token

Model językowy uczy się rozkładu:

```text
P(token_t | token_0, token_1, ..., token_(t-1))
```

Oznacza to: na podstawie wcześniejszych tokenów oszacuj prawdopodobieństwo każdego tokenu, który mógłby wystąpić następny. Model nie otrzymuje reguł gramatyki ani słownika znaczeń jako osobnych instrukcji. Informacje przydatne do przewidywania kodują się w wagach podczas uczenia na przykładach tekstu.

Jeśli tekst tokenów to `[A, B, C, D]`, przykłady następnego tokenu są przesunięte:

```text
x = [A, B, C]  →  y = [B, C, D]
```

Model dostaje x i produkuje przewidywanie w każdej pozycji. Na pozycji wejściowej A ma przewidzieć B, na pozycji B: C itd. W implementacji wiele pozycji jest liczonych równolegle, ale maska zapewnia przyczynowość.

**Twój model przy pracy.** Tak model 1M ocenia kolejne tokeny zdania „Wieczorem stary kot zasnął przy piecu.”. Kolumna „p” mówi, ile prawdopodobieństwa model dał tokenowi, który naprawdę jest następny w zdaniu. „Miejsce” to pozycja tego tokenu na liście wszystkich 1024 tokenów, posortowanej od najbardziej prawdopodobnego.

| Model widzi (ostatni token) | Naprawdę dalej | p | Miejsce | Model stawiał na |
| --- | --- | --- | --- | --- |
| `W` | `ie` | 0,07 | 2 | `i` (0,12) |
| `ie` | `cz` | 0,06 | 6 | `m` (0,23) |
| `cz` | `or` | **0,76** | **1** | `or` |
| `or` | `em` | **0,84** | **1** | `em` |
| `em` | `·st` | 0,005 | 29 | `,` (0,14) |
| `·st` | `ary` | 0,02 | 11 | `ał` (0,28) |
| `ary` | `·ko` | 0,004 | 56 | `,` (0,12) |
| `·ko` | `t` | 0,015 | 19 | `n` (0,12) |
| `t` | `·za` | 0,004 | 44 | `łow` (0,10) |
| `·za` | `s` | 0,02 | 10 | `t` (0,06) |
| `s` | `nął` | **0,27** | **1** | `nął` |
| `nął` | `·przy` | 0,003 | 44 | `.` (0,28) |
| `·przy` | `·pie` | 0,001 | 173 | `k` (0,08) |
| `·pie` | `cu` | **0,42** | **1** | `cu` |
| `cu` | `.` | **0,40** | **1** | `.` |

Z tej tabeli widać, czego nauczył się model z milionem parametrów:

- **Dokańczanie słów idzie mu świetnie.** Po „Wiecz” wie, że będzie „or”, a potem „em” (76% i 84%). Po „zas” trafia „nął”, po „pie” trafia „cu”.
- **Wybór następnego słowa idzie mu słabo.** Po „Wieczorem” nie zgadnie, że będzie „stary” (miejsce 29), a po „zasnął” nie zgadnie „przy” (miejsce 44). To jest trudne, bo zależy od sensu, a nie od pisowni.
- **Jego pomyłki są rozsądne.** Po „zasnął” stawia kropkę (28%). „Wieczorem stary kot zasnął.” to przecież poprawne zdanie.

![Rozkład następnego tokenu po „Wieczorem stary kot” (10 najbardziej prawdopodobnych). Prawdziwy następny token „ za” jest dopiero na 44. miejscu. Animacja dla każdej pozycji zdania: `przewodnik/wyniki/nastepny_token.gif`.](wyniki/nastepny_token.png)

> **Strata na tym jednym zdaniu.** Średnia z −ln(p) dla 15 pozycji z tabeli wynosi ok. 3,3. Strata walidacyjna całego modelu to 3,11. Strata to więc dosłownie średnia „zaskoczenia” modelu prawdziwym tekstem. Jedno zdanie daje wynik bliski średniej z całego zbioru walidacyjnego.

> **W kodzie:** przesunięcie x/y robi `get_batch` w `dataset.py`, przewidywanie `GPT.forward` w `model.py`.

## 8.2 Tekst → tokeny → wektory

Najpierw tokenizer mapuje tekst na id. Id są indeksami kategorii, dlatego embedding zamienia każde id na wektor liczb uczony razem z modelem. Pozycja tokenu też ma znaczenie, więc dodawany jest embedding pozycyjny:

```text
reprezentacja = embedding(token) + embedding(pozycji)
```

Wymiary `d_model` tworzą przestrzeń, w której warstwy sieci przetwarzają informację. Przy `d_model=128` każda pozycja tekstu jest opisana przez 128 liczb.

> **Po co embedding, skoro już mamy id?** Id to tylko numer w słowniku. Token 509 („nął”) nie jest „większy” od tokenu 305 („em”), a 305 nie jest „bliżej” 306 niż 900. Gdyby sieć dostała same numery, wyciągałaby z nich fałszywe zależności. Embedding daje każdemu tokenowi 128 liczb, które trening ustawia tak, żeby tokeny o podobnym użyciu miały podobne wektory.

> **Bez embeddingu pozycji…** Uwaga (8.3) sama w sobie nie wie, który token był pierwszy. Dla niej „kot zjadł mysz” i „mysz zjadła kot” to ten sam zbiór tokenów. Embedding pozycji dokleja do każdego tokenu informację „jestem na miejscu 3”.

> **W kodzie:** `GPT.forward`, pierwsze linie: `tok_emb(idx) + pos_emb(pozycje)`.

## 8.3 Uwaga: ważenie kontekstu

Każda głowica przekształca reprezentacje wejściowe w Q (query), K (key), V (value). Dla danej pozycji Q porównywane jest z K innych pozycji. Wynik skalowany jest przez pierwiastek z wymiaru głowicy, maskowany i normalizowany softmaxem:

```text
scores  = Q · Kᵀ / √d_head
weights = softmax(mask(scores))
output  = weights · V
```

Mnożenie Q · Kᵀ daje macierz podobieństw pozycji. Softmax zamienia każdy wiersz na wagi sumujące się do 1. Wynik jest więc mieszanką wartości V: pozycja może zebrać informacje z wcześniejszych tokenów proporcjonalnie do wag uwagi.

> **Intuicja Q, K, V.** Każdy token wystawia trzy wektory. **Q** to pytanie „czego szukam?”, **K** to etykieta „co mam do zaoferowania”, **V** to sama treść, którą przekaże, jeśli ktoś go wybierze. Iloczyn Q·K mówi, jak bardzo etykieta pasuje do pytania. Softmax robi z tych dopasowań proporcje, a wynik to mieszanka treści V w tych proporcjach.

> **Po co dzielić przez √d_head?** Iloczyn skalarny dwóch wektorów o 32 losowych składowych ma rozrzut ok. √32 ≈ 5,7 razy większy niż pojedyncza składowa. Przy dużych liczbach softmax daje prawie całą wagę jednemu tokenowi, a gradient dla reszty jest bliski zera i model przestaje się uczyć. Dzielenie przez √d_head sprowadza wyniki z powrotem do rozsądnej skali.

**Prawdziwe mapy uwagi Twojego modelu.** Każdy z 16 kwadratów to jedna głowica (W = warstwa, G = głowica). Wiersz to token, który pyta, kolumna to token, na który patrzy. Im ciemniej, tym większa waga. Wszystko nad przekątną jest białe: to maska przyczynowa.

![Mapy uwagi modelu 1M dla zdania przykładowego: 4 warstwy × 4 głowice. Animacja krok po kroku (wyniki Q·K, maska, softmax) dla jednej głowicy: `przewodnik/wyniki/uwaga_krok_po_kroku.gif`.](wyniki/uwaga_mapy.png)

Co widać na mapach:

- **Głowice się wyspecjalizowały, choć nikt im tego nie kazał.** W1 G0 (pomarańczowa ramka) ma wyraźny wzór: „ary” patrzy na „·st”, „t” na „·ko”, „s” i „nął” na „·za”. Każdy token patrzy na **początek słowa, w którym stoi**. To bardzo przydatne przy dokańczaniu słów, a z tabeli w 8.1 wiemy, że to model umie najlepiej.
- **Część głowic patrzy szeroko** (np. W0 G2): rozmyty, równy kolor oznacza „uśrednij wszystko, co było”. Daje to ogólny kontekst: o czym jest tekst, czy to dialog.
- **Pierwszy token „W” często dostaje dużo uwagi.** Małe modele używają pierwszej pozycji jako „miejsca zrzutu”, gdy głowica w danym wierszu nie ma nic lepszego do roboty.

### 8.3.1 Dlaczego maska przyczynowa?

W treningowym batchu cały fragment tekstu jest dostępny jednocześnie. Bez maski pozycja mogłaby odczytać token, który powinna dopiero przewidzieć. Dolnotrójkątna maska pozwala pozycji t widzieć 0..t, ale blokuje t+1..T−1. Dzięki temu równoległe uczenie jest zgodne z przyszłym autoregresywnym generowaniem.

> **Bez maski…** Strata treningowa spada wtedy prawie do zera, bo model zamiast przewidywać, po prostu przepisuje następny token, który widzi. Przy generowaniu przyszłości jednak nie ma i model, który nigdy nie nauczył się przewidywać, pisze bełkot. Wynik treningu wygląda świetnie, a model jest bezużyteczny.

### 8.3.2 Dlaczego kilka głowic?

Jedna głowica liczy uwagę w jednej podprzestrzeni. Kilka głowic może równolegle wyuczyć różne wzorce relacji: jedna „początek słowa”, inna „poprzedni token”, jeszcze inna „ogólny kontekst”. Wyniki są łączone i rzutowane z powrotem do `d_model`. W projekcie `head_dim = d_model / n_head`, czyli 128 / 4 = 32 w modelu 1M i 320 / 5 = 64 w modelu 10M.

> **W kodzie:** `CausalSelfAttention.forward` w `model.py`. Opcjonalny słownik `trace` zapisuje `scores`, `masked` i `weights`, z niego powstały obrazki powyżej.

## 8.4 Blok: uwaga, MLP i połączenia rezydualne

Po attention każde położenie przechodzi także przez MLP `d_model → 4*d_model → d_model`. Attention miesza informacje między pozycjami, a MLP przekształca cechy każdej pozycji osobno. GELU dodaje nieliniowość.

> **Podział pracy w jednym zdaniu:** uwaga zbiera informacje z innych tokenów („co było wcześniej?”), a MLP przetwarza to, co zebrała („co z tego wynika?”). W Twoim modelu 1M MLP ma 55% wszystkich parametrów, uwaga 28%.

> **Bez GELU (nieliniowości)…** Dwie warstwy liniowe jedna po drugiej to matematycznie nadal jedna warstwa liniowa. Cały głęboki model zwinąłby się do jednego mnożenia macierzy i nie nauczyłby się niczego ponad proste statystyki.

Blok NanoWriter jest pre-norm:

```text
x₁ = x + Attention(LayerNorm(x))
x₂ = x₁ + MLP(LayerNorm(x₁))
```

Dodanie x to połączenie rezydualne. Ułatwia przepływ informacji i gradientów przez głębszą sieć. LayerNorm stabilizuje skalę cech przed podwarstwami. Powtarzanie bloków umożliwia stopniowe przekształcanie informacji kontekstowej.

> **Intuicja połączenia rezydualnego.** Wyobraź sobie wspólny notatnik dla każdego tokenu, przekazywany z bloku do bloku. Każdy blok czyta notatnik i **dopisuje** swoje uwagi, ale niczego nie wymazuje. Bez tego każdy blok musiałby przepisać cały notatnik od nowa, a błąd jednego bloku niszczyłby wszystko, co zrobiły poprzednie. Gradient też płynie wtedy „autostradą” przez dodawanie, zamiast przeciskać się przez każdą warstwę.

> **W kodzie:** `Block.forward` w `model.py`.

## 8.5 Logity, softmax i strata

Końcowa projekcja daje dla każdej pozycji V logitów, po jednym na token słownika. Logit nie jest jeszcze prawdopodobieństwem. Softmax:

```text
p_i = exp(logit_i) / Σ_j exp(logit_j)
```

zamienia je na nieujemne wartości sumujące się do 1. Cross-entropy ocenia, ile prawdopodobieństwa przypisano prawidłowemu tokenowi. Dla poprawnego id y strata pojedynczej pozycji to:

```text
loss = -log(p_y)
```

Jeżeli model przypisze poprawnemu tokenowi prawdopodobieństwo bliskie 1, strata będzie mała. Jeżeli przypisze mu prawdopodobieństwo bliskie 0, kara będzie duża. Straty z pozycji batcha są uśredniane.

**Jak czytać wartości straty, na przykładach z tabeli w 8.1:**

| p prawdziwego tokenu | Strata −ln(p) | Przykład |
| --- | --- | --- |
| 0,84 | 0,18 | „or” → „em”: model był pewny i miał rację |
| 0,27 | 1,31 | „s” → „nął”: trafił, ale się wahał |
| 1/1024 | 6,93 | losowe zgadywanie, start treningu |
| 0,001 | 6,91 | „·przy” → „·pie”: mocno zaskoczony |

> **Po co logarytm, a nie po prostu 1 − p?** Logarytm karze pewne pomyłki bardzo mocno. Przy p = 0,001 strata to 6,9, przy p = 0,01 już tylko 4,6. Model, który daje prawdziwemu tokenowi „prawie zero”, dostaje dużo silniejszy sygnał do poprawy niż model, który się tylko lekko pomylił.

> **W kodzie:** `compute_loss` w `train.py` (wywołuje `F.cross_entropy`, które liczy softmax i logarytm naraz).

## 8.6 Gradienty i aktualizacja wag

Strata zależy od wielu macierzy parametrów. Backpropagation stosuje regułę łańcuchową, żeby policzyć pochodną straty względem każdego parametru. Gradient nie mówi bezpośrednio „jaki tekst jest dobry”. Mówi, jak mała zmiana parametru wpłynęłaby na obecną stratę.

> **Intuicja.** Stoisz we mgle na zboczu i chcesz zejść do doliny. Nie widzisz doliny, ale czujesz pod stopami, w którą stronę jest stromo w dół. Gradient to to „czucie” dla każdego z 957 184 parametrów naraz. Learning rate to długość kroku.

AdamW używa gradientów i własnych ruchomych statystyk, aby aktualizować parametry. Learning rate skaluje te aktualizacje. Za duży LR może destabilizować uczenie, a zbyt mały może wymagać znacznie większej liczby kroków. Warmup łagodnie zwiększa LR na początku, a cosine schedule potem go obniża.

Training loop nie zapisuje na stałe gradientów batchy: po każdym kroku je zeruje, wylicza od nowa i wykonuje jedną aktualizację. Checkpoint zapisuje stan optymalizatora, bo same wagi nie wystarczą do dokładnego kontynuowania AdamW.

> **W kodzie:** `train_step` w `train.py`: `loss.backward()` liczy gradienty, `optimizer.step()` robi krok.

## 8.7 Walidacja i uogólnianie

Train służy do uczenia, val do pomiaru. W walidacji nie wykonuje się backward ani kroku optymalizatora. Jeżeli strata train spada, a val rośnie, model może nadmiernie dopasowywać się do treningu (overfitting).

![Prawdziwy przebieg treningu modelu 1M z `data/checkpoints/train_log.jsonl`. Górny wykres: strata, dolny: learning rate (rozgrzewka przez 200 kroków, potem spadek cosinusowy).](wyniki/krzywa_straty.png)

Jak czytać ten wykres:

- **Krok 0: 6,95, czyli prawie dokładnie ln(1024)** (przerywana linia). Model zgaduje losowo.
- **Pierwsze 200 kroków: gwałtowny spadek do ok. 4,5.** Model uczy się najprostszych rzeczy: częstych liter, spacji, końcówek. Każda z nich mocno zmniejsza stratę.
- **Od kroku ok. 1000 spadek zwalnia.** Łatwe rzeczy model już umie, zostały trudne: wybór słów, gramatyka, sens.
- **Linie train i val idą razem do końca** (3,00 i 3,11). To znaczy, że model jeszcze się nie przeucza. Gdyby val zaczęła rosnąć przy spadającej train, trzeba by przerwać trening wcześniej.

W NanoWriter `estimate_loss` korzysta z deterministycznych okien, a skrypt GPT może podawać końcową stratę val i bits per character. Bity na znak w kodzie przeliczane są jako:

```text
bits_per_char = loss × token_count / (char_count × ln(2))
```

To normalizuje stratę tokenową ze względu na inny poziom segmentacji tekstu, ale porównywane modele nadal powinny mieć porównywalny korpus i sposób ewaluacji.

> **Twój model (1M), na liczbach:** 3,112 × 1 295 323 / (3 025 008 × 0,693) = **1,92 bitu na znak**. Inaczej: dla każdej litery model waha się średnio tak, jakby wybierał spośród 2^1,92 ≈ 3,8 równie prawdopodobnych możliwości.

> **Po co bity na znak, skoro jest strata?** Strata zależy od tokenizera. Model 10M ma tokeny średnio 3,24 znaku, a model 1M 2,32 znaku, więc jeden token 10M „niesie” więcej tekstu i strata na token jest naturalnie wyższa. Bity na znak przeliczają to na wspólną jednostkę: literę.

## 8.8 Generowanie jako pętla autoregresyjna

Po treningu model zwraca rozkład następnego tokenu dla promptu. Generator wybiera token, dopisuje go do sekwencji i ponownie pyta model. Ten krok jest powtarzany aż do liczby tokenów z `--length`.

- **Greedy / zachłanne:** wybierz najwyższy logit. W kodzie `temperature=0`.
- **Losowanie:** wylosuj id zgodnie z prawdopodobieństwami softmax.
- **Temperatura:** przy dzieleniu logitów przez T, T<1 zwykle wyostrza rozkład, a T>1 go spłaszcza.
- **Top-k:** po sortowaniu pozostaw tylko k najlepszych tokenów.
- **Top-p:** pozostaw najmniejszy zbiór o skumulowanym prawdopodobieństwie co najmniej p.

**Ten sam rozkład, pięć ustawień.** Poniżej model 1M po „Wieczorem stary kot zasnął przy”. Rozkład z modelu jest jeden, zmienia się tylko to, jak z niego losujemy:

![Rozkład następnego tokenu modelu 1M po pięciu różnych ustawieniach próbkowania. Animacja płynnej zmiany temperatury od 0,3 do 2,0: `przewodnik/wyniki/probkowanie_temperatura.gif`.](wyniki/probkowanie.png)

| Ustawienie | Najlepszy token „k” | Ile tokenów można wylosować | Co to daje w praktyce |
| --- | --- | --- | --- |
| temperatura 0,5 | 31% | wszystkie 1024 (ale ogon prawie zerowy) | tekst bezpieczny, powtarzalny |
| temperatura 1,0 | 8,5% | wszystkie 1024 | dokładnie to, czego model się nauczył |
| temperatura 1,5 | 3,5% | wszystkie 1024, ogon mocno podniesiony | dużo dziwnych słów i literówek |
| top-k 5 | 31% | tylko 5 | nigdy nie wylosuje bzdury z ogona |
| top-p 0,9 | 9,4% | 145 | odcina ogon, ale zostawia różnorodność |

> **Po co top-k/top-p, skoro ogon ma tak małe p?** Pojedynczy token z ogona ma np. 0,0005, ale ogon to ok. 900 takich tokenów. Razem dają kilka–kilkanaście procent. Przy 500 generowanych tokenach model trafi w ogon kilkadziesiąt razy, a każde takie trafienie to zwykle literówka albo urwane słowo, od którego tekst się „rozjeżdża”.

Seed kontroluje losowanie, o ile sprzęt, kod i pozostałe ustawienia są takie same. Zmiana seeda może zmienić tekst mimo identycznego checkpointu.

> **W kodzie:** `generate` w `sample.py`. Domyślne ustawienia `scripts/generate.py`: temperatura 0,8, top-k 50, top-p 0,95.

## 8.9 KV cache: mniej ponownych obliczeń

W autoregresji poprzedni kontekst jest znany i nie zmienia się. Klucze i wartości uwagi obliczone dla wcześniejszych tokenów również się nie zmieniają. Cache zachowuje je między krokami:

```text
bez cache: w każdym kroku przelicz całą widoczną sekwencję
z cache:   oblicz prompt raz, później dopisuj K/V nowego tokenu
```

Cache zmniejsza powtarzane obliczenia, ale zużywa pamięć. Nie zmienia definicji modelu ani oczekiwanego rozkładu następnego tokenu. Testy porównują wyniki cache i trybu bez cache, aby sprawdzić tę zgodność.

> **Dlaczego wolno trzymać K i V, ale nie trzeba trzymać Q?** Dzięki masce przyczynowej stare tokeny nigdy nie patrzą na nowe. Ich K i V nie zmienią się więc, choćby dopisać tysiąc tokenów. Q potrzebne jest tylko dla nowego tokenu, bo tylko on teraz „pyta”. Bez maski przyczynowej KV cache byłby niemożliwy.

> **Sprawdź sam.** `uv run python scripts/generate.py --prompt "Wieczorem" --seed 1` i to samo z `--no-cache`. Tekst będzie identyczny, różni się tylko czas.

## 8.10 Rozmiar modelu i ograniczenia

W przybliżeniu parametry bloków GPT rosną jak `12 × n_layer × d_model²`. Dochodzi do tego embedding słownika `vocab_size × d_model`, embedding pozycji i parametry normalizacji/biasów. Wagi są tylko częścią pamięci treningu. Aktywacje zależą także od batcha, kontekstu i liczby warstw.

> **Sprawdzenie wzoru na Twoim modelu 1M:** 12 × 4 × 128² = 786 432 w blokach, plus 1024 × 128 = 131 072 embeddingu tokenów, plus 256 × 128 = 32 768 embeddingu pozycji. Razem 950 272. Brakujące ok. 7 tys. do 957 184 to biasy i LayerNormy. Wzór trafia z dokładnością do 1%.

Zwiększanie `d_model`, liczby warstw albo kontekstu ma koszt. Większy model nie staje się automatycznie lepszy: potrzebuje danych i obliczeń, a przy małym korpusie może się przeuczyć. Również spadek straty nie daje gwarancji prawdziwych faktów ani spójnego długiego tekstu.

Najważniejszy wniosek: mały GPT nie przechowuje osobnej, jawnej reguły dla każdego zdania. Uczy się wag, które pozwalają przekształcać tokeny i kontekst w rozkład następnego tokenu. Trening dopasowuje te rozkłady do przykładów, a generowanie zamienia kolejne rozkłady w nową sekwencję.
