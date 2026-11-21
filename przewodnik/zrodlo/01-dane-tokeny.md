
# 3. Moduły biblioteki — co robią i dlaczego

## 3.1 `src/nanowriter/data.py` — skąd bierze się tekst

Moduł pobiera teksty z API Wolnych Lektur, selekcjonuje książki, czyści ich treść i tworzy train/val. Nie wie nic o tokenach ani o modelu.

> **Po co?** Model nauczy się dokładnie tego, co dostanie. Gdyby 30% danych stanowiły stopki „Ta lektura, podobnie jak tysiące innych, jest dostępna on-line…”, model wstawiałby je w środek opowiadania. Czyszczenie to nie kosmetyka, tylko decyzja, czego model się nauczy.

### 3.1.1 Katalog i pobieranie

- `PROSE_EPOCHS` i `PROSE_GENRES` określają domyślne epoki i gatunki.
- `catalog_api_url` buduje bezpiecznie ograniczony adres katalogu.
- `list_books` pobiera wpisy katalogowe i normalizuje niekompletne metadane.
- `select_books` wybiera pasujące gatunki, usuwa powtórzone slugi i tasuje deterministycznie przy ustalonym seedzie.
- `book_api_url` sprawdza slug, zanim zostanie użyty w adresie.
- `download_book` pobiera opis książki, odnajduje adres tekstu i zapisuje UTF-8. Najpierw tworzy plik `.part`, a po udanym zapisie atomowo zmienia jego nazwę. Dzięki temu przerwany zapis nie wygląda jak gotowa książka.
- `LimitedFetch` zlicza pobrane bajty, blokuje kolejne żądanie po limicie i wprowadza pauzę między żądaniami.

Limity i pauza mają znaczenie, bo jeden korpus może wymagać wielu żądań HTTP. Skrypt nie powinien bez ograniczeń pobierać katalogu ani plików książek.

> **Bez tego (plik `.part`)…** Gdy prąd zgaśnie w połowie pobierania, na dysku zostaje pół książki pod normalną nazwą. Przy następnym uruchomieniu skrypt uzna ją za gotową i pół książki trafi do treningu. Model nauczy się, że teksty urywają się w połowie zdania.

> **Po co seed w `select_books`?** Żeby dwa uruchomienia dały te same książki w tej samej kolejności. Bez tego nie da się porównać dwóch treningów: nie wiesz, czy model jest lepszy, bo zmieniłeś parametr, czy dlatego, że wylosował inne książki.

### 3.1.2 Czyszczenie i wybór treści

- `load_text` odczytuje UTF-8.
- `is_translation` wykrywa informację o tłumaczu w nagłówku, aby przekłady można było domyślnie odrzucać.
- `clean_text` normalizuje końce linii i Unicode, usuwa część elementów formatowania, przypisy oraz stopkę Wolnych Lektur.
- `build_corpus` przechodzi po wybranych książkach, zapisuje teksty oczyszczone i zwraca `CorpusReport`.

Raport nie tylko podaje rozmiar: wylicza przyjęte książki, powody pominięcia oraz informuje, czy osiągnięto limit pobierania. Dzięki temu problemy z korpusem nie są ukrywane za samą liczbą końcową.

> **Po co normalizacja Unicode?** Litera „ą” może być zapisana jako jeden znak (U+0105) albo jako „a” plus ogonek (U+0061 U+0328). Na ekranie wyglądają tak samo, ale dla tokenizera to różne bajty. Bez normalizacji model musiałby uczyć się dwóch wersji każdego polskiego słowa z ogonkiem.

> **Po co odrzucać przekłady?** Cel projektu to płynny tekst w jednym stylu. Przekład z francuskiego ma inną składnię niż polska proza i miesza styl.

### 3.1.3 Podział train/val

`split_books` tasuje listę książek z ustalonym seedem i wybiera książki walidacyjne do zadanego udziału bajtów. `split_corpus` zapisuje ich treści i `split.json`. Podział całymi książkami chroni przed prostym przeciekiem: model nie dostaje w treningu jednej części tej samej książki, a w walidacji jej sąsiedniego fragmentu.

Rozmiar walidacji jest przybliżony, bo najmniejszą jednostką podziału jest cała książka. Dwie kolejne książki mogą przekroczyć zadany procent, ale żadna książka nie zostanie rozcięta między zbiory.

> **Po co w ogóle walidacja?** Strata treningowa mówi, jak dobrze model zna teksty, które widział. Pytanie brzmi jednak: czy umie pisać, czy tylko pamięta? Na to odpowiada strata na tekście, którego nigdy nie widział, czyli walidacyjna.

> **Bez podziału całymi książkami…** Gdy rozdział 3 „Lalki” trafi do treningu, a rozdział 4 do walidacji, model zna już imiona postaci i styl, więc strata walidacyjna wychodzi za dobra. Wyglądałoby to na sukces, a byłoby tylko przeciekiem.

> **Twój model (1M).** 22 754 742 znaki w train i 3 025 008 w val, czyli ok. 12% walidacji.

## 3.2 `src/nanowriter/tokenizer.py` — zamiana tekstu na identyfikatory

Sieć oczekuje liczb całkowitych. Tokenizer definiuje słownik i mapuje tekst do identyfikatorów (`encode`) oraz identyfikatory z powrotem do tekstu (`decode`).

### 3.2.1 `CharTokenizer`

Tokenizer znakowy tworzy listę znaków i mapę znak → id. Z tekstu treningowego może powstać na przykład:

```text
znaki: [" ", "A", "a", "k", "l", "m", "o", "t"]
id:      0    1    2    3    4    5    6    7
tekst: "Ala ma kota"
id:    [1, 4, 2, 0, 5, 2, 0, 3, 6, 7, 2]
```

To proste i czytelne, ale każdy znak to osobna jednostka. Długie teksty mają więcej kroków, a rzadkie znaki muszą mieć własne id. Ten tokenizer nadaje się do modelu bigramowego i małych demonstracji.

> **Bez lepszego tokenizera…** Kontekst modelu to 256 pozycji. Przy tokenach znakowych to 256 liter, czyli ok. 2–3 zdania. Model nie zobaczyłby nawet początku akapitu, który pisze.

### 3.2.2 `BPETokenizer`

BPE oznacza Byte Pair Encoding. W implementacji NanoWriter bazowe id 0..255 reprezentują bajty. Każde scalanie dodaje jeden nowy token o id od 256. Na przykład scalenie pary (97, 97) oznacza utworzenie tokenu z dwóch bajtów `a` i `a`.

Proces uczenia:

1. tekst jest dzielony wzorcem `SPLIT_PATTERN` na kawałki;
2. kawałki kodowane są do bajtów UTF-8;
3. liczone są częstości sąsiednich par bajtów/tokenów;
4. najczęstsza para zostaje scalona;
5. liczenie jest aktualizowane, a proces trwa do limitu słownika albo braku par.

Podział tekstu ogranicza się do kawałków, więc scalenie nie przechodzi przez granice ustalone przez wzorzec. Remisy rozstrzygane są deterministycznie. W `encode` kawałek jest kodowany scaleniami w kolejności ich rang. Wynik może być cache'owany, bo w jednym tekście powtarza się wiele identycznych kawałków.

> **Po co BPE?** To kompromis między literami a słowami. Słownik całych słów byłby ogromny (polski ma miliony form: kot, kota, kotu, kotem…) i nie poradziłby sobie z nowym słowem. Litery są za drobne (patrz wyżej). BPE sam znajduje częste kawałki: końcówki „-em”, „-nął”, przedrostki „za-”, całe krótkie słowa jak „ przy”.

**Jak to wygląda na Twoim tokenizerze.** Zdanie „Wieczorem stary kot zasnął przy piecu.” ma 38 znaków, ale 40 bajtów, bo „ą” i „ł” zajmują w UTF-8 po dwa bajty. Tokenizer 1M (768 scaleń) robi z nich 16 tokenów w 22 krokach scalania:

![Ostatni krok scalania BPE na zdaniu przykładowym (tokenizer 1M). Kropka · oznacza spację na początku tokenu, mała liczba to id tokenu. Animacja wszystkich kroków: `przewodnik/wyniki/bpe.gif`.](wyniki/bpe.png)

Zwróć uwagę na trzy rzeczy:

- **Spacja należy do następnego słowa:** „·st”, „·ko”, „·przy”. Dzięki temu „ kot” na początku słowa to inny token niż „kot” w środku słowa „szkot”.
- **Częste końcówki stały się tokenami:** „em” (305), „nął” (509), „cu” (648). Model widział je tysiące razy.
- **Rzadkie słowa się rozpadają:** „Wieczorem” to 5 tokenów (W|ie|cz|or|em), bo słownik ma tylko 1024 miejsca.

Bajty pozwalają reprezentować dowolny poprawny tekst Unicode po zakodowaniu UTF-8. Dekodowanie niekompletnej sekwencji bajtów zwraca znak zastępczy, zamiast wywracać cały proces. `save`/`load` zapisują typ, wzorzec i listę scaleń do JSON.

> **Po co zaczynać od bajtów, a nie od liter?** Każdy tekst na świecie da się zapisać 256 bajtami, więc tokenizer nigdy nie spotka „nieznanego znaku”. Emoji, chińskie znaki, literówka: wszystko rozpadnie się najwyżej na pojedyncze bajty, ale się zakoduje.

> **Bez `SPLIT_PATTERN`…** BPE mógłby scalić koniec jednego słowa z początkiem następnego, np. „m s” w „Wieczorem stary”, bo ta para jest częsta. Tokeny przestałyby odpowiadać kawałkom słów i model musiałby uczyć się dziwnych zlepków.

`sample_lines` wybiera podzbiór linii do szybszego treningu słownika, kiedy cały tekst przekracza zadany limit znaków. W skrypcie jest domyślnie używana próbka do 8 milionów znaków.

> **Sprawdź sam.** `uv run python -m przewodnik.bpe` zapisze animację scalania krok po kroku do `przewodnik/wyniki/bpe.gif`.

## 3.3 `src/nanowriter/dataset.py` — ciągi tokenów, batche i pliki `.bin`

Tokeny można trzymać jako tensor, ale pełny tekst w Pythonie byłby cięższy i wolniejszy do wielokrotnego ładowania. Ten moduł zapisuje id w zwartym formacie.

### 3.3.1 Batch `x` i przesunięty cel `y`

`get_batch` losuje starty fragmentów o długości `block_size`. Dla ciągu:

```text
tokeny:  10 11 12 13 14 15 16
fragment zaczynający się od 11 daje:
x:       11 12 13
y:       12 13 14
```

Wartość `y[t]` jest poprawną odpowiedzią po wejściu `x[t]`. Wielkość batcha oznacza liczbę niezależnych fragmentów w jednym kroku. Kształty:

```text
x: [batch_size, block_size]
y: [batch_size, block_size]
```

> **To najważniejsza sztuczka w całym projekcie.** Jeden fragment o długości 256 daje nie jeden, ale 256 przykładów uczących naraz: „po `W` ma być `ie`”, „po `W ie` ma być `cz`”, „po `W ie cz` ma być `or`”… Dlatego nikt nie musi ręcznie oznaczać danych: odpowiedzią jest po prostu następny token tekstu.

Na zdaniu przykładowym (tokenizer 1M):

```text
x: W   ie  cz  or  em  ·st ary ·ko t   ·za s   nął ·przy ·pie cu
y: ie  cz  or  em  ·st ary ·ko t   ·za s   nął ·przy ·pie cu  .
```

Generator losowy działa na CPU, a gotowe batche są przenoszone na wskazane urządzenie. Dzięki jawnie przekazanemu generatorowi trening da się odtwarzać.

> **Po co losowe fragmenty, a nie po kolei?** Gdyby model czytał książki po kolei, przez 500 kroków uczyłby się tylko jednej książki, a potem zapominał ją przy następnej. Losowanie miesza style i tematy w każdym batchu, więc każdy krok poprawia model „średnio dla wszystkich tekstów”.

### 3.3.2 Binarny format tokenów

- `write_tokens` zapisuje little-endian `uint16` bez nagłówka;
- `load_tokens` odrzuca plik o nieparzystej liczbie bajtów i daje tensor `int64` na CPU;
- zakres id to 0..65535, więc największy dozwolony słownik ma 65 536 tokenów;
- `encode_file` używa tokenizera do całego pliku i odrzuca słownik, którego id nie zmieściłyby się w `uint16`.

Format `.bin` jest kompaktowy. Brak nagłówka oznacza jednak, że do interpretacji plików potrzebne są metadane i dokładnie ten sam tokenizer.

> **Po co `uint16`, a nie zwykłe liczby Pythona?** Każdy token zajmuje 2 bajty. Twoje 9,8 mln tokenów treningowych to ok. 19,6 MB na dysku. Lista Pythona z tymi samymi liczbami zajęłaby w pamięci kilkanaście razy więcej i ładowałaby się sekundami zamiast milisekund. Przy rozmówcy (177 mln tokenów) to 355 MB kontra kilka GB.

`TokenMeta` zapisuje `vocab_size`, `dtype` oraz liczbę tokenów i znaków w obu zbiorach. `meta.json` służy skryptowi treningowemu do ustawienia rozmiaru słownika modelu i do obliczenia bitów na znak.

> **Twój model (1M), `data/tokens/meta.json`:** `vocab_size` 1024, train 9 813 598 tokenów z 22 754 742 znaków, czyli średnio **2,32 znaku na token**.
