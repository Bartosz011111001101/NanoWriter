# NanoWriter-rozmówca: jak odtworzyć od zera

Drugi model NanoWritera (~10M parametrów), z którym da się porozmawiać po polsku zwykłym, spokojnym tonem. Ten plik zapisuje wszystkie kroki z pierwszego przebiegu (19–20 listopada 2026), z komendami i liczbami, żeby dało się go powtórzyć.

Wszystkie komendy uruchamiasz z katalogu głównego repo w Git Bash. `PYTHONUTF8=1` jest potrzebne na Windowsie, bez niego polskie znaki w wyjściu się sypią.

## Przegląd

| Krok | Co powstaje | Czas (RTX 5060 Ti) |
| --- | --- | --- |
| 1. Rozmowy z dużego modelu | `data/dialogi/surowe/*.txt` | ok. 1 h pracy agentów |
| 2. Przygotowanie rozmów | `data/rozmowy/{train,val}.txt` | sekundy |
| 3. Napisy i książki | `data/pretrain/{train,val}.txt` | ok. 5 min (pobranie 300 MB) |
| 4. Tokenizer i tokeny | `data/rozmowca/tokeny-*` | ok. 3 min |
| 5. Trening wstępny | `data/rozmowca/ckpt-wstepny/gpt.pt` | ok. 30–35 min GPU |
| 6. Douczanie na rozmowach | `data/rozmowca/ckpt-rozmowy/gpt.pt` | kilka minut |
| 7. Rozmowa | — | — |

Stary NanoWriter (`data/tokens/`, `data/checkpoints/`) zostaje nietknięty. Rozmówca trzyma wszystko w `data/rozmowca/`, `data/pretrain/`, `data/rozmowy/`, `data/opus/` i `data/dialogi/`. Cały `data/` jest poza gitem.

## 1. Rozmowy z dużego modelu

Rozmowy do douczania napisał Claude Haiku (rozumowanie `low`) jako agenci w sesji Claude Code, a nie przez API. Zużyło to ok. 4,8 mln tokenów Haiku z limitu subskrypcji.

Każdy agent dostawał polecenie:

> Przeczytaj `data/dialogi/zasady.md` i stosuj się do niego dokładnie. Napisz 50 rozmów, po 10 na temat, i zapisz je do `data/dialogi/surowe/NNN.txt`. Tematy: …

Dla długich rozmów (partie 041–057) było to 25 rozmów, po 5 na temat, z dopiskiem: „zamiast rozkładu długości z zasad KAŻDA rozmowa ma 6–8 wymian (12–16 linii); przed zakończeniem sprawdź, że żadna nie ma mniej niż 12 linii”. Haiku na `low` pisze same krótkie rozmowy, jeśli nie wymusisz długości wprost.

Wynik: 57 plików, 2425 rozmów (466 długich), ok. 162 tys. słów, 0 duplikatów.

### Treść `data/dialogi/zasady.md`

```markdown
# Zasady pisania rozmów

Rozmowy posłużą do douczenia małego modelu językowego, który ma rozmawiać zwykłym, spokojnym tonem.

## Role
- A zaczyna rozmowę: pisze coś od siebie, pyta, opowiada, narzeka.
- B odpowiada jak zwykły, spokojny, życzliwy człowiek. Krótko, 1–3 zdania. Czasem dopyta, czasem coś doda od siebie.
- Rozmawiają różne pary: znajomi, rodzeństwo, sąsiedzi, para, koledzy z pracy, rodzic z dorosłym dzieckiem. Mieszaj je.

## Zasady
- Codzienna polszczyzna z poprawnymi polskimi znakami. Ton zwykły i stonowany. Bez przekleństw, krzyku i dramatu.
- B NIE jest asystentem: żadnego „Oczywiście! Oto…”, żadnych list, porad w punktach ani wykładów.
- Bez markdownu, emoji, nazwisk prawdziwych osób, marek i dat.
- Każda rozmowa ma od 2 do 8 wymian (od 4 do 16 linii), na przemian A i B, zaczyna A, kończy B.
- Rozkład długości na 50 rozmów: 10 rozmów po 2 wymiany (4 linie), 20 rozmów po 3–4 wymiany (6–8 linii), 20 rozmów po 5–8 wymian (10–16 linii). Dłuższe rozmowy rozwijają temat: B dopytuje, A opowiada więcej, wątek idzie dalej.
- Różnicuj początki: nie zaczynaj kilku rozmów tak samo (np. nie co chwilę od „Cześć”).

## Format pliku, dokładnie taki
A: tekst wypowiedzi w jednej linii
B: tekst odpowiedzi w jednej linii
---
A: ...
B: ...
---

Rozmowy oddziela linia `---`, także po ostatniej. Nic poza tym: bez nagłówków, numerów i pustych linii.

## Jak pisać
Pisz plik w kilku kawałkach (np. Write dla pierwszych 10 rozmów, potem dopisuj przez Bash `cat >> plik <<'EOF'`). Na koniec policz linie `---` (`grep -c '^---$' plik`) i zwróć tylko tę liczbę.
```

### Tematy partii

| Partie | Tematy |
| --- | --- |
| 001–020 | pogoda i weekend, gotowanie, zakupy, zmęczenie po pracy, pies, sen, remont, pociąg, książka/film, egzamin, praca zdalna, nowy kolega, autobus, rower, bieganie, przeziębienie, lekarz, dentysta, urodziny, święta, dziadkowie, rodzeństwo, szkoła, kot, rośliny, działka, sprzątanie, pranie, przeprowadzka, mieszkanie, sąsiedzi, rachunki, zepsuty sprzęt, komputer, muzyka, koncert, planszówki, przepisy, ciasto, śniadanie, kawa, restauracja, morze, góry, zagranica, samolot, walizka, warsztat, nauka jazdy, korki, pory roku, język obcy, hobby, malowanie, fotografia, czytanie, serial, biblioteka, studia, praca dyplomowa, rozmowa o pracę, zmiana pracy, pierwszy dzień, nuda, stres, dobra wiadomość, zmartwienie, tęsknota, stary znajomy, telefon do mamy, spotkanie, wesele, mecz, basen, grzyby, ptaki, zoo, targ, piekarnia, ubrania, fryzjer, poczta, urząd, klucze, spóźnienie, kolejka, brak prądu, odpoczynek, drzemka, sny, nowy rok, dzieciństwo, film w domu, pomoc sąsiadowi |
| 021–040 | powitanie i „co słychać”, jak minął dzień, dobranoc, jednowyrazowe wiadomości A, nuda, prośba o zdanie, co na kolację, niezdecydowanie, drobny sukces, przeprosiny za brak odpowiedzi, kawa przed pracą, przerwa obiadowa, nadgodziny, urlop, poniedziałek, rodzinny obiad, sprzeczka i zgoda, prezent dla mamy, kuzyn, stare zdjęcia, pies sąsiada, adopcja kota, weterynarz, akwarium, ptaki zimą, nowy telefon, internet, gry, zdjęcia w telefonie, ekrany, przepis babci, zupa, nieudany obiad, zakupy na tydzień, owoce, wycieczka rowerowa, wypad za miasto, ognisko, biwak, rzeka, pierwsza praca, praktyki, współlokatorzy, akademik, sesja, urodziny dziecka, pierwszy dzień szkoły, zebranie, zadanie domowe, zabawki, plecy, zdrowe jedzenie, słodycze, woda, rozciąganie, meble, malowanie ścian, kanapa, piwnica, kran, szarówka, śnieg, burza, mgła, słońce po deszczu, muzyka do pracy, gitara, chór, płyty, radio, nieskończona książka, wiersze, czytanie dzieciom, księgarnia, kryminał, spóźniony pociąg, nocny autobus, zgubienie się, pytanie o drogę, taksówka, nieporozumienie z sąsiadem, hałas na klatce, parking, ogródek przed blokiem, zebranie mieszkańców, wolny dzień, leniwa niedziela, szafa, lista spraw, odkładanie na później, wakacje, dawny przyjaciel, liceum, pierwsze mieszkanie, pamiątki, a także A pytające B o niego samego (wolny czas, jedzenie, pora roku, weekend, zwierzęta) |
| 041–057 (długie) | te same grupy tematów co wyżej, po 5 tematów na partię, 6–8 wymian w każdej rozmowie |

## 2. Przygotowanie rozmów

```bash
PYTHONUTF8=1 uv run python scripts/prepare_dialogs.py
```

Walidacja odrzuca rozmowy w złym formacie, brzmiące jak asystent i duplikaty. W pierwszym przebiegu przyjęła 2420 z 2425 rozmów: 2297 do train i 123 do val.

## 3. Napisy filmowe i książki (korpus wstępny)

Źródło napisów to OpenSubtitles v2018 z korpusu OPUS, plik polski `pl.txt.gz` (1931 MB). Pobierane jest tylko pierwsze 300 MB.

```bash
PYTHONUTF8=1 uv run python scripts/prepare_pretrain.py --check-size   # tylko rozmiar, nic nie pobiera
PYTHONUTF8=1 uv run python scripts/prepare_pretrain.py                # pobranie 300 MB + filtr
```

Filtr odrzuca: bardzo krótkie wtrącenia, powtórki, dziwne znaki, przekleństwa (także „cholera”, „dupa”), linie WIELKIMI LITERAMI, angielskie linie i znaczniki napisów. Lista przekleństw to dwie krotki w `src/nanowriter/data.py` (`PROFANITY_STEMS`, `PROFANITY_WORDS`). Do napisów skrypt dokłada książki z `data/clean/` (stary korpus NanoWritera z Wolnych Lektur).

Wynik pierwszego przebiegu: przeczytano 24,0 mln linii, przyjęto 16,1 mln do train i 0,32 mln do val. Do tego 138 książek w train i 11 w val. Razem 613 MB train i 15 MB val.

Cały plik (1,93 GB) zamiast 300 MB: `--max-download-mb 2000 --allow-large`, po usunięciu `data/opus/pl.txt.gz`.

## 4. Tokenizer i tokeny

```bash
PYTHONUTF8=1 uv run python -u scripts/prepare_tokens_rozmowca.py
```

Trenuje BPE z 8192 tokenami na próbce (8,1 mln znaków korpusu i wszystkie rozmowy), koduje korpus i rozmowy, a dla rozmów zapisuje maskę straty. Maska ma 1 na treści wypowiedzi B (razem z końcowym `\n`) i 0 na całej reszcie, także na etykiecie `B:`.

Wynik: korpus train 177,4 mln tokenów (3,24 znaku na token), val 4,3 mln. Rozmowy train 268 tys. tokenów, z czego 45% ma maskę 1.

Uwaga: pierwsze uruchomienie przez nakładkę niskiego priorytetu (`cicho.ps1`) urwało się po cichu w połowie. Bezpośrednie `uv run python -u …` przeszło bez problemu w 3 minuty.

## 5. Trening wstępny

Model ma 6 warstw, 5 głowic, `d_model=320`, kontekst 256 i słownik 8192, co daje 10 101 760 parametrów. Na RTX 5060 Ti robi ok. 7 kroków na sekundę przy partii 64 × 256 i zajmuje mniej niż 1 GB pamięci GPU.

Pomiar szybkości (200 kroków, ok. 40 s):

```bash
PYTHONUTF8=1 uv run python scripts/train_gpt.py --data-dir data/rozmowca/tokeny-wstepne --out-dir data/rozmowca/ckpt-proba --n-layer 6 --n-head 5 --d-model 320 --block-size 256 --batch-size 64 --steps 200 --warmup 20 --eval-every 100 --checkpoint-every 0 --log-every 50
```

Właściwy trening, 12 000 kroków, czyli ~200 mln tokenów (reguła Chinchilla: ok. 20 tokenów na parametr):

```bash
PYTHONUTF8=1 uv run python -u scripts/train_gpt.py --data-dir data/rozmowca/tokeny-wstepne --out-dir data/rozmowca/ckpt-wstepny --n-layer 6 --n-head 5 --d-model 320 --block-size 256 --batch-size 64 --steps 12000 --lr 1e-3 --min-lr 1e-4 --warmup 500 --eval-every 500 --checkpoint-every 1000 --log-every 100
```

Przerwany trening wznawiasz tym samym wierszem z dopisanym `--resume`.

Wynik pierwszego przebiegu: 19,7 minuty (ok. 10 kroków na sekundę). Strata spadła z 9,07 (krok 0) do 3,29 treningowej i 4,09 walidacyjnej w kroku 11 999. Walidacja na całym zbiorze val: strata 3,51, czyli 1,56 bitu na znak. Strata walidacyjna spadała do końca, bez przeuczenia. Pełne wyjście jest w `data/rozmowca/ckpt-wstepny/trening-wyjscie.txt`.

**Kopia modelu wstępnego:** `data/rozmowca/ckpt-wstepny-kopia/` (identyczna bajt w bajt). To punkt startu dla kolejnych wersji douczania. Każdą nową wersję puszczaj z innym `--out-dir`, np. `--out-dir data/rozmowca/ckpt-rozmowy-2`, i z `--init-from data/rozmowca/ckpt-wstepny-kopia/gpt.pt`, jeśli oryginał kiedyś zginie.

## 6. Douczanie na rozmowach

```bash
PYTHONUTF8=1 uv run python scripts/finetune_rozmowca.py --steps <M>
```

Domyślne ścieżki: start z `data/rozmowca/ckpt-wstepny/gpt.pt`, dane z `data/rozmowca/tokeny-rozmowy`, wynik w `data/rozmowca/ckpt-rozmowy`. Strata liczy się tylko na wypowiedziach B. Liczbę kroków `<M>` szacujesz tak: ≈ 3 × liczba tokenów rozmów / (16 × 256), czyli ok. 200 kroków. Jeśli strata walidacyjna zaczyna rosnąć, model się przeucza: zmniejsz `--steps`.

Wynik pierwszego przebiegu (`--steps 200`, 7 sekund): strata walidacyjna na wypowiedziach B spadła z 4,18 do 3,37 (perplexity 29). Od kroku ~150 stała w miejscu (3,36–3,37), ale nie rosła, więc 200 kroków to górna granica. Więcej kroków przy tych 2420 rozmowach najpewniej da przeuczenie. Pełne wyjście jest w `data/rozmowca/douczanie-wyjscie.txt`.

Przykładowe odpowiedzi (seed 1):

| Ty | Model |
| --- | --- |
| cześć, co u ciebie? | Nie, nie, od kilku lat, mam już dość. |
| jestem dziś strasznie zmęczony | Nie ma co się bać |
| co robiłeś w weekend? | Nie, nie mogłem zasnąć, ale jestem zmęczony. |
| pada od rana, nie chce mi się wychodzić | Nie ma co się spieszyć w telefonie, ale lepiej mieć kogoś, kto się ucieszy niż ja. |

Odpowiedzi są gramatyczne, krótkie i spokojne w tonie, ale często nie na temat i często zaczynają się od „Nie”. To skutek mało liczebnych rozmów (2420) i małego modelu. Pomysły na kolejne wersje: więcej rozmów, osobno długich; dropout przy douczaniu; niższa temperatura w czacie.

## 7. Rozmowa

```bash
PYTHONUTF8=1 uv run python scripts/chat.py --checkpoint data/rozmowca/ckpt-rozmowy/gpt.pt --tokenizer data/rozmowca/tokeny-wstepne/tokenizer.json
```

Piszesz linię, model odpowiada jedną linią. Pamięta ostatnie 4 wymiany (`--max-turns`). Rozmowę kończysz `/koniec` albo Ctrl+C. Domyślne próbkowanie: temperatura 0,7, top-k 40, top-p 0,9.

## Kod

Kod powstał w zadaniach 07–10 (moduły `data`, `tokenizer`, `dataset`, `train`, `sample` oraz skrypty `prepare_dialogs.py`, `prepare_pretrain.py`, `prepare_tokens_rozmowca.py`, `finetune_rozmowca.py`, `chat.py`). Każde zadanie ma testy i test bramki w `tests/`.
