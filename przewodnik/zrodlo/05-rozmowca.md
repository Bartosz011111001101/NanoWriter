
# 9. Drugi model: rozmówca 10M

Model 1M pisze coś, co wygląda jak polska proza z XIX wieku, ale nie da się z nim porozmawiać. Ten rozdział opisuje, co trzeba było zmienić, żeby powstał model, który odpowiada na zdanie jednym zdaniem, i dlaczego właśnie to. Pełna instrukcja odtworzenia z komendami jest w `ROZMOWCA.md`.

## 9.1 Co się zmieniło i dlaczego

| | Model 1M | Rozmówca 10M | Dlaczego zmiana |
| --- | --- | --- | --- |
| Dane | 22,8 mln znaków książek | 575 mln znaków: napisy filmowe + te same książki | Rozmowa to dialog, a w napisach jest go najwięcej. Model uczy się z tego, co czyta. |
| Słownik BPE | 1024 | 8192 | Więcej tekstu w tym samym kontekście (9.2). |
| Warstwy × głowice × `d_model` | 4 × 4 × 128 | 6 × 5 × 320 | Więcej danych udźwignie większy model (9.3). |
| Parametry | 957 184 | 10 101 760 | ok. 10,5 razy więcej |
| Trening | jeden etap, 5000 kroków | pretrening 12 000 kroków + douczanie 200 kroków | Najpierw język, potem styl rozmowy (9.4). |
| Strata | zwykła, na każdym tokenie | przy douczaniu tylko na odpowiedziach | Model ma się uczyć odpowiadać, a nie zadawać pytania (9.4). |

> **Najważniejsza lekcja tego rozdziału.** Architektura obu modeli jest **ta sama**: ten sam `model.py`, ta sama pętla `train.py`, to samo `sample.py`. Zmieniły się tylko dane, rozmiar i sposób liczenia straty. Tak samo jest z dużymi modelami: różnica między modelem, który dopisuje tekst, a asystentem, który odpowiada, to przede wszystkim dane i etap douczania, a nie inna sieć.

## 9.2 Większy słownik: 1024 → 8192

![To samo zdanie pocięte słownikiem modelu 1M (1024 tokeny) i modelu 10M (8192 tokeny).](wyniki/tokenizacja.png)

| Zdanie | Słownik 1024 | Słownik 8192 |
| --- | --- | --- |
| „Wieczorem stary kot zasnął przy piecu.” | 16 tokenów (2,4 znaku na token) | 11 tokenów (3,5 znaku na token) |
| „Dzień dobry, jak minął ci weekend?” | 15 tokenów (2,3) | 9 tokenów (3,8) |
| „Zażółć gęślą jaźń.” | 13 tokenów (1,4) | 11 tokenów (1,6) |

Co z tego wynika:

- **Częste słowa stają się jednym tokenem:** „·stary”, „·kot”, „·weekend”, „Dzień”. Model nie musi już składać ich z kawałków, więc może się skupić na wyborze słowa, czyli na tym, co w tabeli z rozdziału 8.1 szło mu najsłabiej.
- **Kontekst „mieści” więcej tekstu.** 256 tokenów to przy słowniku 1024 ok. 590 znaków, a przy 8192 ok. 830 znaków. W rozmowie oznacza to kilka wymian więcej w pamięci.
- **Rzadkie ciągi liter dalej się rozpadają.** „Zażółć gęślą jaźń” to celowo dziwne zdanie, więc oba słowniki tną je prawie na litery.

> **Koszt większego słownika.** Embedding rośnie liniowo z rozmiarem słownika: 8192 × 320 = 2,6 mln parametrów, czyli 26% całego modelu 10M. Słownik 65 536 zjadłby 21 mln parametrów, dwa razy więcej niż cały model.

## 9.3 Większy model: ile parametrów na ile danych

![Lewy wykres: z czego składają się parametry obu modeli. Prawy: bity na znak na własnym zbiorze walidacyjnym każdego modelu.](wyniki/rozmiary.png)

**Reguła Chinchilla** mówi, że przy ustalonej mocy obliczeniowej najlepiej wychodzi ok. 20 tokenów treningowych na parametr. Twoje modele:

| | Parametry | Tokeny przeczytane w treningu | Tokenów na parametr | Ile razy przeczytał cały korpus |
| --- | --- | --- | --- | --- |
| 1M | 0,96 mln | 81,9 mln | ok. 86 | 8,3 razy |
| 10M | 10,1 mln | 196,6 mln (12 000 × 64 × 256) | ok. 19,5 | ok. 1,1 raza |

Model 1M był więc „przetrenowany” w sensie Chinchilli: dużo kroków na mały model i mały korpus, który czytał w kółko. Lepszy wynik dałby większy model na tych samych danych. Model 10M trafia prawie dokładnie w regułę.

> **Uwaga przy porównywaniu bitów na znak (prawy wykres).** 1,92 i 1,82 to wyniki na **różnych** zbiorach walidacyjnych: model 1M był sprawdzany na książkach, a 10M na napisach i książkach. Napisy są prostsze (krótkie, potoczne zdania), więc porównanie jest tylko orientacyjne.

Próbki obu modeli po tym samym początku „Wieczorem stary kot”:

```text
1M:          Wieczorem stary kot i cóż?! Od typu cierpienia, o którym drzemał
             ukryte zapomnieniem, zgroził się.
             — Dostałbym wczoraj umarłym.

10M wstępny: Wieczorem stary kotku.
             On jest za młody w moim życiu.
             Czy ktoś mnie słyszy?
             Panie Vasquez!
             Czy ten mężczyzna był jednym z nich?
```

Widać po nich dane treningowe. Model 1M pisze długie, książkowe zdania z myślnikami dialogów. Model 10M po pretreningu pisze krótkie linie jak w napisach filmowych, każda od nowej linii, z imionami bohaterów. Zdania są poprawniejsze, ale **jeszcze nie odpowiada**, tylko dopisuje kolejne kwestie z filmu.

## 9.4 Pretrening i douczanie z maską straty

Trening rozmówcy ma dwa etapy:

1. **Pretrening** (`scripts/train_gpt.py`, ok. 20 minut): 12 000 kroków na 177 mln tokenów napisów i książek. Strata spadła z 9,07 do 3,29 (train). Model uczy się **języka**: słów, gramatyki, rytmu dialogu.
2. **Douczanie** (`scripts/finetune_rozmowca.py`, 7 sekund): 200 kroków na 2420 krótkich rozmowach w formacie `A: …` / `B: …`. Model uczy się **zadania**: na linię A odpowiedz jedną spokojną linią B.

> **Po co dwa etapy, a nie od razu trening na rozmowach?** 2420 rozmów to tylko ok. 268 tys. tokenów, 660 razy mniej niż korpus pretreningu. Na tak małym zbiorze model od zera nauczyłby się najwyżej formatu „A:/B:”, ale nie polskiego. Pretrening daje język, a douczanie tylko „ustawia” go w trybie rozmowy.

> **Bez ostrożności przy douczaniu…** Za dużo kroków na małym zbiorze i model nauczy się rozmów na pamięć. Dlatego douczanie ma niski LR (0,0003 zamiast 0,001) i tylko 200 kroków. Strata walidacyjna od kroku ok. 150 stała w miejscu (3,36–3,37), więc więcej kroków nic by nie dało.

**Maska straty.** W rozmowie są dwie role, a model ma się nauczyć tylko jednej. Gdyby strata liczyła się na każdym tokenie, model uczyłby się też pisać kwestie A, czyli zadawać pytania zamiast na nie odpowiadać. Rozwiązanie: każdy token dostaje 0 albo 1, a strata liczy się tylko tam, gdzie jest 1.

![Rozmowa przykładowa po tokenizacji. Zielone tokeny (treść wypowiedzi B razem z końcowym znakiem nowej linii) liczą się do straty, szare nie. Mała liczba pod tokenem to strata modelu wstępnego na tej pozycji. Animacja: `przewodnik/wyniki/maska_straty.gif`.](wyniki/maska_straty.png)

Szczegóły, które łatwo przeoczyć:

- **Etykieta „B:” też jest szara.** Format rozmowy zawsze ją wstawia, więc nie ma czego się uczyć. W czacie kontekst budowany przez `build_context` (`sample.py`) sam kończy się na „B:”, a model pisze dalej.
- **Końcowe `\n` w wypowiedzi B jest zielone.** To ważne: model uczy się **kiedy skończyć**. Bez tego pisałby w nieskończoność, bo nigdy nie dostałby sygnału, że po odpowiedzi trzeba postawić koniec linii.
- **Strata na zielonych tokenach jest niższa niż średnia** (3,20 kontra 4,20). Kwestie A zaczynają nowy temat, którego model nie mógł przewidzieć. Odpowiedź B wynika z pytania, więc jest łatwiejsza.

W kodzie to jedna różnica w `compute_loss` (`train.py`): z maską liczona jest średnia ważona strat pozycji, a wagami są zera i jedynki maski.

## 9.5 Wynik i jego granice

Po douczaniu strata walidacyjna na odpowiedziach B spadła z 4,18 do 3,37 (perplexity ok. 29). Przykładowe odpowiedzi (seed 1):

| Ty | Model |
| --- | --- |
| cześć, co u ciebie? | Nie, nie, od kilku lat, mam już dość. |
| jestem dziś strasznie zmęczony | Nie ma co się bać |
| co robiłeś w weekend? | Nie, nie mogłem zasnąć, ale jestem zmęczony. |
| pada od rana, nie chce mi się wychodzić | Nie ma co się spieszyć w telefonie, ale lepiej mieć kogoś, kto się ucieszy niż ja. |

Co model umie po douczaniu: odpowiada **jedną linią**, gramatycznie, krótko i spokojnym tonem. To wszystko nauczyło się w 200 krokach, dzięki pretreningowi. Czego nie umie: trzymać się tematu. Odpowiedzi często są obok pytania i zaczynają się od „Nie”.

> **Dlaczego „Nie” na początku?** W napisach filmowych „Nie” to najczęstsze pierwsze słowo kwestii: w pierwszych 300 tys. linii zbioru `data/pretrain/val.txt` zaczyna co szesnastą linię (17 682 razy), przed „To” i myślnikiem. Model po pretreningu ma silne przekonanie, że kwestia dialogu zaczyna się od „Nie”, a 2420 rozmów nie wystarczyło, żeby je przełamać. To dobry przykład tego, że **pretrening zostawia ślad**, którego krótkie douczanie nie usuwa.

Kierunki poprawy wynikają z tego, co widać wyżej: więcej rozmów do douczania (szczególnie dłuższych), dropout przy douczaniu przeciw uczeniu na pamięć i niższa temperatura w czacie, żeby rzadziej losować odpowiedzi z ogona.

> **Sprawdź sam.**
>
> ```bash
> uv run python scripts/chat.py --checkpoint data/rozmowca/ckpt-rozmowy/gpt.pt --tokenizer data/rozmowca/tokeny-wstepne/tokenizer.json
> ```
>
> W PowerShell najpierw `$env:PYTHONUTF8 = "1"`. Rozmowę kończy `/koniec`.
