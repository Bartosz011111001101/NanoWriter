---
name: planner_skill
description: Używaj, gdy zadanie trzeba rozpisać na fale i prompty kodu dla wykonawców (Haiku, Sonnet, Opus), ocenić trudność promptu, opisać wykonawcy stan obecny kodu albo napisać lub zmienić kontrakt modułu.
---

# Planowanie

Prompt kodu daje wykonawcy całą wiedzę potrzebną do pracy: co jest teraz w kodzie, co ma powstać, gdzie to wpiąć i na co uważać. Wykonawca nie przeszukuje repozytorium i nie projektuje niczego, czego prompt mu wprost nie zleca. Dobrze rozpisany prompt zrobi Haiku, nawet jeśli to cały typ z powiązaniami (typ bazowy, interfejs) albo trudny algorytm.

**Ty projektujesz, wykonawca implementuje.** Projekt to struktura (pliki, typy i ich powiązania: dziedziczenie, interfejsy, kompozycja), sygnatury (argumenty, typy, zwracane wartości) i zachowanie (co ma się stać dla jakiego wejścia). Implementacja, czyli ciała funkcji, należy do wykonawcy. W prompcie nie piszesz ciał funkcji ani kodu w krokach.

## Na start czytasz

1. `SKILLS/projekt/profil.md`: stack, komendy, układ repozytorium, mapa modułów, etapy, decyzje użytkownika.
2. `SKILLS/projekt/konwencje.md`: format opisu funkcji, reguła prywatności, styl. Szkielety w „Ma być” piszesz zgodnie z konwencjami, ale samych konwencji nie przepisujesz do promptu, bo wykonawcy czytają je sami.
3. Pliki pułapek wymienione w profilu (`SKILLS/projekt/pulapki/`).
4. Plan wstępny, jeśli profil go podaje, kontrakty z `Docs/Kontrakty/` i raporty z `Docs/Postep/` poprzednich zadań.
5. **Aktualny kod każdego pliku, który zadanie zmienia.** Z niego, a nie z pamięci ani z kontraktu, piszesz sekcję „Stan obecny”.

`Docs/` jest w `.gitignore`, więc Grep i Glob mogą go pomijać. Listuj przez `ls`, czytaj po pełnej ścieżce.

## Rozpoznanie kodu: gdzie to umieścić

Zanim rozpiszesz prompty, ustal dla każdej rzeczy, której zadanie potrzebuje (funkcja, klasa, stała, typ), czy już istnieje i gdzie ma żyć.

1. Uruchom „Mapę kodu” z profilu. Dostajesz wszystkie publiczne typy, funkcje i metody projektu.
2. Dla każdej potrzebnej rzeczy uruchom „Szukanie istniejącego kodu” z profilu. Szukaj po nazwie i po 2–3 słowach opisujących, co robi (np. `path|ścieżk|katalog`).
3. Rozstrzygnij według tabeli. Bierz pierwszy pasujący wiersz:

| Co znalazłeś | Decyzja | W prompcie |
| --- | --- | --- |
| Istnieje i robi dokładnie to | Używasz istniejącego. | „Do użycia” z sygnaturą i importem. |
| Istnieje i robi prawie to | Rozszerzasz istniejące (nowy argument z wartością domyślną, nowa metoda). Nowej, podobnej funkcji nie tworzysz. | Prompt zmienia tę funkcję: „Było” i „Ma być”. |
| Potrzebują tego co najmniej dwa moduły | Jedno miejsce: najniższa warstwa (a bez warstw: wspólny moduł), z której oba mogą importować, we wcześniejszej fali. | Wcześniejszy prompt to tworzy, późniejsze mają to w „Do użycia”. |
| Nie istnieje, pasuje do „Odpowiada za” któregoś modułu z profilu | Nowy kod w tym module. | Zwykły prompt. |
| Nie istnieje i nie pasuje do żadnego modułu | Nie tworzysz nowego modułu. | Wpis w „Otwartych pytaniach”: proponowany moduł i jego warstwa, jeśli projekt ma warstwy. |

4. Jeśli sekcja „Warstwy” w profilu ma wartość inną niż „brak”, sprawdź z nią kierunek każdego importu. Import z wyższej warstwy oznacza, że kod jest w złym miejscu: przenieś wspólny kawałek niżej, zamiast importować w górę. Gdy projekt nie ma warstw, pilnujesz tylko, żeby dwa moduły nie importowały się nawzajem.
5. Nowa biblioteka zewnętrzna tylko wtedy, gdy nie wystarczy ani biblioteka standardowa, ani zależności już zainstalowane. Taka biblioteka zawsze trafia do „Otwartych pytań”.

## Co tworzysz

```
Docs/Plan/NN-zadanie/plan.md
Docs/Prompts/NN-zadanie/kod/PP-nazwa.md      jeden plik na kawałek pracy
Docs/Kontrakty/<modul>.md                    nowe albo zmienione kontrakty
```

Promptów testów nie piszesz, robi to tester. W każdym prompcie kodu zostawiasz sekcję „Propozycja testów”.

## Cięcie na prompty i fale

- Jeden prompt to jeden spójny kawałek: kilka funkcji albo jeden typ z metodami.
- **Fala** to prompty, które idą równolegle. Prompty w jednej fali:
  - nie zmieniają tych samych plików;
  - nie zależą od siebie (żaden nie używa tego, co pisze inny prompt tej samej fali).
- Prompt korzystający z wyniku wcześniejszego promptu idzie do późniejszej fali.
- **Jak najmniej fal.** Każda fala to pełna runda testów i weryfikacji. Układaj fale tak, żeby w każdej pracowało kilka modułów naraz, zamiast dawać jeden moduł na falę.
- Gdy jeden plik dostaje w zadaniu więcej niż 3 prompty, zaproponuj podział tego pliku na mniejsze. Jeśli plik ma już kod, wpisz to do „Otwartych pytań”.
- Najpierw najprostsza działająca wersja, optymalizacje w późniejszych falach.
- Ostatnia fala zawiera test bramki z profilu, jeśli zadanie zamyka etap.

## Trudność

Trudność zależy od zakresu zmian i od tego, czy umiesz rozwiązanie rozpisać z góry. Złożoność samego algorytmu jej nie podnosi. Sprawdzaj wiersze od góry i bierz pierwszy pasujący:

| Trudność | Model kodu | Warunek |
| --- | --- | --- |
| trudne | Opus | Nie umiesz rozpisać rozwiązania z góry, bo trzeba je zaprojektować albo zbadać, **albo** prompt zmienia więcej niż 5 plików. |
| średnie | Sonnet | Prompt zmienia 3–5 plików **albo** przerabia istniejącą logikę, z której korzystają inne moduły, i zmiany muszą się zgadzać w kilku miejscach. |
| łatwe | Haiku | Do 2 plików, struktura i sygnatury są ustalone, a każdy przypadek zachowania jest wypisany w „Krokach”. |

Testy nigdy nie idą na Haiku. Model wykonawcy testów dobiera tester (minimum Sonnet).

Do liczby plików nie wliczasz plików, w których prompt dopisuje tylko rejestrację nowego kodu (lista eksportów, deklaracja modułu, wpis w manifeście). Co w tym projekcie znaczy „moduł”, mówi sekcja „Mapa modułów” profilu.

Wahasz się między łatwe a średnie? Rozpisz dokładniej „Stan obecny” i „Kroki” albo podziel prompt. „Trudne” to wyjątek.

## Szablon: prompt kodu

Sekcje zawsze w tej kolejności i zawsze wszystkie, niezależnie od trudności. Pole, które nie dotyczy promptu, ma wartość „brak”. Przed pierwszym promptem zadania przeczytaj `SKILLS/planner_skill/przyklad.md`, bo pokazuje poziom szczegółu, który wystarcza Haiku.

````markdown
# <PP-nazwa>

- **Trudność:** łatwe | średnie | trudne
- **Fala:** <F>
- **Zależy od:** <PP-nazwa, ...> | brak
- **Pliki (tylko te wolno zmieniać):** <ścieżki>
- **Kontrakt:** Docs/Kontrakty/<modul>.md

## Cel
<co powstaje i po co, jedno zdanie>

## Stan obecny
- **Pliki:** <które istnieją i z której fali, które są nowe>
- **Struktura `<plik>`:** <zarys każdego zmienianego pliku: importy, typy, funkcje, sygnatury, bez ciał>
- **Typy:** <typ (klasa, struktura, interfejs): co rozszerza albo implementuje, jak się go tworzy, pola, które metody musi mieć albo nadpisać, co wywołać z typu bazowego> | brak
- **Do użycia:** <istniejąca funkcja, typ albo stała: sygnatura, import, do czego jej użyć> | brak
- **Importy:** <linie do dopisania>. Wolno z: <moduły>. Nie wolno z: <moduły>. Korzysta z tego kodu: <kto i jak wywołuje>
- **Współbieżność:** <co działa gdzie (worker, wątek, proces, async), jak się komunikuje, start i stop> | brak
- **Dane i stan:** <formaty plików, konfiguracja, zmienne środowiska> | brak
- **Punkt wpięcia:** <gdzie dokładnie dopisać>
- **Nie ruszać:** <co zostaje bez zmian>

## Ma być
<szkielet w języku projektu: pełne sygnatury z typami, opis funkcji (komentarz dokumentujący w formacie z konwencji projektu) z przykładem wejście → wyjście, bez ciała funkcji>

## Kroki
1. <wejście albo warunek> → <efekt>

## Uwaga
- <wpis z pliku pułapek, skopiowany w całości> | brak

## Gotowe, gdy
- `<komenda>` wypisuje `<dokładny wynik>`.

## Propozycja testów (dla testera)
- <przypadek: wejście → oczekiwany wynik>
````

**Ma być.** Zmiana istniejącej funkcji: „Było” (zarys) i „Ma być” (pełny szkielet). Prompt trudny: ustalasz granice, czyli publiczne sygnatury, niezmienniki i to, co ma przejść. Środek oznaczasz jako „Do zaprojektowania przez wykonawcę: …”. Wykonawca opisuje swój projekt w raporcie, a Ty bierzesz go do promptów następnych fal.

**Kroki.** Zachowanie, nie kod. Każdy krok ma postać „wejście albo warunek → efekt” i używa tylko nazw ze szkieletu i z „Do użycia”. Nie podajesz wywołań bibliotek ani instrukcji języka. Jeśli wynik zależy od konkretnego wywołania (np. tryb otwarcia pliku, kolejność argumentów), opisujesz to w „Uwadze” jako pułapkę. Ile kroków:
- łatwe: każdy przypadek osobno, łącznie z pustym wejściem, granicami i błędami;
- średnie: zachowanie każdej funkcji i to, jak funkcje ze sobą współpracują;
- trudne: niezmienniki, które muszą zachodzić, i sposób sprawdzenia poprawności.

**Gotowe, gdy.** Jedna linia, komenda „Szybkie sprawdzenie” z profilu. Działa w kilka sekund, bez sieci, bez zapisu poza katalogiem tymczasowym i bez operacji z sekcji „Ograniczenia” profilu. Zasady zapisu kodu wpisanego w komendę (znaki, cudzysłowy) bierzesz z plików pułapek. Nigdy nie wymaga plików testów, bo wykonawca kodu ich nie pisze. Gdy „Szybkie sprawdzenie” ma w profilu wartość „brak”, wpisujesz komendę „Sprawdzenie statyczne” z profilu i wynik „bez błędów w plikach promptu”.

## Szablon: plan.md

```markdown
# Zadanie NN: <nazwa>

## Cel
<jedno zdanie>. Gotowe, gdy: <coś do uruchomienia i zobaczenia>.

## Fale
| Fala | Prompt | Trudność | Pliki |
| --- | --- | --- | --- |
| <F> | <PP-nazwa> | łatwe / średnie / trudne | <ścieżki> |

## Kontrakty
- utworzone/zmienione: Docs/Kontrakty/<modul>.md

## Otwarte pytania
- <decyzja dla użytkownika> albo "brak"
```

## Szablon: kontrakt

Plik `Docs/Kontrakty/<modul>.md`, sekcje:

- **Odpowiedzialność**
- **Publiczne API:** sygnatury z typami, typy z powiązaniami (dziedziczenie, implementowane interfejsy).
- **Struktury danych:** dla każdej struktury nazwa, typ i kształt albo format.
- **Pliki:** formaty na dysku.
- **Niezmienniki:** co tester ma sprawdzić.
- **Zmiany:** data i opis.

Szkielety w promptach muszą być zgodne z kontraktem. Każda zmiana kontraktu to wpis w „Zmiany”.

## Przed oddaniem planu

Sprawdź każdy punkt. Jeśli któryś nie przechodzi, popraw prompt:

- [ ] „Stan obecny” każdego promptu jest przepisany z aktualnego kodu albo ze szkieletu promptu wcześniejszej fali, jeśli kod jeszcze nie istnieje.
- [ ] Każda nowa funkcja i każdy nowy typ przeszły „Rozpoznanie kodu”: nic podobnego nie istnieje, a jeśli istnieje, jest w „Do użycia” albo prompt je rozszerza.
- [ ] Żadna logika nie jest zaplanowana w dwóch promptach naraz.
- [ ] Żaden import w „Importach” nie tworzy cyklu, a gdy projekt ma warstwy, każdy idzie z wyższej warstwy do niższej.
- [ ] Każdy przykład w opisie funkcji jest policzony ręcznie i się zgadza, łącznie z białymi znakami i typem wyniku.
- [ ] Żadne dwa prompty jednej fali nie mają wspólnego pliku w „Plikach”.
- [ ] Wszystko, czego prompt używa, powstaje we wcześniejszej fali albo już istnieje.
- [ ] Szkielety zgadzają się z kontraktem.
- [ ] Trudność wynika z tabeli, czyli liczby plików i tego, czy każdy przypadek zachowania jest wypisany.
- [ ] W „Ma być” nie ma ciał funkcji, a w „Krokach” kodu ani wywołań bibliotek.
- [ ] Komenda „Gotowe, gdy” spełnia zasady.

## Odpowiedź dla orkiestratora

1. Ścieżka planu i lista fal (numer fali, nazwy promptów, trudności).
2. Utworzone i zmienione kontrakty.
3. Otwarte pytania albo „brak”.

## Częste błędy

| Błąd | Poprawka |
| --- | --- |
| „Stan obecny” napisany z pamięci albo z kontraktu | Przeczytaj plik i przepisz zarys z kodu. Wykonawca zablokuje się na pierwszej niezgodności. |
| Nowa funkcja pomocnicza, choć podobna już istnieje | Uruchom „Szukanie istniejącego kodu” i wpisz znalezioną do „Do użycia”. |
| Ta sama logika w dwóch promptach różnych modułów | Jeden wcześniejszy prompt w najniższej wspólnej warstwie, w pozostałych „Do użycia”. |
| Import z wyższej warstwy „bo tam już jest” | Przenieś ten kawałek do niższej warstwy w osobnym, wcześniejszym prompcie. |
| Szkielet bez typów albo bez przykładu w opisie funkcji | Dopisz typy i przykład wejście → wyjście. |
| Ciało funkcji w „Ma być” albo kod i wywołania bibliotek w „Krokach” | Zostaw sygnaturę z pominiętym ciałem. Kroki zapisz jako „wejście → efekt”. Szczegół wywołania, od którego zależy wynik, przenieś do „Uwagi”. |
| Typ bez informacji, co rozszerza albo implementuje i co wywołać z typu bazowego | Uzupełnij pole „Typy” w „Stanie obecnym”. |
| Trudność podniesiona, bo algorytm jest trudny, choć kroki go opisują | Zostaw „łatwe”. Trudność to zakres i to, czy rozwiązanie jest rozpisane. |
| Jeden moduł na falę, przez co fal jest dużo | Połącz w jednej fali prompty z różnych modułów. |
| Dwa prompty jednej fali zmieniają ten sam plik | Przenieś jeden do następnej fali. |
| „Gotowe, gdy” wymaga testów albo trwa minuty | Daj komendę „Szybkie sprawdzenie” z profilu, działającą w kilka sekund. |
| Decyzja użytkownika podjęta przez plannera | Przenieś ją do „Otwartych pytań”. |
