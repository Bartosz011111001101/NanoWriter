# Szablon: SKILLS/projekt/ w nowym projekcie

Skille wołają poniższe nazwy dosłownie, więc każda musi istnieć. Gdy projekt czegoś nie ma, wpisz „brak”. Agent pomija wtedy ten krok i zapisuje to w raporcie.

## profil.md: sekcje

| Sekcja | Co zawiera | Kto używa |
| --- | --- | --- |
| Projekt | 1–2 zdania: co powstaje | wszyscy |
| Stack i system | język, menedżer pakietów, narzędzie testów, system i powłoka agentów | wszyscy |
| Komendy | tabela niżej | wszyscy |
| Konfiguracja sprawdzenia statycznego | pliki i sekcje konfiguracji narzędzi (tylko do odczytu) | wszyscy |
| Układ repozytorium | gdzie kod, gdzie testy, wzór ścieżki pliku testów jednego promptu, plik wspólnego przygotowania testów, plan wstępny albo „brak” | planner, tester |
| Pułapki obowiązujące w projekcie | lista plików z `SKILLS/projekt/pulapki/` | planner, tester |
| Mapa modułów | co w tym projekcie znaczy „moduł” (plik, pakiet, katalog) oraz tabela: moduł, „Odpowiada za”, etap | planner |
| Warstwy | kolejność od najwyższej do najniższej i narzędzie, które jej pilnuje, albo „brak” | planner, tester |
| Etapy i testy bramki | tabela etap → test bramki, albo „brak” | orkiestrator, planner |
| Decyzje użytkownika | czego agenci nie rozstrzygają sami | orkiestrator, planner |
| Ograniczenia | operacje, przed którymi trzeba zapytać | orkiestrator, planner |
| Git | gałąź, na której commituje orkiestrator, i czy wolno wypychać | orkiestrator |

## profil.md: wiersze tabeli „Komendy” (nazwy dokładnie takie)

| Nazwa | Do czego służy |
| --- | --- |
| Szybkie sprawdzenie | komenda dla „Gotowe, gdy”: uruchamia kawałek kodu bez plików testów; „brak”, gdy język tego nie umie |
| Wszystkie testy | weryfikacja fali |
| Jeden plik testów | pełna komenda uruchamiająca testy jednego pliku |
| Sprawdzenie składni | „brak”, gdy język nie ma osobnego sprawdzenia składni |
| Sprawdzenie statyczne jednego pliku | gdy narzędzie działa tylko na pakiecie, komenda dla pakietu |
| Sprawdzenie statyczne | cały projekt |
| Mapa kodu | lista publicznych typów, funkcji i metod |
| Szukanie istniejącego kodu | szukanie po słowach w kodzie i w kontraktach |
| Nowa biblioteka | jak dodać zależność |

## konwencje.md

- Sekcja „Kod”: styl; czym w tym języku jest „opis funkcji” i jaki ma format; jak oznaczyć funkcję pomocniczą jako prywatną.
- Sekcja „Testy”: czym jest „wspólne przygotowanie testów” i gdzie leży; zasady testów.

## katalog-testow.md

- Przypadki brzegowe dziedziny.
- Tabela: moduł, test, co sprawdza.

## pulapki/<technologia>.md

Pierwsze linie każdego pliku:

    # Pułapki: <technologia>
    Format: **Obszar.** Objaw → przyczyna → jak uniknąć.
