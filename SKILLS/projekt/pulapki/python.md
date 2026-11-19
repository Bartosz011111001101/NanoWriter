# Pułapki: Python

Plik wymienny między projektami. Planner kopiuje pasujące wpisy do sekcji „Uwaga” w prompcie, a tester używa ich jako hipotez przy diagnozie. Nowy wpis dopisuje tester po rundzie poprawek, jeśli przyczyna może się powtórzyć w innym miejscu.

Format: **Obszar.** Objaw → przyczyna → jak uniknąć.

- **Argumenty domyślne.** Lista lub słownik „pamięta” wartości między wywołaniami → mutowalny argument domyślny (`def f(x=[])`) → `x: list | None = None`, a w ciele `x = [] if x is None else x`.
- **Kodowanie plików.** Polskie znaki psują się albo jest `UnicodeDecodeError` na Windowsie → `open()` bez `encoding` używa cp1250 → zawsze `encoding="utf-8"`.
- **Kodowanie konsoli.** Program albo narzędzie wywraca się z `UnicodeEncodeError: 'charmap'` przy wypisywaniu, ale tylko z przekierowanym wyjściem (`>`, plik, agent) → Windows używa cp1250 dla stdout → uruchamiaj z `PYTHONUTF8=1`.
- **Moduł csv.** Na Windowsie puste linie między wierszami albo `\r\n` zamiast `\n` → plik otwarty bez `newline=""` i domyślny `lineterminator` → `open(..., newline="")` i `csv.writer(f, lineterminator="\n")`.
- **Zakresy i wycinki.** Brakuje ostatniego elementu albo jeden jest podwójnie → pomylone `range(n)` i `range(n + 1)`, `x[a:b]` z `b` włącznie → sprawdź przykład z docstringu na granicy (0, 1, n).
- **Puste wejście.** `IndexError` albo `ZeroDivisionError` dla pustej listy lub tekstu → kod zakłada co najmniej jeden element → obsłuż pusty przypadek na początku funkcji, zgodnie z docstringiem.
- **Kopia zamiast referencji.** Zmiana wyniku zmienia wejście wywołującego → zwrócona ta sama lista lub słownik → kopiuj (`list(x)`, `dict(x)`), gdy docstring mówi o nowym obiekcie.
- **Komendy `-c`.** `uv run python -c "..."` zwraca błąd składni albo krzaki → cudzysłowy, nowe linie lub polskie znaki w komendzie jednolinijkowej → jedna linia, tylko ASCII, cudzysłowy pojedyncze wewnątrz podwójnych.
- **Import cykliczny.** `ImportError: cannot import name` przy imporcie modułu → dwa moduły importują się nawzajem na górze pliku → zależność w jedną stronę, wspólne typy w osobnym module.
- **Dziedziczenie.** Pole klasy bazowej nie istnieje w podklasie → brak `super().__init__(...)` w konstruktorze podklasy → wywołaj go jako pierwszą instrukcję.
- **Kolejność słownika.** Wynik różni się od przykładu kolejnością → kod opiera się na kolejności iteracji po `set` → sortuj albo używaj `dict` lub `list`, gdy kolejność ma znaczenie.
- **Długość linii w opisach funkcji.** `ruff check` zgłasza E501 w docstringu albo komentarzu, choć ciało funkcji jest czyste → przykład wywołania albo zdanie w opisie funkcji ze szkieletu dłuższe niż `line-length` (tu 100), a wykonawca nie zmienia opisów funkcji → już w szkielecie zawijaj opisy i przykłady do 100 znaków (przykład z wieloma argumentami rozbij na kilka linii z wcięciem).
- **Końce linii przy zapisie tekstu.** Plik zapisany na Windowsie ma `\r\n` zamiast `\n`, a test porównujący bajty pliku (JSON, tekst, `meta`) nie przechodzi → `Path.write_text(...)` albo `open(..., "w")` bez `newline` tłumaczy każde `"\n"` na `os.linesep` → `write_text(..., encoding="utf-8", newline="\n")` albo `open(..., "w", encoding="utf-8", newline="\n")`, a gdy kontrakt podaje dokładne bajty, także `write_bytes(tekst.encode("utf-8"))`.
