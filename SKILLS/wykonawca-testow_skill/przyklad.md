# Przykład: ze specyfikacji do testu

Ilustracja. Projekt i nazwy są zmyślone (ten sam przykład co w `SKILLS/tester_skill/przyklad.md`). Pokazuje, że test robi dokładnie to, co mówi specyfikacja, i nic więcej.

Specyfikacja:

```markdown
## Przygotowanie
- Import: `from app.exporters import CsvExporter`
- Obiekty: `CsvExporter` z katalogiem tymczasowym jako `out_dir`.
- Zasoby: katalog tymczasowy.

### test_csv_export_example
- Przypadek: przykład z docstringu.
- `export([{"a": "1", "b": "2"}], "x")` → zwraca `out_dir / "x.csv"`, plik w UTF-8 ma treść dokładnie `"a,b\n1,2\n"`
```

Kod testu. Katalog tymczasowy to fixture `tmp_path` z pytest, a sposób odczytu pliku wybrał wykonawca:

```python
from app.exporters import CsvExporter


def test_csv_export_example(tmp_path):
    path = CsvExporter(tmp_path).export([{"a": "1", "b": "2"}], "x")

    assert path == tmp_path / "x.csv"
    # Dokładna treść, łącznie z końcami linii: "\r\n" to też błąd.
    assert path.read_text(encoding="utf-8") == "a,b\n1,2\n"
```
