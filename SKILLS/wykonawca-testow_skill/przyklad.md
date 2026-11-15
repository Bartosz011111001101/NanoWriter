# Przykład: ze specyfikacji do testu

Ilustracja. Projekt i nazwy są zmyślone (ten sam przykład co w `SKILLS/tester_skill/przyklad.md`). Pokazuje, że test robi dokładnie to, co mówi specyfikacja, i nic więcej.

Specyfikacja:

```markdown
### test_csv_export_example
- Wejście: `CsvExporter(tmp_path).export([{"a": "1", "b": "2"}], "x")`
- Oczekiwane: zwraca `tmp_path / "x.csv"`, a `read_text(encoding="utf-8") == "a,b\n1,2\n"`
```

Kod testu:

```python
from app.exporters import CsvExporter


def test_csv_export_example(tmp_path):
    path = CsvExporter(tmp_path).export([{"a": "1", "b": "2"}], "x")

    assert path == tmp_path / "x.csv"
    # Dokładna treść, łącznie z końcami linii: "\r\n" to też błąd.
    assert path.read_text(encoding="utf-8") == "a,b\n1,2\n"
```
