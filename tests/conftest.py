import sys
from pathlib import Path

# Pakiet `przewodnik` leży w korzeniu repo i nie jest zainstalowany (nie ma go w wheel),
# a konsola `pytest` sama nie dodaje korzenia do `sys.path`.
_KORZEN = str(Path(__file__).resolve().parents[1])
if _KORZEN not in sys.path:
    sys.path.insert(0, _KORZEN)
