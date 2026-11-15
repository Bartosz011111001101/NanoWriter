# NanoWriter

Mały LLM budowany od zera. Python 3.13, uv, PyTorch (CUDA 12.8), pytest.

- Uruchamianie: `uv run python ...`, testy: `uv run pytest`, nowe biblioteki: `uv add <pakiet>`.
- Kod w `src/nanowriter/`, testy w `tests/`, skrypty w `scripts/`, dane w `data/` (poza gitem).
- Agenci w `.claude/agents/`, ich skille w `SKILLS/<agent>_skill/SKILL.md`. Każdy agent czyta swój skill na starcie. Skille są ogólne, między projektami przenosi się je bez zmian.
- Wiedza o tym projekcie jest w `SKILLS/projekt/`: `profil.md` (stack, komendy, moduły, etapy, bramki), `konwencje.md`, `katalog-testow.md` i `pulapki/` (wymienne pliki według technologii).
- `Docs/` (plany, prompty, postęp, kontrakty) jest lokalny i poza gitem.

W tym projekcie główna sesja pełni rolę orkiestratora:

@.claude/agents/orchestrator.md

@SKILLS/orchestrator_skill/SKILL.md
