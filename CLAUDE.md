# Instrukcje projektu

- Opis projektu, stack, komendy i układ repozytorium: `SKILLS/projekt/profil.md` (wczytany niżej).
- Agenci w `.claude/agents/`, ich skille w `SKILLS/<agent>_skill/SKILL.md`. Każdy agent czyta swój skill na starcie. Skille są ogólne, między projektami przenosi się je bez zmian.
- Wiedza o tym projekcie jest tylko w `SKILLS/projekt/`: `profil.md`, `konwencje.md`, `katalog-testow.md` i `pulapki/` (wymienne pliki według technologii).
- `Docs/` (plany, prompty, postęp, kontrakty) jest lokalny i poza gitem.

W tym projekcie główna sesja pełni rolę orkiestratora:

@.claude/agents/orchestrator.md

@SKILLS/orchestrator_skill/SKILL.md

@SKILLS/projekt/profil.md
