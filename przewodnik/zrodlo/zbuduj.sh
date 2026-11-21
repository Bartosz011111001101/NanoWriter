#!/usr/bin/env bash
# Składa przewodnik do PDF: Markdown → HTML (pandoc) → PDF (Chrome bez okna).
# Uruchom z katalogu głównego repo w Git Bash: bash przewodnik/zrodlo/zbuduj.sh
# Obrazki muszą istnieć w przewodnik/wyniki/ (uv run python -m przewodnik.zrob_wszystko).
set -euo pipefail
cd "$(dirname "$0")/.."
HTML="$(mktemp -d)/przewodnik.html"
"/c/Program Files/Pandoc/pandoc" \
  zrodlo/00-wstep.md zrodlo/01-dane-tokeny.md zrodlo/02-model-trening.md \
  zrodlo/03-skrypty-testy-pojecia.md zrodlo/04-teoria.md zrodlo/05-rozmowca.md \
  -f markdown -t html5 -s --toc --toc-depth=2 --css zrodlo/styl.css \
  --embed-resources --resource-path=. --metadata pagetitle="Przewodnik NanoWriter" \
  -o "$HTML"
OUT="$(cd .. && pwd -W)/Przewodnik-NanoWriter-v2.pdf"
"/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --disable-gpu \
  --no-pdf-header-footer --print-to-pdf="$OUT" "file:///$(cygpath -m "$HTML")" 2>/dev/null
echo "Zapisano: $OUT"
