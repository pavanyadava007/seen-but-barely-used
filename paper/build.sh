#!/usr/bin/env bash
# Rebuild the paper from the current result files: numbers/tables -> figures -> PDF.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
$PY build_numbers.py
$PY make_figures.py > /dev/null
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1 || { grep -A3 "^!" main.log; exit 1; }
echo "built main.pdf"; cat status.txt | grep -c MINI | xargs echo "results still from nuScenes-mini:"
