#!/usr/bin/env bash
# arXiv source bundle: tex + generated numbers/tables + figures + compiled .bbl (arXiv does not run bibtex itself).
# Refuses while results are still mini or the draft switch is on.
set -e
cd "$(dirname "$0")"
./build.sh
grep -q "^MINI" status.txt && { echo "STOP: some results are still nuScenes-mini (see status.txt)"; exit 1; }
grep -q '^\\drafttrue' main.tex && { echo "STOP: set \\draftfalse in main.tex first"; exit 1; }
rm -rf arxiv && mkdir -p arxiv/tables arxiv/figures
cp main.tex numbers.tex pending.tex main.bbl arxiv/
# only the tables and figures main.tex actually uses (no stale or orphaned files)
for f in $(grep -o '{\(tables\|figures\)/[^}]*}' main.tex | tr -d '{}' | sort -u); do cp "$f" "arxiv/$f"; done
tar -czf arxiv_source.tar.gz -C arxiv .
echo "wrote arxiv_source.tar.gz"
