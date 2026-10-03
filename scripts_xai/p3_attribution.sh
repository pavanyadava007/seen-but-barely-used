#!/usr/bin/env bash
# Phase 3: perception-level attribution (DFA, gradient, SmoothGrad, grad x feature, D-RISE),
# ROAD faithfulness, localisation, MPRT + target randomisation, then summary + figures.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD SDXAI_VERSION=${SDXAI_VERSION:-mini}
source scripts_xai/disk_guard.sh
OUT=work_dirs/xai/attribution
mkdir -p $OUT
if [ "${SKIP_RUN:-0}" != "1" ]; then
  python -m sdxai.attribution.run_p3 --out $OUT "$@" 2>&1 | grep -v -i warn | grep -v "it/s\]" > $OUT/run_p3.log
fi
python -m sdxai.attribution.summary_p3 --dir $OUT 2>&1 | grep -v -i warn
