#!/usr/bin/env bash
# Phase 5: planner masking + rankings (+ randomised-planner sanity), anchor relocation, rescore ablation, summary.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD SDXAI_VERSION=${SDXAI_VERSION:-mini}
source scripts_xai/disk_guard.sh
F="grep -v -i warn"
python -m sdxai.interventions.planner_mask 2>&1 | $F | grep -v "it/s\]"
python -m sdxai.interventions.anchor_relocate 2>&1 | $F | grep -v "it/s\]"
python -m sdxai.interventions.rescore_ablate 2>&1 | $F | grep -v "it/s\]"
python -m sdxai.interventions.summary_p5 2>&1 | $F
