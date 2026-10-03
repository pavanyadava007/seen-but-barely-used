#!/usr/bin/env bash
# Phase 6: temporal ablation (history truncation, det temp_gnn, planner queue).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD SDXAI_VERSION=${SDXAI_VERSION:-mini}
source scripts_xai/disk_guard.sh
F="grep -v -i warn"
python -m sdxai.interventions.temporal_ablate 2>&1 | $F | grep -v "it/s\]"
