#!/usr/bin/env bash
# Phase 7: Phase-4b pedestrian concepts, statistics + report.md
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD SDXAI_VERSION=${SDXAI_VERSION:-mini}
source scripts_xai/disk_guard.sh
F="grep -v -i warn"
# Phase 4b (fine-grained pedestrian concepts) needs Phase-4 TCAV gradients and Phase-5 planner rows
[ -f work_dirs/xai/concepts/ped_concepts/summary.json ] || python -m sdxai.concepts.ped_concepts 2>&1 | $F
python -m sdxai.report.make_report 2>&1 | $F
