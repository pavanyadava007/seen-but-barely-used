#!/usr/bin/env bash
# Phase 4: class directions, probes, SAEs (sdxai TopK + Overcomplete toolbox), downstream + steering, TCAV.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD SDXAI_VERSION=${SDXAI_VERSION:-mini}
source scripts_xai/disk_guard.sh
F="grep -v -i warn"
python -m sdxai.concepts.class_directions 2>&1 | $F
python -m sdxai.concepts.probes 2>&1 | $F | tail -25
python -m sdxai.concepts.sae 2>&1 | $F
python -m sdxai.concepts.overcomplete_suite 2>&1 | $F | grep -v "^Epoch"
python -m sdxai.concepts.overcomplete_suite --l1-sweep 2>&1 | $F | grep -v "^Epoch"
python -m sdxai.concepts.sae_downstream 2>&1 | $F | grep -v "it/s\]"
python -m sdxai.concepts.tcav 2>&1 | $F | grep -v "it/s\]"
