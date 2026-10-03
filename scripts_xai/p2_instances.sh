#!/usr/bin/env bash
# Phase 2: instance table with GT matching, derived fields and strata.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD SDXAI_VERSION=${SDXAI_VERSION:-mini}
source scripts_xai/disk_guard.sh
python -m sdxai.instances --version $SDXAI_VERSION 2>&1 | grep -v -i warn
