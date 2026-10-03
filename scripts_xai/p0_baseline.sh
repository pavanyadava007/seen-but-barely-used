#!/usr/bin/env bash
# Phase 0: upstream test on the stage-2 checkpoint; metrics -> work_dirs/xai/p0_baseline/*.json
# Two identical runs quantify run-to-run nondeterminism (the deformable-aggregation CUDA kernel
# accumulates with atomicAdd). Upstream's pure-PyTorch deformable path cannot run the full
# stage-2 model (InstanceQueue assumes the DAF feature-map format), so op correctness is
# checked at module level in tests/test_daf_equivalence.py instead.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD SDXAI_VERSION=${SDXAI_VERSION:-mini}
CFG=sdxai/configs/sd_small_stage2_xai.py
OUT=work_dirs/xai/p0_baseline
mkdir -p $OUT
source scripts_xai/disk_guard.sh

for tag in run1 run2; do
  [ -s $OUT/metrics_$tag.json ] && { echo "skip $tag (done)"; continue; }
  python -m sdxai.tools.test_seqeval $CFG ckpt/sparsedrive_stage2.pth --deterministic --eval bbox \
    --cfg-options work_dir=$OUT/$tag > $OUT/test_$tag.log 2>&1
  python -m sdxai.p0_metrics $OUT/test_$tag.log $OUT/metrics_$tag.json
done
python -m sdxai.p0_metrics --compare $OUT/metrics_run1.json $OUT/metrics_run2.json $OUT/p0_summary.json
