#!/usr/bin/env bash
# Phase 1: equivalence tests, then sequential extraction of val (full) and train (light) splits.
# Usage: bash scripts_xai/p1_extract.sh [--version mini|trainval]
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION=mini
[ "${1:-}" = "--version" ] && VERSION=$2
export PYTHONPATH=$PWD SDXAI_VERSION=$VERSION
OUT=work_dirs/xai/extract
mkdir -p $OUT
source scripts_xai/disk_guard.sh
python -m pytest -q -p no:warnings tests/test_phase1_equivalence.py 2>&1 | tee $OUT/p1_tests.log
python -m sdxai.extract --version $VERSION --split test  --out $OUT 2>&1 | grep -v -i warn | tail -12
python -m sdxai.extract --version $VERSION --split train --out $OUT 2>&1 | grep -v -i warn | tail -12
python - <<'PY'
import glob, json, os
import h5py, numpy as np
d = []
for p in sorted(glob.glob("work_dirs/xai/extract/test_*.h5")):
    with h5py.File(p) as f:
        for k in f:
            d.append(np.linalg.norm(f[k]["plan/final"][()] - f[k]["plan/final_eager"][()], axis=-1).max())
tot = sum(os.path.getsize(p) for p in glob.glob("work_dirs/xai/extract/test_*.h5"))
acc = {"eager_vs_flash_plan_maxdiff_m": float(max(d)), "eager_vs_flash_plan_meandiff_m": float(np.mean(d)),
       "val_h5_bytes": tot, "num_val_frames": len(d),
       "pass_eager": bool(max(d) < 1e-3), "pass_size": bool(tot < 2 * 1024**3)}
json.dump(acc, open("work_dirs/xai/extract/p1_acceptance.json", "w"), indent=2)
print(json.dumps(acc, indent=2))
PY
