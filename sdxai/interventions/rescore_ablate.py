"""Phase 5 (4): drop one pedestrian's predicted trajectories from the collision-aware rescore.

HierarchicalPlanningDecoder.rescore penalises (-999) every ego mode of the chosen command whose
box path collides with the top-1 predicted trajectory of any agent with confidence >= 0.5.
For every pedestrian agent with confidence >= 0.5 (i.e. one the rescore actually considers) its
trajectories are removed from the rescore only (confidence set to 0 for the rescore; the planner
network itself is untouched). Reported: # ego modes whose penalty changes, selection change,
dl2 of the final plan, and whether that pedestrian was the *only* reason for a penalty.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from sdxai.interventions.planner_mask import effects
from sdxai import metrics as MT
from sdxai.replay import Replay
from sdxai.runner import PED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=os.environ.get("SDXAI_VERSION", "mini"))
    ap.add_argument("--out", default="work_dirs/xai/interventions")
    ap.add_argument("--frame-stride", type=int, default=1, help="evaluate every k-th val frame (all frames still stream through the model, so temporal state is exact)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rp = Replay(version=a.version)
    rows, frames = [], []
    idx = list(range(0, len(rp.r.dataset), a.frame_stride))
    for fr in tqdm(rp.frames(indices=set(idx)), total=len(idx), desc="p5 rescore"):
        final = fr.base["final"].float().cpu().numpy()
        fr._base_col = MT.plan_collision(final, fr.gt_traj, fr.fut_boxes) if fr.gt_mask.all() else {}
        fr._base_clear = MT.min_clearance(final, fr.fut_boxes, fr.fut_labels, "pedestrian")
        base_pen = (fr.base["plan_cls_full"][fr.base["cmd"]] < -500).cpu().numpy()
        rerun = fr.run(None)  # sanity: a no-op rerun reproduces the base plan exactly
        prob = fr.det["classification"][-1][0].float().sigmoid()
        conf = prob.max(-1).values
        peds = ((prob.argmax(-1) == PED) & (conf >= 0.5)).nonzero().flatten().tolist()
        # all-agents-off reference: which penalties are due to any agent at all
        from sdxai import planner as PL
        h = rp.head.motion_plan_head
        dec_off = PL.decode_plan(h, fr.det, fr.base_outs, fr.metas, use_rescore=False)
        frames.append({"frame": fr.i, "base_reproduced": bool((rerun["final"] == fr.base["final"]).all()),
                       "n_penalised_modes": int(base_pen.sum()), "n_peds_considered": len(peds),
                       "rescore_changes_selection": int(dec_off["mode_idx"] != fr.base["mode_idx"])})
        for j in peds:
            dec = fr.run(None, rescore_drop=[j])
            pen = (dec["plan_cls_full"][dec["cmd"]] < -500).cpu().numpy()
            rows.append({"frame": fr.i, "scene_token": fr.info["scene_token"], "slot": j, "conf": float(conf[j]),
                         "dist": float(fr.det["prediction"][-1][0, j, :2].norm()),
                         "n_modes_penalty_changed": int((pen != base_pen).sum()),
                         "penalty_removed": int((base_pen & ~pen).sum()),
                         **{f"drop_{k}": v for k, v in effects(fr, dec).items()}})
    rp.close()
    df = pd.DataFrame(rows)
    fdf = pd.DataFrame(frames)
    df.to_parquet(os.path.join(a.out, "rescore_ablate.parquet"), index=False)
    fdf.to_parquet(os.path.join(a.out, "rescore_frames.parquet"), index=False)
    print(fdf.describe().round(3))
    if len(df):
        print(df[["n_modes_penalty_changed", "drop_mode_changed", "drop_dl2_avg"]].describe().round(4))


if __name__ == "__main__":
    main()
