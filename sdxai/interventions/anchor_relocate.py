"""Phase 5 (3): geometric vs appearance reliance for pedestrian agents in the planner's top-50.

For every selected pedestrian agent (argmax = pedestrian, score > 0.3):
  into     : anchor moved 3 m laterally TOWARDS the ego path (x in the lidar frame), feature fixed
  out      : anchor moved 3 m AWAY from the ego path, feature fixed
  swap     : anchor fixed, instance feature replaced by the feature of the farthest other
             pedestrian slot of the same frame (> 30 m if available)
The moved anchor is used everywhere the planner reads det geometry: anchor embedding (queries
and gnn keys) and the collision-aware rescore's agent boxes (decode_plan anchor_override).
Motion-mode anchors depend only on class and yaw, so they are unchanged.
Reliance: mean |dl2| of relocation (geometry) vs swap (appearance), paired per agent.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sdxai import planner as PL
from sdxai.instances import ego_path, project_to_path
from sdxai.interventions.planner_mask import effects
from sdxai import metrics as MT
from sdxai.replay import Replay
from sdxai.runner import PED

SHIFT = 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=os.environ.get("SDXAI_VERSION", "mini"))
    ap.add_argument("--out", default="work_dirs/xai/interventions")
    ap.add_argument("--frame-stride", type=int, default=1, help="evaluate every k-th val frame (all frames still stream through the model, so temporal state is exact)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rp = Replay(version=a.version)
    rows = []
    idx = list(range(0, len(rp.r.dataset), a.frame_stride))
    for fr in tqdm(rp.frames(indices=set(idx)), total=len(idx), desc="p5 relocate"):
        final = fr.base["final"].float().cpu().numpy()
        fr._base_col = MT.plan_collision(final, fr.gt_traj, fr.fut_boxes) if fr.gt_mask.all() else {}
        fr._base_clear = MT.min_clearance(final, fr.fut_boxes, fr.fut_labels, "pedestrian")
        prob = fr.det["classification"][-1][0].float().sigmoid()
        anc = fr.det["prediction"][-1][0].float()
        feats = fr.det["instance_feature"][0]
        is_ped = (prob.argmax(-1) == PED) & (prob.max(-1).values > 0.3)
        sel = fr.base_outs["sel_idx"][0].tolist()
        peds_all = torch.where(is_ped)[0].tolist()
        path = ego_path(fr.info["gt_ego_fut_trajs"], fr.info["gt_ego_fut_masks"])
        for j in sel:
            if not bool(is_ped[j]):
                continue
            p = anc[j, :2].cpu().numpy()
            lat, s = project_to_path(p[None], path)
            # direction towards the path: nearest path point along x (lateral in lidar frame)
            dirx = -np.sign(p[0] - np.interp(p[1], path[:, 1], path[:, 0])) or 1.0
            row = {"frame": fr.i, "scene_token": fr.info["scene_token"], "slot": j, "x": float(p[0]), "y": float(p[1]),
                   "dist": float(np.linalg.norm(p)), "lateral_to_path": float(lat[0]), "s_along_path": float(s[0]),
                   "score": float(prob[j].max())}
            for name, dx in [("into", SHIFT * dirx), ("out", -SHIFT * dirx)]:
                a_new = anc[j].clone()
                a_new[0] += float(dx)
                dec = fr.run(PL.Intervention(replace_anchor={j: a_new}), anchor_override={j: a_new})
                row.update({f"{name}_{k}": v for k, v in effects(fr, dec).items()})
            others = [o for o in peds_all if o != j]
            if others:
                d = anc[others, :2].norm(dim=-1)
                far = others[int(d.argmax())]
                row["swap_src_dist"] = float(d.max())
                dec = fr.run(PL.Intervention(replace_feature={j: feats[far]}))
                row.update({f"swap_{k}": v for k, v in effects(fr, dec).items()})
            rows.append(row)
    rp.close()
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(a.out, "anchor_relocate.parquet"), index=False)
    print(df[[c for c in df.columns if c.endswith("dl2_avg")]].describe().round(4))


if __name__ == "__main__":
    main()
