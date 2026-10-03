"""Planning / detection metrics used by Phases 4-6.

Planning (ego trajectory in the current lidar frame, 6 x 0.5 s, cumulative positions):
  plan_l2_at      : L2 between two trajectories at 1 / 2 / 3 s (indices 1, 3, 5)
  collision_flags : upstream box-collision test (planning_eval.check_collision, UniAD 0.5 m offset)
                    of the ego box along the plan vs GT future boxes, per timestep; `col_any`
                    counts only timesteps where the GT ego trajectory itself does not collide
                    (upstream obj_box_col definition)
  min_clearance   : min polygon distance ego box <-> GT boxes of a class over the horizon (m)
  ttc_plan        : first timestep (s) at which the plan collides with a GT box, 3.5 if none
                    (right-censored at the 3 s horizon)
Detection:
  ap_center       : nuScenes-style AP for one class, centre-distance matching at thresholds
                    {0.5, 1, 2, 4} m, precision interpolated on 101 recall points, AP = mean
                    over recall > 0.1 of max(prec - 0.1, 0) / 0.9 (nuScenes detection eval).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from shapely.geometry import Polygon

from projects.mmdet3d_plugin.datasets.evaluation.planning.planning_eval import check_collision, get_yaw
from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners

EGO_L, EGO_W, EGO_H = 4.084, 1.85, 1.56
T_IDX = {"1s": 1, "2s": 3, "3s": 5}
CENSOR = 3.5


def plan_l2_at(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    d = np.linalg.norm(np.asarray(a)[:, :2] - np.asarray(b)[:, :2], axis=-1)
    out = {f"dl2_{k}": float(d[i]) for k, i in T_IDX.items()}
    out["dl2_avg"] = float(np.mean([d[1], d[3], d[5]]))
    return out


def _ego_boxes(traj: torch.Tensor) -> torch.Tensor:
    n = traj.shape[0]
    box = traj.new_zeros((n, 7))
    box[:, :2] = traj
    box[:, 3:6] = box.new_tensor([EGO_L, EGO_W, EGO_H])
    box[:, 6] = get_yaw(traj)
    return box


def collision_flags(traj: np.ndarray, fut_boxes: List[np.ndarray]) -> np.ndarray:
    traj_t = torch.as_tensor(np.asarray(traj, np.float32))
    ego = _ego_boxes(traj_t)
    T = min(len(fut_boxes), traj_t.shape[0])
    out = np.zeros(traj_t.shape[0], bool)
    for t in range(T):
        boxes = torch.as_tensor(np.asarray(fut_boxes[t], np.float32)[:, :7])
        out[t] = bool(check_collision(ego[t].clone(), boxes))
    return out


def plan_collision(traj: np.ndarray, gt_traj: np.ndarray, fut_boxes: List[np.ndarray]) -> Dict[str, object]:
    c = collision_flags(traj, fut_boxes)
    g = collision_flags(gt_traj, fut_boxes)
    eff = c & ~g
    first = np.where(eff)[0]
    return {"col_any": bool(eff.any()), "col_steps": eff.astype(int).tolist(),
            "ttc_plan": float((first[0] + 1) * 0.5) if len(first) else CENSOR}


def min_clearance(traj: np.ndarray, fut_boxes: List[np.ndarray], fut_labels: Optional[List[np.ndarray]] = None,
                  label: Optional[str] = None, offset: float = 0.5) -> float:
    traj_t = torch.as_tensor(np.asarray(traj, np.float32))
    ego = _ego_boxes(traj_t).numpy()
    ego[:, 0] += offset * np.cos(ego[:, 6])
    ego[:, 1] += offset * np.sin(ego[:, 6])
    best = np.inf
    T = min(len(fut_boxes), len(ego))
    for t in range(T):
        boxes = np.asarray(fut_boxes[t])[:, :7]
        if fut_labels is not None and label is not None:
            boxes = boxes[np.asarray(fut_labels[t]) == label]
        if len(boxes) == 0:
            continue
        ep = Polygon(box3d_to_corners(ego[t:t + 1])[0, [0, 3, 7, 4], :2])
        near = np.linalg.norm(boxes[:, :2] - ego[t, :2], axis=1) < 30
        for c in box3d_to_corners(boxes[near])[:, [0, 3, 7, 4], :2]:
            best = min(best, ep.distance(Polygon(c)))
    return float(best)


def ap_center(pred_xy: np.ndarray, pred_score: np.ndarray, pred_frame: np.ndarray,
              gt_xy: np.ndarray, gt_frame: np.ndarray, thresholds: Sequence[float] = (0.5, 1.0, 2.0, 4.0)) -> Dict[str, float]:
    order = np.argsort(-pred_score)
    pred_xy, pred_frame = pred_xy[order], pred_frame[order]
    n_gt = len(gt_xy)
    out = {}
    gt_by_frame = {}
    for i, f in enumerate(gt_frame):
        gt_by_frame.setdefault(int(f), []).append(i)
    for thr in thresholds:
        taken = np.zeros(n_gt, bool)
        tp = np.zeros(len(pred_xy))
        for j, (xy, f) in enumerate(zip(pred_xy, pred_frame)):
            cand = gt_by_frame.get(int(f), [])
            best, bi = np.inf, -1
            for gi in cand:
                if taken[gi]:
                    continue
                d = np.linalg.norm(gt_xy[gi] - xy)
                if d < best:
                    best, bi = d, gi
            if bi >= 0 and best <= thr:
                taken[bi] = True
                tp[j] = 1
        ctp = np.cumsum(tp)
        cfp = np.cumsum(1 - tp)
        rec = ctp / max(n_gt, 1)
        prec = ctp / np.maximum(ctp + cfp, 1e-9)
        rgrid = np.linspace(0, 1, 101)
        p_interp = np.zeros_like(rgrid)
        for k, r in enumerate(rgrid):
            m = prec[rec >= r]
            p_interp[k] = m.max() if len(m) else 0.0
        p = p_interp[rgrid > 0.1 + 1e-9]
        out[f"ap@{thr}"] = float(np.mean(np.clip(p - 0.1, 0, None)) / 0.9)
    out["ap"] = float(np.mean([out[f"ap@{t}"] for t in thresholds]))
    return out
