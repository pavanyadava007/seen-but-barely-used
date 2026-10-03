"""Phase 6: temporal ablation - how much pedestrian perception and planning rely on memory.

(A) history truncation (det instance bank AND planner instance queue): for every val frame t the
    full model is re-run from a reset state over frames max(scene_start, t-H) .. t, H in {0,1,2,3}
    (H = number of past frames available); "full" = the normal sequential stream, run twice
    (full, full_rerun) to measure the atomicAdd nondeterminism floor.
(B) det temp_gnn values zeroed (normal stream, eager attention on the det head's temp_gnn only):
    removes temporal cross-attention while the 600 carried instances remain.
(C) planner-only (normal stream, cached det outputs): planner temp_gnn keeps only the k most recent
    history frames (k in {0,1,2}) or all temporal keys are masked (current frame only).
Metrics at frame t: pedestrian recall (GT pedestrians, same filter as Phase 2, class-aware
Hungarian at 2 m on predictions with argmax = pedestrian and score > 0.3) split by visibility
(1-2 = occluded, 3-4 = visible) and distance; plan dl2 vs the full-history plan; L2 to GT.
Memory reliance = recall(full) - recall(ablated), per stratum, frame-cluster bootstrap CIs.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from sdxai import metrics as MT
from sdxai import planner as PL
from sdxai.eager_attention import use_eager_attention
from sdxai.instances import GT_RANGE
from sdxai.replay import Replay
from sdxai.runner import PED, SDRunner, reset_state

HISTORY = [0, 1, 2, 3]


def gt_ped_table(inst: pd.DataFrame) -> dict:
    g = inst[(inst.split == "val") & inst.gt_idx.notna() & (inst.gt_class == "pedestrian")]
    return {(int(f), int(i)): (int(v), float(d)) for f, i, v, d in zip(g.frame, g.gt_idx, g.visibility, g.gt_dist)}


def ped_recall_rows(frame, info, cls, anc, gtab, tag):
    prob = cls[0].float().sigmoid().cpu().numpy()
    a = anc[0].float().cpu().numpy()
    pidx = np.where((prob.argmax(-1) == PED) & (prob.max(-1) > 0.3))[0]
    names, boxes = info["gt_names"], info["gt_boxes"]
    pts = info["num_lidar_pts"] + info["num_radar_pts"]
    rng = np.linalg.norm(boxes[:, :2], axis=-1)
    gidx = np.where((names == "pedestrian") & (pts > 0) & (rng <= GT_RANGE))[0]
    hit = set()
    if len(pidx) and len(gidx):
        D = np.linalg.norm(a[pidx, None, :2] - boxes[None, gidx, :2], axis=-1)
        r, c = linear_sum_assignment(D)
        hit = {int(gidx[cc]) for rr, cc in zip(r, c) if D[rr, cc] <= 2.0}
    rows = []
    for g in gidx:
        vis, dist = gtab.get((frame, int(g)), (0, float(rng[g])))
        rows.append({"tag": tag, "frame": frame, "scene_token": info["scene_token"], "gt_idx": int(g),
                     "visibility": vis, "gt_dist": dist, "detected": int(int(g) in hit)})
    return rows


def plan_row(frame, info, final, ref_final, tag, gt_traj, gt_mask):
    r = {"tag": tag, "frame": frame, "scene_token": info["scene_token"]}
    if ref_final is not None:
        r.update(MT.plan_l2_at(final, ref_final))
    if gt_mask.all():
        r["l2_gt_avg"] = MT.plan_l2_at(final, gt_traj)["dl2_avg"]
    return r


def full_pass(r: SDRunner, tag: str, gtab, zero_det_temp: bool = False):
    """Normal sequential stream. Returns recall rows, finals per frame."""
    rec, finals = [], {}
    h = r.head
    ctx = None
    if zero_det_temp:
        names = [f"head.det_head.layers.{i}" for i, op in enumerate(h.det_head.operation_order) if op == "temp_gnn"]
        ctx = use_eager_attention(r.model, prefixes=names)
        eag = ctx.__enter__()
        for e in eag.values():
            e.zero_values = True
    try:
        reset_state(r.model)
        for i in tqdm(range(len(r.dataset)), desc=tag):
            data = r.batch(i)
            metas = r.metas_of(data)
            with torch.no_grad():
                fm = r.features(data["img"])
                det, mp = r.perception(fm, metas)
                P = PL.prepare(h.motion_plan_head, det, mp, fm, metas, h.det_head.anchor_encoder,
                               h.det_head.instance_bank.mask, h.det_head.instance_bank.anchor_handler)
                outs = PL.layers(h.motion_plan_head, P)
                dec = PL.decode_plan(h.motion_plan_head, det, outs, metas)
                PL.commit(h.motion_plan_head, P, outs, det, metas)
            info = r.dataset.data_infos[i]
            rec += ped_recall_rows(i, info, det["classification"][-1], det["prediction"][-1], gtab, tag)
            finals[i] = dec["final"].float().cpu().numpy()
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    return rec, finals


def truncated_pass(r: SDRunner, H: int, gtab, stride: int = 1):
    rec, finals = [], {}
    h = r.head
    infos = r.dataset.data_infos
    for t in tqdm(range(0, len(r.dataset), stride), desc=f"history{H}"):
        start = t
        while start > 0 and t - start < H and infos[start - 1]["scene_token"] == infos[t]["scene_token"]:
            start -= 1
        reset_state(r.model)
        for i in range(start, t + 1):
            data = r.batch(i)
            metas = r.metas_of(data)
            with torch.no_grad():
                fm = r.features(data["img"])
                det, mp = r.perception(fm, metas)
                P = PL.prepare(h.motion_plan_head, det, mp, fm, metas, h.det_head.anchor_encoder,
                               h.det_head.instance_bank.mask, h.det_head.instance_bank.anchor_handler)
                outs = PL.layers(h.motion_plan_head, P)
                if i == t:
                    dec = PL.decode_plan(h.motion_plan_head, det, outs, metas)
                PL.commit(h.motion_plan_head, P, outs, det, metas)
        rec += ped_recall_rows(t, infos[t], det["classification"][-1], det["prediction"][-1], gtab, f"history{H}")
        finals[t] = dec["final"].float().cpu().numpy()
    return rec, finals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=os.environ.get("SDXAI_VERSION", "mini"))
    ap.add_argument("--instances", default="work_dirs/xai/instances/instances.parquet")
    ap.add_argument("--out", default="work_dirs/xai/temporal")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="evaluate every k-th val frame; full passes still stream all frames")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    inst = pd.read_parquet(a.instances)
    gtab = gt_ped_table(inst)
    r = SDRunner(version=a.version)
    infos = r.dataset.data_infos
    rec, plans = [], []
    rec_f, fin_full = full_pass(r, "full", gtab)
    rec += rec_f
    runs = {"full_rerun": full_pass(r, "full_rerun", gtab),
            "det_temp_gnn_zero_values": full_pass(r, "det_temp_gnn_zero_values", gtab, zero_det_temp=True)}
    for H in HISTORY:
        runs[f"history{H}"] = truncated_pass(r, H, gtab, a.frame_stride)
    for tag, (rr, fin) in runs.items():
        rec += rr
        for t, f in fin.items():
            info = infos[t]
            gt = np.cumsum(info["gt_ego_fut_trajs"], axis=0)
            plans.append(plan_row(t, info, f, fin_full[t], tag, gt, info["gt_ego_fut_masks"].astype(bool)))
    for t, f in fin_full.items():
        info = infos[t]
        plans.append(plan_row(t, info, f, None, "full", np.cumsum(info["gt_ego_fut_trajs"], axis=0),
                              info["gt_ego_fut_masks"].astype(bool)))
    del r
    torch.cuda.empty_cache()

    # (C) planner-only temporal ablation on cached det outputs
    rp = Replay(version=a.version, capture_det=False)
    idx = list(range(0, len(rp.r.dataset), a.frame_stride))
    for fr in tqdm(rp.frames(indices=set(idx)), total=len(idx), desc="planner temporal"):
        base = fr.base["final"].float().cpu().numpy()
        for tag, iv in [("planner_keep0", PL.Intervention(temporal_keep=0)),
                        ("planner_keep1", PL.Intervention(temporal_keep=1)),
                        ("planner_keep2", PL.Intervention(temporal_keep=2)),
                        ("planner_zero_temp_gnn", PL.Intervention(zero_temp_gnn=True))]:
            d = fr.run(iv)
            plans.append(plan_row(fr.i, fr.info, d["final"].float().cpu().numpy(), base, tag, fr.gt_traj, fr.gt_mask))
    rp.close()
    R = pd.DataFrame(rec)
    Pl = pd.DataFrame(plans)
    R = R[R.frame % a.frame_stride == 0]  # identical frame set for every run -> paired comparisons
    Pl = Pl[Pl.frame % a.frame_stride == 0]
    R.to_parquet(os.path.join(a.out, "temporal_recall_rows.parquet"), index=False)
    Pl.to_parquet(os.path.join(a.out, "temporal_plan_rows.parquet"), index=False)
    R["vis_bin"] = np.where(R.visibility <= 2, "occluded(1-2)", "visible(3-4)")
    print(R.groupby(["tag", "vis_bin"]).detected.mean().unstack().round(4))
    print(Pl.groupby("tag")[["dl2_avg", "l2_gt_avg"]].mean().round(4))


if __name__ == "__main__":
    main()
