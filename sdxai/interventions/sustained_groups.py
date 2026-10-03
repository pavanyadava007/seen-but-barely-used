"""Phase 5b: complete planner-side removal of agents, answering which routes carry pedestrian information.

The Phase-5 masking removes one agent's key from the planner attention of the current frame only. Two
planner-side routes stay open: (i) the collision-aware rescore, which penalises modes that collide with the
agent's predicted motion, and (ii) the planner's temporal queue, whose ego history holds post-planner ego
features of earlier frames that attended to the agent unmasked. This module streams the whole validation
split once (perception once per frame) and keeps one planner queue per sustained condition:

  base                 unmodified planner (its queue is the reference stream)
  {peds,veh,agents}_cur      group masked from current-frame attention only (base queue)
  {peds,veh,agents}_cur_rs   ... and removed from the rescore (base queue)
  {peds,veh,agents}_full     masked from attention and rescore in EVERY frame, masked pass committed to its
                             own queue (so the ego history never saw the group)

Groups: detections with max score > 0.3 and argmax class pedestrian / vehicle (car, truck, bus, trailer,
construction vehicle) / any. For every GT-matched corridor pedestrian (critical_w8 true positive) the single-agent
2x2 on the base queue is added: attention only, rescore only, both.

Per (frame, condition): Delta-L2 to the base plan at 1/2/3 s and averaged, longitudinal and lateral change at
3 s, mode switch, plan-vs-GT-future collision, minimum clearance to GT future pedestrians.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sdxai import metrics as MT
from sdxai import planner as PL
from sdxai.replay import Replay
from sdxai.runner import _QUEUE_KEYS, _clone, CLASS_NAMES, reset_state

VEH = [CLASS_NAMES.index(n) for n in ("car", "truck", "bus", "trailer", "construction_vehicle")]
PED = CLASS_NAMES.index("pedestrian")
GROUPS = ("peds", "veh", "agents")


def q_snap(head):
    return {k: _clone(getattr(head.instance_queue, k)) for k in _QUEUE_KEYS}


def q_load(head, s):
    if s is None:
        head.instance_queue.reset()
        return
    for k, v in s.items():
        setattr(head.instance_queue, k, _clone(v))


def group_idx(prob, g, thr=0.3):
    det = prob.max(-1).values > thr
    cls = prob.argmax(-1)
    sel = det if g == "agents" else det & (cls == PED if g == "peds" else torch.isin(cls, torch.tensor(VEH, device=cls.device)))
    return torch.where(sel)[0].tolist()


def effect(final, base, fr, base_col, base_clear):
    out = MT.plan_l2_at(final, base)
    out["d_long_3s"] = float(final[-1, 1] - base[-1, 1])  # lidar frame: y forward
    out["d_lat_3s"] = float(final[-1, 0] - base[-1, 0])
    if fr["gt_mask"].all() and len(fr["fut_boxes"]) >= 6:
        c = MT.plan_collision(final, fr["gt_traj"], fr["fut_boxes"])
        out["col"] = int(c["col_any"])
        out["col_flip"] = int(c["col_any"] != base_col)
        mc = MT.min_clearance(final, fr["fut_boxes"], fr["fut_labels"], "pedestrian")
        out["d_min_clear_ped"] = (mc - base_clear) if np.isfinite(mc) and np.isfinite(base_clear) else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=os.environ.get("SDXAI_VERSION", "mini"))
    ap.add_argument("--instances", default="work_dirs/xai/instances/instances.parquet")
    ap.add_argument("--out", default="work_dirs/xai/interventions")
    ap.add_argument("--limit", type=int, default=-1)
    a = ap.parse_args()
    inst = pd.read_parquet(a.instances)
    crit = inst[(inst.split == "val") & (inst.kind == "tp") & (inst.gt_class == "pedestrian") & inst.critical_w8.astype(bool)]
    crit_slots = {int(f): d.pred_idx.astype(int).tolist() for f, d in crit.groupby("frame")}

    rp = Replay(version=a.version, capture_det=False)
    r, h = rp.r, rp.head.motion_plan_head
    eager = rp.eager
    reset_state(r.model)
    queues = {"base": None, **{f"{g}_full": None for g in GROUPS}}
    rows, single = [], []
    n = len(r.dataset) if a.limit < 0 else min(a.limit, len(r.dataset))
    peak = 0
    for i in tqdm(range(n), desc="p5b sustained"):
        if i % 500 == 0:
            peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
            print(f"frame {i}: peak GPU {peak:.1f} GiB", flush=True)
        data = r.batch(i)
        metas = r.metas_of(data)
        info = r.dataset.data_infos[i]
        with torch.no_grad():
            fm = r.features(data["img"])
            det, mp = r.perception(fm, metas)
            prob = det["classification"][-1][0].float().sigmoid()
            gidx = {g: group_idx(prob, g) for g in GROUPS}
            gt, gmask, fb, fl = rp._fut(i)
            fr = {"gt_traj": gt, "gt_mask": gmask, "fut_boxes": fb, "fut_labels": fl}

            def run(P, iv=None, drop=None):
                rp.recorder.clear()  # the eager-attention recorder keeps maps until cleared
                outs = PL.layers(h, P, iv, eager)
                return outs, PL.decode_plan(h, det, outs, metas, rescore_drop=drop or None)

            def prep():
                return PL.prepare(h, det, mp, fm, metas, rp.head.det_head.anchor_encoder,
                                  rp.head.det_head.instance_bank.mask, rp.head.det_head.instance_bank.anchor_handler)

            # base stream
            q_load(h, queues["base"])
            P = prep()
            outs_b, dec_b = run(P)
            base = dec_b["final"].float().cpu().numpy()
            col_ok = gmask.all() and len(fb) >= 6
            base_col = int(MT.plan_collision(base, gt, fb)["col_any"]) if col_ok else 0
            base_clear = MT.min_clearance(base, fb, fl, "pedestrian") if col_ok else np.nan
            common = {"frame": i, "scene_token": info["scene_token"], "sample_token": info["token"],
                      "n_peds": len(gidx["peds"]), "n_veh": len(gidx["veh"]), "n_agents": len(gidx["agents"]),
                      "has_corridor_ped": i in crit_slots, "base_col": base_col if col_ok else np.nan,
                      "base_mode": int(dec_b["mode_idx"])}
            res = {}
            for g in GROUPS:  # current-frame variants on the base queue
                if gidx[g]:
                    iv = PL.Intervention(mask_keys=gidx[g])
                    res[f"{g}_cur"] = run(P, iv)[1]
                    res[f"{g}_cur_rs"] = run(P, iv, gidx[g])[1]
            for j in crit_slots.get(i, []):  # single corridor pedestrian 2x2
                iv = PL.Intervention(mask_keys=[j])
                for tag, (ivx, drop) in {"attn": (iv, None), "rescore": (None, [j]), "both": (iv, [j])}.items():
                    d = run(P, ivx, drop)[1]
                    e = effect(d["final"].float().cpu().numpy(), base, fr, base_col, base_clear)
                    single.append({**common, "slot": j, "cond": tag, "mode_changed": int(d["mode_idx"] != dec_b["mode_idx"]), **e})
            PL.commit(h, P, outs_b, det, metas)
            queues["base"] = q_snap(h)
            for g in GROUPS:  # sustained streams, each with its own queue
                q_load(h, queues[f"{g}_full"])
                Pg = prep()
                iv = PL.Intervention(mask_keys=gidx[g]) if gidx[g] else None
                outs_g, dec_g = run(Pg, iv, gidx[g])
                res[f"{g}_full"] = dec_g
                PL.commit(h, Pg, outs_g, det, metas)
                queues[f"{g}_full"] = q_snap(h)
            q_load(h, queues["base"])
            for cond, d in res.items():
                e = effect(d["final"].float().cpu().numpy(), base, fr, base_col, base_clear)
                rows.append({**common, "cond": cond, "mode_changed": int(d["mode_idx"] != dec_b["mode_idx"]), **e})
    rp.close()
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(a.out, "sustained_groups.parquet"), index=False)
    pd.DataFrame(single).to_parquet(os.path.join(a.out, "single_corridor_2x2.parquet"), index=False)
    df = pd.DataFrame(rows)
    print(df.groupby("cond")[["dl2_avg", "mode_changed", "col", "d_min_clear_ped"]].mean().round(4).to_string())
    if single:
        print(pd.DataFrame(single).groupby("cond")[["dl2_avg", "mode_changed"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
