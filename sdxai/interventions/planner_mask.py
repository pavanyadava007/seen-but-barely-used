"""Phase 5 (1, 2, 5): instance-level interventions on the planner and explanation-predicts-behaviour.

Per val frame (replay, eager planner, cached det/map outputs; the planner-only reruns are
deterministic so every effect is relative to the same frame's unmodified plan):
  (a) mask     : each of the <= 50 selected det keys masked in every planner `gnn` layer
                 (key-validity mask, equivalent to zeroing its key/value row and setting
                 key_padding_mask)                                   -> every selected agent
  (b) mean     : the agent's instance feature replaced by the mean val pedestrian feature
                 (affects its own query and its key/value row)       -> pedestrian agents
  (c) drop     : removed from the top-50 candidate set, next instance shifts in -> pedestrians
Effects vs the unmodified selected trajectory: dl2 at 1/2/3 s, d ttc_plan / d min clearance to
GT pedestrians (from GT future boxes), collision-flag flip, d entropy of the command's mode
distribution (softmax over its 6 raw logits), d rescore penalty (# modes penalised by rescore),
mode change.
Rankings of the <= 50 agents (predict mask-effect (a)):
  attn    : ego-row attention mass summed over the planner gnn layers (mean over heads)
  grad    : ||d selected plan / d agent feature||_F
  physics : 1 / ttc_est from the predicted box and velocity and the current ego velocity
            (ego_status, no future information)
  random  : uniform random
Sanity (5): the whole analysis repeated with planner weights randomised (all params of
motion_plan_head redrawn N(mean, std)); Phase 3 maps depend only on the image backbone and det
head, which this does not touch (checked by comparing det outputs).
"""
from __future__ import annotations

import argparse
import pickle
import glob
import contextlib
import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sdxai import metrics as MT
from sdxai import planner as PL
from sdxai.replay import Replay, ego_attention_mass
from sdxai.runner import CLASS_NAMES, PED


def mean_ped_feature(inst: pd.DataFrame) -> torch.Tensor:
    from sdxai.concepts.data import gather
    d = inst[(inst.split == "val") & (inst.kind == "tp") & (inst.gt_class == "pedestrian")].reset_index(drop=True)
    d["pred_idx"] = d.pred_idx.astype(int)
    return torch.from_numpy(gather(d, 6).mean(0)).cuda()


def entropy(logits: torch.Tensor) -> float:
    p = torch.softmax(logits.float(), -1)
    return float(-(p * torch.log(p + 1e-12)).sum())


def rescore_penalty(dec) -> int:
    return int((dec["plan_cls_full"][dec["cmd"]] < -500).sum())


def effects(fr, dec) -> dict:
    final = dec["final"].float().cpu().numpy()
    base = fr.base["final"].float().cpu().numpy()
    out = MT.plan_l2_at(final, base)
    out["mode_changed"] = int(dec["mode_idx"] != fr.base["mode_idx"])
    out["d_entropy"] = entropy(dec["plan_cls_cmd_raw"]) - entropy(fr.base["plan_cls_cmd_raw"])
    out["d_rescore_penalty"] = rescore_penalty(dec) - rescore_penalty(fr.base)
    if fr.gt_mask.all() and len(fr.fut_boxes) >= 6:
        c1 = MT.plan_collision(final, fr.gt_traj, fr.fut_boxes)
        out["col_flip"] = int(c1["col_any"] != fr._base_col["col_any"])
        out["d_ttc_plan"] = c1["ttc_plan"] - fr._base_col["ttc_plan"]
        mc = MT.min_clearance(final, fr.fut_boxes, fr.fut_labels, "pedestrian")
        out["d_min_clear_ped"] = (mc - fr._base_clear) if np.isfinite(mc) and np.isfinite(fr._base_clear) else np.nan
    return out


def physics_scores(fr, sel: np.ndarray) -> np.ndarray:
    anc = fr.det["prediction"][-1][0, sel].float().cpu().numpy()
    p = anc[:, :2]
    v = anc[:, 8:10]
    es = fr.metas["ego_status"][0].float().cpu().numpy()
    v_ego = np.array([-es[7], es[6]])  # ego frame (x fwd, y left) -> lidar frame (x right, y fwd)
    rng = np.linalg.norm(p, axis=1)
    closing = -((p * (v - v_ego[None])).sum(1)) / np.maximum(rng, 1e-6)
    ttc = np.where(closing > 1e-3, rng / np.maximum(closing, 1e-3), 20.0)
    return 1.0 / np.clip(ttc, 0.05, 20.0)


@contextlib.contextmanager
def randomised_planner(model, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    saved = []
    try:
        for n, p in model.head.motion_plan_head.named_parameters():
            if p.numel() > 1 and p.requires_grad:
                saved.append((p, p.data.clone()))
                p.data.copy_(torch.randn(p.shape, generator=g, device=p.device) * p.data.float().std() + p.data.float().mean())
        yield
    finally:
        for p, d in saved:
            p.data.copy_(d)


def run(tag: str, inst: pd.DataFrame, version: str, randomise: bool, seed: int, ped_prob_thr: float = 0.3,
        stride: int = 1, ckpt_dir: str = None, ckpt_every: int = 50):
    """ckpt_dir: rows are checkpointed every ckpt_every frames; a restarted run reloads them and skips those
    frames (the trainval run spans the machine's 8 h power-off)."""
    rp = Replay(version=version)
    mean_ped = mean_ped_feature(inst)
    agent_rows, done = [], set()
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        for f in sorted(glob.glob(os.path.join(ckpt_dir, "chunk_*.pkl"))):
            ch = pickle.load(open(f, "rb"))
            agent_rows += ch["rows"]; done |= set(ch["frames"])
    pend_frames, pend_start = [], len(agent_rows)

    def flush():
        nonlocal pend_frames, pend_start
        if ckpt_dir and pend_frames:
            f = os.path.join(ckpt_dir, f"chunk_{pend_frames[0]:06d}.pkl")
            pickle.dump({"rows": agent_rows[pend_start:], "frames": pend_frames}, open(f + ".tmp", "wb"))
            os.replace(f + ".tmp", f)
        pend_frames, pend_start = [], len(agent_rows)
    ctx = randomised_planner(rp.r.model, seed) if randomise else contextlib.nullcontext()
    with ctx:
        idx = list(range(0, len(rp.r.dataset), stride))
        for fr in tqdm(rp.frames(indices=set(idx)), total=len(idx), desc=f"p5 {tag}"):
            if fr.i in done:
                continue
            rng = np.random.RandomState(seed + fr.i)  # per frame: a resumed run gives identical random scores
            final = fr.base["final"].float().cpu().numpy()
            fr._base_col = MT.plan_collision(final, fr.gt_traj, fr.fut_boxes) if fr.gt_mask.all() else {}
            fr._base_clear = MT.min_clearance(final, fr.fut_boxes, fr.fut_labels, "pedestrian")
            sel = fr.base_outs["sel_idx"][0].cpu().numpy()
            attn = ego_attention_mass(fr, rp.gnn_names).cpu().numpy()
            grad = fr.plan_grad_scores().cpu().numpy()[sel]
            phys = physics_scores(fr, sel)
            rnd = rng.rand(len(sel))
            prob = fr.det["classification"][-1][0].float().sigmoid().cpu().numpy()
            anc = fr.det["prediction"][-1][0].float().cpu().numpy()
            for r_, j in enumerate(sel):
                is_ped = bool(prob[j].argmax() == PED and prob[j].max() > ped_prob_thr)
                row = {"tag": tag, "frame": fr.i, "sample_token": fr.info["token"], "scene_token": fr.info["scene_token"],
                       "slot": int(j), "rank_in_topk": r_, "pred_class": CLASS_NAMES[int(prob[j].argmax())],
                       "pred_score": float(prob[j].max()), "is_ped": is_ped,
                       "pred_x": float(anc[j, 0]), "pred_y": float(anc[j, 1]),
                       "score_attn": float(attn[r_]), "score_grad": float(grad[r_]),
                       "score_physics": float(phys[r_]), "score_random": float(rnd[r_])}
                e = effects(fr, fr.run(PL.Intervention(mask_keys=[int(j)])))
                row.update({f"mask_{k}": v for k, v in e.items()})
                if is_ped:
                    e = effects(fr, fr.run(PL.Intervention(replace_feature={int(j): mean_ped})))
                    row.update({f"mean_{k}": v for k, v in e.items()})
                    e = effects(fr, fr.run(PL.Intervention(drop_from_topk=[int(j)])))
                    row.update({f"drop_{k}": v for k, v in e.items()})
                agent_rows.append(row)
            pend_frames.append(fr.i)
            if len(pend_frames) >= ckpt_every:
                flush()
    flush()
    rp.close()
    return pd.DataFrame(agent_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=os.environ.get("SDXAI_VERSION", "mini"))
    ap.add_argument("--instances", default="work_dirs/xai/instances/instances.parquet")
    ap.add_argument("--out", default="work_dirs/xai/interventions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frame-stride", type=int, default=1, help="evaluate every k-th val frame (all frames still stream through the model, so temporal state is exact)")
    ap.add_argument("--randomised-stride", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    inst = pd.read_parquet(a.instances)
    df = run("trained", inst, a.version, False, a.seed, stride=a.frame_stride,
             ckpt_dir=os.path.join(a.out, "partial_trained"))
    df.to_parquet(os.path.join(a.out, "planner_mask.parquet"), index=False)
    dfr = run("planner_randomised", inst, a.version, True, a.seed, stride=a.randomised_stride,
              ckpt_dir=os.path.join(a.out, "partial_randomised"))
    dfr.to_parquet(os.path.join(a.out, "planner_mask_randomised.parquet"), index=False)
    print(df.groupby("is_ped")[["mask_dl2_avg", "mask_mode_changed"]].mean())


if __name__ == "__main__":
    main()
