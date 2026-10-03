"""Phase 3 orchestrator: perception-level attribution on val pedestrians.

Frame/target selection (mini): per val scene, `--frames-per-scene` frames evenly spaced after a
5-frame temporal warm-up; in each, up to `--max-targets` true-positive pedestrians (same-class
match), critical_w8 first, then random. Targets are resolved at runtime by re-matching the GT
annotation to the current run's predictions (slot indices are not stable across runs because of
atomicAdd nondeterminism in the deformable-aggregation kernel).

Outputs (work_dirs/xai/attribution/):
  attr_results.parquet  one row per (target, method): localisation, ROAD MoRF/LeRF AUC, strata
  road_curves.parquet   per (target, method) MoRF/LeRF curves
  mprt.parquet          per (target, method, stage) similarity to the original map
  target_rand.parquet   target-randomisation proxy for label randomisation
  maps/frame_<i>.npz    4x-downsampled maps for figures
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from sdxai.attribution import sanity as SN
from sdxai.attribution.common import DetStream, detection_similarity, gt_box_masks
from sdxai.attribution.deformable_saliency import dfa_saliency
from sdxai.attribution.drise import drise
from sdxai.attribution.faithfulness import localisation, road_curves
from sdxai.attribution.gradient import grad_feat, grad_img, smoothgrad
from sdxai.instances import GT_RANGE, MATCH_DIST, SCORE_THR, analysis_stratum
from sdxai.runner import CLASS_NAMES, PED, REPO, SDRunner, seed_all

METHODS = ["dfa", "grad", "smoothgrad", "grad_feat", "drise", "random"]


def select_targets(inst: pd.DataFrame, frames_per_scene: int, max_targets: int, seed: int):
    rng = np.random.RandomState(seed)
    ped = inst[(inst.split == "val") & (inst.kind == "tp") & inst.same_class.fillna(False)
               & (inst.gt_class == "pedestrian")]
    chosen = {}
    for scene, d in ped.groupby("scene_token"):
        fis = sorted(d.frame_in_scene.unique())
        fis = [f for f in fis if f >= 5]  # skip the first frames (temporal state still warming up)
        if not fis:  # scene whose pedestrians all appear only in its first 5 frames (never on mini)
            continue
        pick = np.unique(np.linspace(0, len(fis) - 1, frames_per_scene).round().astype(int))
        for f in [fis[i] for i in pick]:
            rows = d[d.frame_in_scene == f]
            cflag = analysis_stratum()["flag"]
            crit = rows[rows[cflag].astype(bool)]
            rest = rows[~rows[cflag].astype(bool)].sample(frac=1.0, random_state=rng)
            rows = pd.concat([crit, rest]).head(max_targets)
            chosen[int(rows.frame.iloc[0])] = rows
    return chosen


def resolve(det, info, gt_indices):
    """Map GT indices -> current query slot via Hungarian on centre distance (as in Phase 2)."""
    cls = det["classification"][-1][0].float().sigmoid().cpu().numpy()
    anc = det["prediction"][-1][0].float().cpu().numpy()
    score = cls.max(-1)
    pidx = np.where(score > SCORE_THR)[0]
    names, boxes = info["gt_names"], info["gt_boxes"]
    pts = info["num_lidar_pts"] + info["num_radar_pts"]
    rng = np.linalg.norm(boxes[:, :2], axis=-1)
    gidx = np.where(np.array([n in CLASS_NAMES for n in names], dtype=bool) & (pts > 0) & (rng <= GT_RANGE))[0]
    D = np.linalg.norm(anc[pidx, None, :2] - boxes[None, gidx, :2], axis=-1)
    r_, c_ = linear_sum_assignment(D)
    g2q = {int(gidx[c]): int(pidx[r]) for r, c in zip(r_, c_) if D[r, c] <= MATCH_DIST}
    out = {}
    for g in gt_indices:
        q = g2q.get(int(g))
        if q is not None and cls[q].argmax() == PED:
            out[int(g)] = q
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=os.environ.get("SDXAI_VERSION", "mini"))
    ap.add_argument("--instances", default="work_dirs/xai/instances/instances.parquet")
    ap.add_argument("--out", default="work_dirs/xai/attribution")
    ap.add_argument("--frames-per-scene", type=int, default=8)
    ap.add_argument("--max-targets", type=int, default=12)
    ap.add_argument("--drise-n", type=int, default=500)
    ap.add_argument("--smoothgrad-n", type=int, default=16)
    ap.add_argument("--mprt-frames", type=int, default=4)
    ap.add_argument("--mprt-targets", type=int, default=3)
    ap.add_argument("--mprt-drise-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "maps"), exist_ok=True)
    # resumable per target frame (the trainval run spans the machine's 8 h power-off): each finished frame's
    # rows go to partial/frame_<i>.pkl; on restart those frames are only streamed (temporal state), not recomputed
    part_dir = os.path.join(a.out, "partial")
    os.makedirs(part_dir, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "p3_config.json"), "w"), indent=2)
    seed_all(a.seed)
    inst = pd.read_parquet(a.instances)
    targets = select_targets(inst, a.frames_per_scene, a.max_targets, a.seed)
    r = SDRunner(version=a.version)
    stream = DetStream(r, capture_dfa=True)
    infos = r.dataset.data_infos
    last = max(targets)
    rows, curves, mprt_rows, trand_rows = [], [], [], []
    mprt_frames = sorted(targets)[:: max(1, len(targets) // a.mprt_frames)][: a.mprt_frames]
    t0 = time.time()
    n_done = 0
    for i in tqdm(range(last + 1), desc="p3 stream"):
        sel = i in targets
        part = os.path.join(part_dir, f"frame_{i}.pkl")
        done = sel and os.path.exists(part)
        det, cap = stream.advance(i, capture=sel and not done)
        if not sel:
            continue
        if done:
            P = pickle.load(open(part, "rb"))
            rows += P["rows"]; curves += P["curves"]; mprt_rows += P["mprt"]; trand_rows += P["trand"]
            n_done += 1
            continue
        n0 = (len(rows), len(curves), len(mprt_rows), len(trand_rows))
        info = infos[i]
        trows = targets[i]
        g2q = resolve(det, info, trows.gt_idx.astype(int).tolist())
        trows = trows[trows.gt_idx.astype(int).isin(list(g2q))]
        if len(trows) == 0:
            pickle.dump({"rows": [], "curves": [], "mprt": [], "trand": []}, open(part, "wb"))
            continue
        qs = [g2q[int(g)] for g in trows.gt_idx]
        proj = stream.metas["projection_mat"][0].float().cpu().numpy()
        ref_anchor = det["prediction"][-1][0, qs].float()
        ref_prob = det["classification"][-1][0, qs].float().sigmoid()
        with torch.no_grad():
            c0, a0 = stream.forward_batch(stream.data["img"])
            s0 = detection_similarity(c0, a0, ref_anchor, ref_prob)[0].cpu().numpy()
        dr = drise(stream, ref_anchor, ref_prob, n=a.drise_n, seed=a.seed + i)
        frame_maps = {}
        for k, (_, row) in enumerate(trows.iterrows()):
            q = qs[k]
            gt_mask, rects = gt_box_masks(info["gt_boxes"][int(row.gt_idx)], proj)
            maps = {}
            maps["dfa"], dfa_sum = dfa_saliency(cap, q, proj)
            maps["grad"] = grad_img(stream, q)
            maps["smoothgrad"] = smoothgrad(stream, q, n=a.smoothgrad_n, seed=a.seed)
            maps["grad_feat"] = grad_feat(stream, q)
            maps["drise"] = dr[k]
            maps["random"] = np.random.RandomState(a.seed + 1000 * i + k).rand(*gt_mask.shape).astype(np.float32)
            meta = {c: row[c] for c in ["frame", "sample_token", "scene_token", "frame_in_scene", "gt_idx",
                                        "gt_ann_token", "gt_dist", "visibility", "vis_bin", "dist_bin",
                                        "critical", "in_ego_corridor", "critical_w8", "in_corridor_w8",
                                        "crowd_density", "moving", "atypical"]}
            meta.update({"slot": q, "is_temporal_slot": q < 600, "s0": float(s0[k]),
                         "n_cams_visible": int(sum(x is not None for x in rects)), **{k2: str(v) for k2, v in dfa_sum.items()}})
            for m in METHODS:
                loc = localisation(maps[m], gt_mask)
                road = road_curves(stream, maps[m], gt_mask, ref_anchor, ref_prob, float(s0[k]), k, seed=a.seed)
                rows.append({**meta, "method": m, **loc, "morf_auc": road["morf_auc"],
                             "lerf_auc": road["lerf_auc"], "budget_px": road["budget_px"]})
                curves.append({"frame": i, "gt_idx": int(row.gt_idx), "method": m,
                               "morf": road["morf_curve"], "lerf": road["lerf_curve"]})
            frame_maps[int(row.gt_idx)] = {m: maps[m][:, ::4, ::4].astype(np.float16) for m in METHODS}
            frame_maps[int(row.gt_idx)]["gt_mask"] = gt_mask[:, ::4, ::4]

            # target-randomisation proxy (label randomisation)
            rs = np.random.RandomState(a.seed + i + k)
            other = int(rs.choice([c for c in range(10) if c != PED]))
            for m, fn in [("grad", lambda c: grad_img(stream, q, cls_idx=c)),
                          ("grad_feat", lambda c: grad_feat(stream, q, cls_idx=c))]:
                trand_rows.append({"frame": i, "gt_idx": int(row.gt_idx), "method": m,
                                   "other_class": CLASS_NAMES[other], **SN.similarity(maps[m], fn(other))})

            # MPRT on a subset (temporal slots only, so slot identity survives randomisation)
            if i in mprt_frames and k < a.mprt_targets and q < 600:
                ref = {  # originals under the same settings as the randomised maps
                    "dfa": maps["dfa"], "grad": maps["grad"], "grad_feat": maps["grad_feat"],
                    "smoothgrad": smoothgrad(stream, q, n=8, seed=a.seed),
                    "drise": drise(stream, ref_anchor[k:k + 1], ref_prob[k:k + 1],
                                   n=a.mprt_drise_n, seed=a.seed + i)[0],
                }
                for st in range(1, SN.N_STAGES):
                    with SN.randomised(r.model, st, seed=a.seed + st):
                        stream.hooks.start()
                        with torch.no_grad():
                            stream.forward_batch(stream.data["img"])
                        capr = stream.hooks.stop()
                        rmaps = {
                            "dfa": dfa_saliency(capr, q, proj)[0],
                            "grad": grad_img(stream, q),
                            "smoothgrad": smoothgrad(stream, q, n=8, seed=a.seed),
                            "grad_feat": grad_feat(stream, q),
                            "drise": drise(stream, ref_anchor[k:k + 1], ref_prob[k:k + 1],
                                           n=a.mprt_drise_n, seed=a.seed + i)[0],
                        }
                    for m, mp in rmaps.items():
                        mprt_rows.append({"frame": i, "gt_idx": int(row.gt_idx), "slot": q, "method": m,
                                          "stage": st, "stage_name": SN.STAGE_NAMES[st],
                                          **SN.similarity(ref[m], mp)})
        np.savez_compressed(os.path.join(a.out, "maps", f"frame_{i}.npz"),
                            **{f"{g}/{m}": v for g, d in frame_maps.items() for m, v in d.items()})
        pd.DataFrame(rows).to_parquet(os.path.join(a.out, "attr_results.parquet"), index=False)
        pickle.dump({"rows": rows[n0[0]:], "curves": curves[n0[1]:], "mprt": mprt_rows[n0[2]:],
                     "trand": trand_rows[n0[3]:]}, open(part + ".tmp", "wb"))
        os.replace(part + ".tmp", part)
        n_done += 1
        with open(os.path.join(a.out, "progress.txt"), "a") as fp:
            fp.write(f"{time.strftime('%F %T')} frame {i} done ({n_done}/{len(targets)} target frames, "
                     f"{len(trows)} targets{', MPRT' if i in mprt_frames else ''})\n")
    pd.DataFrame(rows).to_parquet(os.path.join(a.out, "attr_results.parquet"), index=False)
    pd.DataFrame(curves).to_parquet(os.path.join(a.out, "road_curves.parquet"), index=False)
    pd.DataFrame(mprt_rows).to_parquet(os.path.join(a.out, "mprt.parquet"), index=False)
    pd.DataFrame(trand_rows).to_parquet(os.path.join(a.out, "target_rand.parquet"), index=False)
    print(f"done in {time.time() - t0:.0f}s; {len(rows) // len(METHODS)} targets")


if __name__ == "__main__":
    main()
