"""Phase 2: instance table (one row per matched prediction, unmatched prediction, or missed GT).

Matching: class-agnostic Hungarian assignment on BEV centre distance, accepted if <= 2 m
(nuScenes centre-distance convention; class-agnostic so that confusions such as ped->bicycle stay
visible; `same_class` flags class agreement). Predictions: final-layer instances with
score = max_c sigmoid(cls) > 0.3. GT: detection classes, lidar+radar points > 0, range <= 50 m.

Frames: lidar frame of the current sample (x right, y forward).
Derived fields:
  in_ego_corridor : |lateral| < 2.5 m to the ego GT future path (0.5..3 s), path extended straight
                    ahead to >= 5 m when the ego barely moves, object must be ahead (0 < s <= L+5)
  ttc_est         : range / closing speed (GT velocities, ego velocity = first GT future step / 0.5 s),
                    clipped to [0, 20] s; 20 when not closing
  night, rain     : keywords in scene.description
  crowd_density   : # GT pedestrians within 20 m of ego
  atypical        : child | construction_worker | police_officer | sitting_lying_down
  moving          : attribute *.moving | cycle.with_rider
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd
from nuscenes import NuScenes
from scipy.optimize import linear_sum_assignment

from sdxai.runner import CLASS_NAMES, REPO

SCORE_THR = 0.3
MATCH_DIST = 2.0
GT_RANGE = 50.0
CORRIDOR_HALF_WIDTH = 2.5
TTC_CRITICAL = 4.0
# Widened critical stratum (recorded deviation): mini-val has no GT within 2.5 m of the ego path.
WIDE_HALF_WIDTH = 8.0
WIDE_TTC = 8.0
DIST_BINS = [0, 15, 30, np.inf]
DIST_LABELS = ["<15", "15-30", ">30"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ego_path(fut_trajs: np.ndarray, fut_mask: np.ndarray) -> np.ndarray:
    """(K,2) polyline from ego origin along GT future; extended straight ahead to >= 5 m."""
    steps = fut_trajs[fut_mask.astype(bool)] if fut_mask is not None else fut_trajs
    pts = np.concatenate([np.zeros((1, 2)), np.cumsum(steps, axis=0)], axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = seg.sum()
    if L < 5.0:
        if L > 0.5:
            d = (pts[-1] - pts[0]) / L
        else:
            d = np.array([0.0, 1.0])  # +y is forward in the lidar frame
        pts = np.concatenate([pts, pts[-1:] + d * (5.0 - L)], axis=0)
    return pts


def project_to_path(p: np.ndarray, path: np.ndarray):
    """Return (lateral distance, arc length s) of points p (N,2) to polyline."""
    a, b = path[:-1], path[1:]
    ab = b - a
    ab2 = np.maximum((ab ** 2).sum(-1), 1e-9)
    t = np.clip(((p[:, None] - a[None]) * ab[None]).sum(-1) / ab2[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    d = np.linalg.norm(p[:, None] - proj, axis=-1)
    j = d.argmin(1)
    cum = np.concatenate([[0], np.cumsum(np.sqrt(ab2))])
    s = cum[j] + t[np.arange(len(p)), j] * np.sqrt(ab2[j])
    # before-start points: lateral is distance to origin; mark s <= 0 for points behind the ego
    fwd = (path[1] - path[0]) / max(np.linalg.norm(path[1] - path[0]), 1e-6)
    behind = (p @ fwd) <= 0
    s = np.where(behind & (j == 0), -1.0, s)
    return d[np.arange(len(p)), j], s


def ttc(p: np.ndarray, v_obj: np.ndarray, v_ego: np.ndarray, cap: float = 20.0) -> np.ndarray:
    rng = np.linalg.norm(p, axis=-1)
    v_rel = v_obj - v_ego[None]
    closing = -(p * v_rel).sum(-1) / np.maximum(rng, 1e-6)
    out = np.where(closing > 1e-3, rng / np.maximum(closing, 1e-3), cap)
    return np.clip(out, 0, cap)


class GTIndex:
    def __init__(self, version: str):
        self.nusc = NuScenes(f"v1.0-{version}", os.path.join(REPO, "data/nuscenes"), verbose=False)
        self.vis = {v["token"]: int(v["token"]) for v in self.nusc.visibility}
        self.attr = {a["token"]: a["name"] for a in self.nusc.attribute}

    def scene_flags(self, scene_token: str) -> Dict[str, object]:
        sc = self.nusc.get("scene", scene_token)
        desc = sc["description"].lower()
        return {"scene_name": sc["name"], "night": "night" in desc, "rain": "rain" in desc}

    def ann_meta(self, sample_token: str) -> List[dict]:
        s = self.nusc.get("sample", sample_token)
        out = []
        for t in s["anns"]:
            a = self.nusc.get("sample_annotation", t)
            attrs = [self.attr[x] for x in a["attribute_tokens"]]
            out.append({
                "gt_ann_token": t,
                "gt_instance_token": a["instance_token"],
                "gt_category": a["category_name"],
                "visibility": self.vis.get(a["visibility_token"], 0),
                "gt_attrs": "|".join(attrs),
            })
        return out


def build_rows(split_tag: str, h5_glob: str, infos: dict, gti: GTIndex) -> List[dict]:
    rows = []
    by_token = {x["token"]: x for x in infos}
    for path in sorted(glob.glob(h5_glob)):
        with h5py.File(path) as f:
            for key in sorted(f.keys(), key=int):
                g = f[key]
                tok = g.attrs["sample_token"]
                info = by_token[tok]
                flags = gti.scene_flags(info["scene_token"])
                meta = gti.ann_meta(tok)
                cls = g["det/layer_cls"][-1]
                anc = g["det/layer_anchor"][-1]
                prob = _sigmoid(cls)
                score = prob.max(-1)
                pcls = prob.argmax(-1)
                sel = set(g["plan/sel_idx"][()].tolist())
                pred_idx = np.where(score > SCORE_THR)[0]

                # GT
                names = info["gt_names"]
                pts = info["num_lidar_pts"] + info["num_radar_pts"]
                boxes = info["gt_boxes"]
                vel = np.nan_to_num(info["gt_velocity"])
                rng = np.linalg.norm(boxes[:, :2], axis=-1)
                gmask = np.array([n in CLASS_NAMES for n in names], dtype=bool) & (pts > 0) & (rng <= GT_RANGE)
                gidx = np.where(gmask)[0]

                path = ego_path(info["gt_ego_fut_trajs"], info["gt_ego_fut_masks"])
                v_ego = info["gt_ego_fut_trajs"][0] / 0.5
                ped_gt = [i for i in gidx if names[i] == "pedestrian"]
                crowd = int((rng[ped_gt] <= 20).sum()) if ped_gt else 0

                # Hungarian on centre distance
                if len(pred_idx) and len(gidx):
                    D = np.linalg.norm(anc[pred_idx, None, :2] - boxes[None, gidx, :2], axis=-1)
                    r_, c_ = linear_sum_assignment(D)
                    ok = D[r_, c_] <= MATCH_DIST
                    pairs = list(zip(pred_idx[r_[ok]], gidx[c_[ok]]))
                else:
                    pairs = []
                matched_p = {p for p, _ in pairs}
                matched_g = {q for _, q in pairs}

                base = {"split": split_tag, "frame": int(key), "sample_token": tok,
                        "scene_token": info["scene_token"], "frame_in_scene": int(g.attrs["frame_in_scene"]),
                        "crowd_density": crowd, **flags}

                def gt_fields(q):
                    lat, s = project_to_path(boxes[q:q + 1, :2], path)
                    in_corr = bool(lat[0] < CORRIDOR_HALF_WIDTH and 0 < s[0] <= path_len + 5.0)
                    m = meta[q]
                    return {
                        "gt_idx": int(q), "gt_class": names[q], "gt_x": float(boxes[q, 0]),
                        "gt_y": float(boxes[q, 1]), "gt_dist": float(rng[q]),
                        "gt_azimuth": float(np.degrees(np.arctan2(boxes[q, 0], boxes[q, 1]))),
                        "gt_vx": float(vel[q, 0]), "gt_vy": float(vel[q, 1]),
                        "gt_speed": float(np.linalg.norm(vel[q])), "num_lidar_pts": int(info["num_lidar_pts"][q]),
                        "lateral_to_path": float(lat[0]), "s_along_path": float(s[0]),
                        "in_ego_corridor": in_corr,
                        "ttc_est": float(ttc(boxes[q:q + 1, :2], vel[q:q + 1], v_ego)[0]),
                        **m,
                    }

                path_len = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())

                def pred_fields(p):
                    return {"pred_idx": int(p), "pred_class": CLASS_NAMES[pcls[p]], "pred_score": float(score[p]),
                            "pred_ped_prob": float(prob[p, 8]), "pred_x": float(anc[p, 0]), "pred_y": float(anc[p, 1]),
                            "pred_dist": float(np.linalg.norm(anc[p, :2])), "in_topk50": p in sel,
                            "is_temporal_slot": bool(p < 600), "instance_id": int(g["det/instance_id"][p])}

                for p, q in pairs:
                    rows.append({**base, "kind": "tp", **pred_fields(p), **gt_fields(q),
                                 "match_dist": float(np.linalg.norm(anc[p, :2] - boxes[q, :2])),
                                 "same_class": CLASS_NAMES[pcls[p]] == names[q]})
                for p in pred_idx:
                    if p not in matched_p:
                        rows.append({**base, "kind": "fp", **pred_fields(p)})
                for q in gidx:
                    if q not in matched_g:
                        rows.append({**base, "kind": "fn", **gt_fields(q)})
    return rows


def add_strata(df: pd.DataFrame) -> pd.DataFrame:
    cls = df["gt_class"].fillna(df["pred_class"])
    dist = df["gt_dist"].fillna(df["pred_dist"])
    df["cls"] = cls
    df["dist_bin"] = pd.cut(dist, DIST_BINS, labels=DIST_LABELS, right=False).astype(str)
    df["vis_bin"] = np.where(df["visibility"].isna(), "na",
                             np.where(df["visibility"] <= 2, "occluded(1-2)", "visible(3-4)"))
    df["in_ego_corridor"] = df["in_ego_corridor"].fillna(False).astype(bool)
    df["critical"] = df["in_ego_corridor"] & (df["ttc_est"].fillna(20) < TTC_CRITICAL)
    ahead = df["s_along_path"].fillna(-1) > 0
    df["in_corridor_w8"] = ahead & (df["lateral_to_path"].fillna(np.inf) < WIDE_HALF_WIDTH)
    df["critical_w8"] = df["in_corridor_w8"] & (df["ttc_est"].fillna(20) < WIDE_TTC)
    attrs = df["gt_attrs"].fillna("")
    cat = df["gt_category"].fillna("")
    df["moving"] = attrs.str.contains("moving") | attrs.str.contains("with_rider")
    df["atypical"] = (cat.str.contains("child|construction_worker|police_officer")
                      | attrs.str.contains("sitting_lying_down"))
    df["stratum"] = (df["cls"] + "|" + df["dist_bin"] + "|" + df["vis_bin"] + "|"
                     + np.where(df["in_ego_corridor"], "corr", "off") + "|"
                     + np.where(df["night"], "night", "day"))
    return df


def analysis_stratum(p2_path: str = "work_dirs/xai/instances/p2_acceptance.json") -> dict:
    """Critical stratum used by Phases 3-5: the strict plan definition when it has >= 300 val GT
    pedestrians, otherwise the recorded widened `critical_w8`. Returns {"flag": column, "corridor": column}."""
    try:
        acc = json.load(open(p2_path))
        strict = acc.get("val_ped_critical_gt", 0) >= 300
    except (OSError, ValueError):
        strict = False
    return ({"flag": "critical", "corridor": "in_ego_corridor"} if strict
            else {"flag": "critical_w8", "corridor": "in_corridor_w8"})


def per_class_pr(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for split, d in df.groupby("split"):
        for c in CLASS_NAMES:
            tp = int(((d.kind == "tp") & d.same_class & (d.gt_class == c)).sum())
            n_gt = int((d.gt_class == c).sum())
            n_pred = int(((d.kind != "fn") & (d.pred_class == c)).sum())
            out.append({"split": split, "class": c, "n_gt": n_gt, "n_pred": n_pred, "tp": tp,
                        "recall": tp / n_gt if n_gt else np.nan,
                        "precision": tp / n_pred if n_pred else np.nan})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=os.environ.get("SDXAI_VERSION", "mini"))
    ap.add_argument("--extract", default="work_dirs/xai/extract")
    ap.add_argument("--out", default="work_dirs/xai/instances")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    gti = GTIndex(a.version)
    root = os.path.join(REPO, "data/infos", "mini" if a.version == "mini" else "")
    rows = []
    for split_tag, split_file in [("val", "nuscenes_infos_val.pkl"), ("train", "nuscenes_infos_train.pkl")]:
        infos = pickle.load(open(os.path.join(root, split_file), "rb"))["infos"]
        h5 = os.path.join(a.extract, ("test" if split_tag == "val" else "train") + "_*.h5")
        rows += build_rows(split_tag, h5, infos, gti)
    df = add_strata(pd.DataFrame(rows))
    df.to_parquet(os.path.join(a.out, "instances.parquet"), index=False)
    pr = per_class_pr(df)
    pr.to_csv(os.path.join(a.out, "per_class_pr.csv"), index=False)
    # Acceptance: detection AP trend (Phase 0, mini val) vs recall per class
    ap_path = "work_dirs/xai/p0_baseline/metrics_run1.json"
    ap_full = json.load(open(ap_path))["full"] if os.path.exists(ap_path) else {}
    val_pr = pr[pr.split == "val"].copy()
    val_pr["AP_dist2.0"] = [ap_full.get(f"img_bbox_NuScenes/{c}_AP_dist_2.0") for c in val_pr["class"]]
    ok = val_pr.dropna(subset=["recall", "AP_dist2.0"])
    ok = ok[ok.n_gt >= 20]
    from scipy.stats import spearmanr
    rho = float(spearmanr(ok["recall"], ok["AP_dist2.0"]).correlation) if len(ok) >= 3 else float("nan")
    ped = df[(df.split == "val") & (df.gt_class == "pedestrian")]
    crit_ped = ped[ped.critical]
    crit_ped_tp = crit_ped[crit_ped.kind == "tp"]
    thr_used = TTC_CRITICAL
    widened = {}
    for w in [2.5, 4.0, 6.0, 8.0, 10.0]:
        for thr in [4.0, 8.0, 20.01]:
            c = (ped.lateral_to_path < w) & (ped.s_along_path > 0) & (ped.ttc_est < thr)
            widened[f"halfwidth{w}_ttc<{thr}"] = int(c.sum())
    acc = {
        "rows": len(df), "rows_by_split_kind": df.groupby(["split", "kind"]).size().to_dict(),
        "val_ped_gt": int(len(ped)), "val_ped_critical_gt": int(len(crit_ped)),
        "val_ped_critical_tp": int(len(crit_ped_tp)),
        "val_ped_in_corridor": int(ped.in_ego_corridor.sum()),
        "critical_count_by_ttc_threshold": widened,
        "recall_vs_AP2.0_spearman_classes_ngt>=20": rho,
        "pass_ap_trend": bool(rho > 0.5) if rho == rho else False,
        "pass_300_critical": bool(len(crit_ped) >= 300),
        "analysis_stratum": "critical" if len(crit_ped) >= 300 else "critical_w8",
        "note": ("mini-val has 2 scenes; the >=300 critical-stratum requirement is defined on full val "
                 "and cannot be met on mini. Counts for widened TTC thresholds are recorded."),
        "ttc_threshold_used": thr_used,
        "widened_stratum": {"name": "critical_w8", "half_width_m": WIDE_HALF_WIDTH, "ttc_s": WIDE_TTC,
                            "val_ped_gt": int(ped.critical_w8.sum()),
                            "val_ped_tp_in_topk50": int((ped.critical_w8 & (ped.kind == "tp")
                                                         & ped.in_topk50.fillna(False)).sum())},
    }
    acc["rows_by_split_kind"] = {f"{k[0]}/{k[1]}": v for k, v in acc["rows_by_split_kind"].items()}
    json.dump(acc, open(os.path.join(a.out, "p2_acceptance.json"), "w"), indent=2)
    print(val_pr.to_string(index=False))
    print(json.dumps(acc, indent=2))


if __name__ == "__main__":
    main()
