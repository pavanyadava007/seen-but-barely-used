"""Concept -> planner treatment (review request R11), recomputed with position-based linking.

For each fine-grained pedestrian concept (positives vs. negatives as in ped_concepts.concept_table), the validation
pedestrians are linked to the Phase-5 planner rows by position (ped_concepts.link_by_position), and the difference
positive minus negative of (a) the attention percentile within the frame and (b) the masking effect (cm) is
averaged over distance bins (0-15, 15-30, 30-100 m) with a scene-cluster bootstrap CI (1,000 resamples).
Writes concept_planner.json and tables/tab_concept_planner.tex."""
import json
import os
SDXAI_ROOT = os.environ.get("SDXAI_ROOT", os.getcwd())  # workspace with SparseDrive/, nuscenes/, navsim_exp/, b2d/
import sys

import numpy as np
import pandas as pd

R = SDXAI_ROOT
X = f"{R}/SparseDrive_tv/work_dirs/xai"
sys.path.insert(0, f"{R}/SparseDrive_tv")
from sdxai.concepts.ped_concepts import concept_table, link_by_position  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_parquet(f"{X}/concepts/ped_concepts/labels.parquet")
d = d[d.split == "val"]
pm = pd.read_parquet(f"{X}/interventions/planner_mask.parquet")
pm = pm[pm.tag == "trained"].copy()
pm["attn_pct"] = pm.groupby("frame").score_attn.rank(pct=True)
pm = pm[pm.is_ped][["frame", "pred_x", "pred_y", "attn_pct", "mask_dl2_avg"]]
rng = np.random.RandomState(0)


# link all validation pedestrians once (labels carry a row id), then select each concept's rows
d = d.reset_index(drop=True)
d["rid"] = np.arange(len(d))
L = link_by_position(d, pm).set_index("rid")[["attn_pct", "mask_dl2_avg"]]
d = d.join(L, on="rid", how="inner")
d["db"] = pd.cut(d.gt_dist, [0, 15, 30, 100], labels=False)
d["cm"] = 100 * d.mask_dl2_avg
NB = 3


def boot_strat(j, m, n=1000):
    """distance-stratified pos-neg difference; scene bootstrap via per-scene sums (fast)."""
    scenes = j.scene_token.unique()
    si = {k: i for i, k in enumerate(scenes)}
    S = np.zeros((len(scenes), NB, 2)); C = np.zeros((len(scenes), NB, 2))
    for (sc, b, y), g in j.groupby(["scene_token", "db", "y"]):
        S[si[sc], int(b), int(y)] = g[m].sum(); C[si[sc], int(b), int(y)] = len(g)

    def stat(w):
        s_, c_ = np.tensordot(w, S, 1), np.tensordot(w, C, 1)
        ok = (c_[:, 0] > 0) & (c_[:, 1] > 0)
        if not ok.any():
            return np.nan
        diff = s_[ok, 1] / c_[ok, 1] - s_[ok, 0] / c_[ok, 0]
        return float(np.average(diff, weights=c_[ok].sum(1)))
    est = stat(np.ones(len(scenes)))
    bs = [stat(np.bincount(rng.randint(0, len(scenes), len(scenes)), minlength=len(scenes))) for _ in range(n)]
    return [est, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]


res, rows = {}, []
for name, (fp, fn, conf, tier) in concept_table().items():
    pos, neg = fp(d).fillna(False).astype(bool), fn(d).fillna(False).astype(bool)
    pos, neg = pos & ~neg, neg & ~pos
    j = d[pos | neg].copy()
    j["y"] = pos[pos | neg].astype(int).values
    j = j.dropna(subset=["db"])
    if j.y.sum() < 10 or (1 - j.y).sum() < 10:
        res[name] = {"n_pos": int(j.y.sum()), "n_neg": int((1 - j.y).sum())}
        continue
    out = {"n_pos": int(j.y.sum()), "n_neg": int((1 - j.y).sum())}
    for m in ("attn_pct", "cm"):
        out[m] = boot_strat(j, m)
    res[name] = out
    a, c = out["attn_pct"], out["cm"]
    rows.append(f"{name.replace('_', ' ')} & {out['n_pos']} & {a[0]:+.2f} [{a[1]:+.2f}, {a[2]:+.2f}] & "
                f"{c[0]:+.2f} [{c[1]:+.2f}, {c[2]:+.2f}] \\\\")
json.dump(res, open(os.path.join(HERE, "concept_planner.json"), "w"), indent=1, default=float)
open(os.path.join(HERE, "tables", "tab_concept_planner.tex"), "w").write("\n".join(rows) + "\n")
print(json.dumps(res, indent=1, default=float))
