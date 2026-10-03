"""TCAV (Kim et al. 2018) for the pedestrian logit in decoder layers 2 / 4 / 6.

Concepts (contrasts from instances.parquet, GT pedestrians only):
  occluded     : visibility 1-2 vs 3-4
  corridor     : in_corridor_w8 vs not (strict corridor is empty on mini-val; see Phase 2)
  near         : dist < 15 m vs > 30 m
CAV: logistic regression (standardised) on train-split layer-l features, concept vs counter-set,
balanced by subsampling; 30 CAVs from bootstrap resamples. Random CAVs: 30 CAVs trained on random
partitions of the pooled concept + counter sets (same sizes).
Gradients: exact, live. A forward hook on the norm module feeding refine-l replaces its output by
a leaf tensor; for every val GT-pedestrian true positive the gradient of its own final-layer
pedestrian logit w.r.t. its own layer-l feature row is taken (one backward per query), through
all remaining decoder layers (temporal / graph attention, deformable sampling, refinement).
TCAV score = fraction of those pedestrians with grad . v_cav > 0.
Test: two-sided Welch t-test concept-CAV scores vs random-CAV scores; Bonferroni over
concepts x layers.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from sdxai.attribution.common import DetStream
from sdxai.attribution.run_p3 import resolve
from sdxai.concepts.data import gather
from sdxai.runner import PED, SDRunner

LAYERS = [2, 4, 6]
N_CAV = 30
CONCEPTS = {
    "occluded": (lambda d: d.vis_bin == "occluded(1-2)", lambda d: d.vis_bin == "visible(3-4)"),
    "corridor_w8": (lambda d: d.in_corridor_w8.astype(bool), lambda d: ~d.in_corridor_w8.astype(bool)),
    "corridor_strict": (lambda d: d.in_ego_corridor.astype(bool), lambda d: ~d.in_ego_corridor.astype(bool)),
    "near": (lambda d: d.dist_bin == "<15", lambda d: d.dist_bin == ">30"),
}


def fit_cav(Xp, Xn, rng):
    n = min(len(Xp), len(Xn))
    ip = rng.choice(len(Xp), n, replace=True)
    inn = rng.choice(len(Xn), n, replace=True)
    X = np.concatenate([Xp[ip], Xn[inn]])
    y = np.r_[np.ones(n), np.zeros(n)]
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), y)
    v = clf.coef_[0] / sc.scale_  # direction in raw feature space
    return v / np.linalg.norm(v), clf.score(sc.transform(X), y)


def norm_before_refine(det_head, layer: int):
    idx = [i for i, op in enumerate(det_head.operation_order) if op == "refine"][layer - 1]
    assert det_head.operation_order[idx - 1] == "norm"
    return det_head.layers[idx - 1]


def live_gradients(inst: pd.DataFrame, version: str, stride: int = 1):
    """{layer: (n_ped, 256) gradients} for val GT-pedestrian TPs, plus row metadata."""
    r = SDRunner(version=version)
    st = DetStream(r, capture_dfa=False)
    peds = inst[(inst.split == "val") & (inst.kind == "tp") & (inst.gt_class == "pedestrian")]
    by_frame = {f: d for f, d in peds.groupby("frame")}
    grads = {L: [] for L in LAYERS}
    meta = []
    holder = {}

    def make_hook():
        def hook(mod, args, out):
            leaf = out.detach().requires_grad_(True)
            holder["leaf"] = leaf
            return leaf
        return hook

    for i in tqdm(range(len(r.dataset)), desc="tcav grads"):
        det, _ = st.advance(i)
        if i not in by_frame or i % stride:
            continue
        rows = by_frame[i]
        g2q = resolve(det, r.dataset.data_infos[i], rows.gt_idx.astype(int).tolist())
        rows = rows[rows.gt_idx.astype(int).isin(list(g2q))]
        if len(rows) == 0:
            continue
        qs = [g2q[int(g)] for g in rows.gt_idx]
        per_layer = {}
        for L in LAYERS:
            h = norm_before_refine(r.head.det_head, L).register_forward_hook(make_hook())
            try:
                cls, _ = st.forward_batch(st.data["img"], grad=True)
                leaf = holder["leaf"]
                gl = []
                for k, q in enumerate(qs):
                    g, = torch.autograd.grad(cls[0, q, PED], leaf, retain_graph=k < len(qs) - 1)
                    gl.append(g[0, q].float().cpu().numpy())
                per_layer[L] = np.stack(gl)
            finally:
                h.remove()
        for L in LAYERS:
            grads[L].append(per_layer[L])
        meta.append(rows.assign(slot=qs))
    return {L: np.concatenate(v) for L, v in grads.items()}, pd.concat(meta, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=os.environ.get("SDXAI_VERSION", "mini"))
    ap.add_argument("--instances", default="work_dirs/xai/instances/instances.parquet")
    ap.add_argument("--out", default="work_dirs/xai/concepts/tcav")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frame-stride", type=int, default=1, help="evaluate every k-th val frame (all frames still stream through the model, so temporal state is exact)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    inst = pd.read_parquet(a.instances)
    grads, gmeta = live_gradients(inst, a.version, a.frame_stride)
    np.savez_compressed(os.path.join(a.out, "grads.npz"), **{f"L{L}": g for L, g in grads.items()})
    gmeta.to_parquet(os.path.join(a.out, "grad_rows.parquet"), index=False)

    tr = inst[(inst.split == "train") & (inst.kind == "tp") & (inst.gt_class == "pedestrian")].reset_index(drop=True)
    tr["pred_idx"] = tr.pred_idx.astype(int)
    rng = np.random.RandomState(a.seed)
    rows = []
    for L in LAYERS:
        X = gather(tr, L)
        G = grads[L]
        for cname, (fp, fn) in CONCEPTS.items():
            Xp, Xn = X[fp(tr).values], X[fn(tr).values]
            if len(Xp) < 10 or len(Xn) < 10:
                continue
            c_scores, c_acc, r_scores = [], [], []
            for _ in range(N_CAV):
                v, acc = fit_cav(Xp, Xn, rng)
                c_scores.append(float((G @ v > 0).mean()))
                c_acc.append(acc)
                pool = np.concatenate([Xp, Xn])
                perm = rng.permutation(len(pool))
                vr, _ = fit_cav(pool[perm[:len(Xp)]], pool[perm[len(Xp):]], rng)
                r_scores.append(float((G @ vr > 0).mean()))
            t = ttest_ind(c_scores, r_scores, equal_var=False)
            rows.append({"layer": L, "concept": cname, "n_pos_train": len(Xp), "n_neg_train": len(Xn),
                         "n_val_peds": len(G), "tcav_mean": np.mean(c_scores), "tcav_std": np.std(c_scores),
                         "random_mean": np.mean(r_scores), "random_std": np.std(r_scores),
                         "cav_train_acc": np.mean(c_acc), "t": float(t.statistic), "p": float(t.pvalue)})
    res = pd.DataFrame(rows)
    res["p_bonferroni"] = np.minimum(1.0, res.p * len(res))
    res["significant"] = res.p_bonferroni < 0.05
    res.to_csv(os.path.join(a.out, "tcav_results.csv"), index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
