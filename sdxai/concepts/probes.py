"""Linear probes on decoder layers 2/4/6 instance features, with Hewitt & Liang control tasks.

Rows: true-positive matches (GT-labelled instances) from train+val.
Tasks: class (10-way GT class), dist_bin (3), visibility (4 levels), in_corridor_w8, moving,
       ttc<4s (ttc_est < 4 s).
Splits:
  scene split  : fit on mini-train scenes, test on mini-val scenes (generalisation headline)
  frame split  : random 70/30 split of frames over the pooled data (object identities recur)
Control task (Hewitt & Liang 2019): every "type" = GT instance token receives a random label drawn
from the task's empirical label distribution (fixed per type). Only meaningful when types recur,
so control accuracy and selectivity (= task acc - control acc) are computed on the frame split.
Probe: standardised features, multinomial logistic regression (C=1.0, lbfgs, 3000 iters).
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sdxai.concepts.data import gather

LAYERS = [2, 4, 6]


def tasks(df: pd.DataFrame) -> dict:
    return {
        "class": df.gt_class.values,
        "dist_bin": df.dist_bin.values,
        "visibility": df.visibility.fillna(0).astype(int).values,
        "in_corridor_w8": df.in_corridor_w8.astype(bool).values,
        "in_ego_corridor": df.in_ego_corridor.astype(bool).values,
        "moving": df.moving.astype(bool).values,
        "ttc<4s": (df.ttc_est < 4.0).values,
    }


def control_labels(y: np.ndarray, types: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    vals, counts = np.unique(y, return_counts=True)
    uniq = np.unique(types)
    lab = dict(zip(uniq, rng.choice(vals, size=len(uniq), p=counts / counts.sum())))
    return np.array([lab[t] for t in types])


def fit_eval(Xtr, ytr, Xte, yte):
    if len(np.unique(ytr)) < 2:
        return np.nan, np.nan, np.nan
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    clf.fit(Xtr, ytr)
    p = clf.predict(Xte)
    maj = pd.Series(ytr).value_counts().index[0]
    return accuracy_score(yte, p), balanced_accuracy_score(yte, p), float(np.mean(yte == maj))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="work_dirs/xai/instances/instances.parquet")
    ap.add_argument("--out", default="work_dirs/xai/concepts/probes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=250_000,
                    help="random subsample of TP rows (trainval scale); mini has ~10 k rows")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    inst = pd.read_parquet(a.instances)
    df = inst[inst.kind == "tp"].reset_index(drop=True)
    if len(df) > a.max_rows:
        df = df.sample(a.max_rows, random_state=a.seed).reset_index(drop=True)
    df["pred_idx"] = df.pred_idx.astype(int)
    T = tasks(df)
    types = df.gt_instance_token.values
    rng = np.random.RandomState(a.seed)
    fkey = df.split + ":" + df.frame.astype(str)
    frames = fkey.unique()
    test_frames = set(rng.choice(frames, int(0.3 * len(frames)), replace=False))
    fsplit_te = fkey.isin(test_frames).values
    ssplit_te = (df.split == "val").values
    rows = []
    for L in LAYERS:
        X = gather(df, L)
        for tname, y in T.items():
            acc_s, bacc_s, maj_s = fit_eval(X[~ssplit_te], y[~ssplit_te], X[ssplit_te], y[ssplit_te])
            acc_f, bacc_f, maj_f = fit_eval(X[~fsplit_te], y[~fsplit_te], X[fsplit_te], y[fsplit_te])
            yc = control_labels(y, types, a.seed + L)
            acc_c, bacc_c, maj_c = fit_eval(X[~fsplit_te], yc[~fsplit_te], X[fsplit_te], yc[fsplit_te])
            rows.append({"layer": L, "task": tname, "n": len(y), "n_classes": len(np.unique(y)),
                         "acc_scene_split": acc_s, "bacc_scene_split": bacc_s, "majority_scene_split": maj_s,
                         "acc_frame_split": acc_f, "bacc_frame_split": bacc_f, "majority_frame_split": maj_f,
                         "control_acc_frame_split": acc_c, "control_majority": maj_c,
                         "selectivity_pts": 100 * (acc_f - acc_c)})
            print(rows[-1]["layer"], tname, f"scene {acc_s:.3f} frame {acc_f:.3f} ctrl {acc_c:.3f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "probe_results.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for t, d in res.groupby("task"):
        axes[0].plot(d.layer, d.acc_scene_split, marker="o", label=t)
        axes[1].plot(d.layer, d.selectivity_pts, marker="o", label=t)
    axes[0].set_title("probe accuracy (train scenes -> val scenes)", fontsize=9)
    axes[1].set_title("selectivity (task - control, frame split) [pts]", fontsize=9)
    for ax in axes:
        ax.set_xlabel("decoder layer")
        ax.set_xticks(LAYERS)
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "probe_layer_curves.png"), dpi=120)
    plt.close()
    cls6 = res[(res.task == "class") & (res.layer == 6)].iloc[0]
    acc = {"class_probe_layer6_acc_scene_split": cls6.acc_scene_split,
           "class_probe_layer6_selectivity_pts": cls6.selectivity_pts,
           "pass_class_probe>95%": bool(cls6.acc_scene_split > 0.95),
           "pass_selectivity>30pts": bool(cls6.selectivity_pts > 30)}
    json.dump(acc, open(os.path.join(a.out, "probe_acceptance.json"), "w"), indent=2)
    print(res.round(3).to_string(index=False))
    print(json.dumps(acc, indent=2))


if __name__ == "__main__":
    main()
