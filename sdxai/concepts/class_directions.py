"""Class directions of the final classification layer.

W_cls (10 x 256) is the last Linear of the final SparseBox3DRefinementModule's cls branch; it acts
on the penultimate cls activation h (layer_cls_h[5]), not on the raw instance feature. For a
pedestrian-matched instance the logit margin to class c is
    m_c = h . (w_ped - w_c) + (b_ped - b_c)
and the projection onto the unit direction (w_ped - w_c)/|w_ped - w_c| is reported alongside.
`confused` = GT pedestrian with min_c m_c < 0 (another class wins).
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
import torch

from sdxai.concepts.data import gather
from sdxai.runner import CLASS_NAMES, DEFAULT_CKPT, PED

CONTRASTS = ["bicycle", "motorcycle", "traffic_cone"]


def final_wcls(ckpt: str = DEFAULT_CKPT):
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    keys = sorted({int(k.split(".")[3]) for k in sd if k.startswith("head.det_head.layers.") and ".cls_layers." in k})
    last = keys[-1]
    sub = sorted({int(k.split(".")[5]) for k in sd if k.startswith(f"head.det_head.layers.{last}.cls_layers.")
                  and k.endswith(".weight")})
    W = sd[f"head.det_head.layers.{last}.cls_layers.{sub[-1]}.weight"].float().numpy()
    b = sd[f"head.det_head.layers.{last}.cls_layers.{sub[-1]}.bias"].float().numpy()
    assert W.shape == (10, 256)
    return W, b, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="work_dirs/xai/instances/instances.parquet")
    ap.add_argument("--out", default="work_dirs/xai/concepts/class_directions")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W, b, layer_idx = final_wcls()
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    cos = pd.DataFrame(Wn @ Wn.T, index=CLASS_NAMES, columns=CLASS_NAMES)
    cos.to_csv(os.path.join(a.out, "wcls_cosine.csv"))

    inst = pd.read_parquet(a.instances)
    ped = inst[(inst.kind == "tp") & (inst.gt_class == "pedestrian")].reset_index(drop=True)
    ped["pred_idx"] = ped.pred_idx.astype(int)
    H = gather(ped, 6, what="layer_cls_h")
    logits = H @ W.T + b
    ped["margin_min"] = (logits[:, PED:PED + 1] - np.delete(logits, PED, axis=1)).min(1)
    ped["runner_up"] = [CLASS_NAMES[j] for j in np.where(np.arange(10)[None] == PED, -np.inf, logits).argmax(1)]
    for c in CONTRASTS:
        j = CLASS_NAMES.index(c)
        d = W[PED] - W[j]
        ped[f"margin_{c}"] = H @ d + (b[PED] - b[j])
        ped[f"proj_{c}"] = H @ (d / np.linalg.norm(d))
    ped["confused"] = ped.margin_min < 0
    ped.drop(columns=[c for c in ped.columns if c.startswith("gt_attrs")], errors="ignore") \
        .to_parquet(os.path.join(a.out, "ped_margins.parquet"), index=False)

    # per-stratum summary
    rows = []
    for key in ["split", "dist_bin", "vis_bin", "in_corridor_w8", "night"]:
        for v, d in ped.groupby(key):
            row = {"by": key, "value": str(v), "n": len(d), "confused_rate": float(d.confused.mean())}
            for c in CONTRASTS:
                row[f"margin_{c}_mean"] = float(d[f"margin_{c}"].mean())
                row[f"margin_{c}_p05"] = float(d[f"margin_{c}"].quantile(0.05))
            rows.append(row)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(a.out, "margins_by_stratum.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, c in zip(axes, CONTRASTS):
        for (vb, d) in ped.groupby("vis_bin"):
            ax.hist(d[f"margin_{c}"], bins=40, alpha=0.5, density=True, label=vb)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"logit margin ped - {c}", fontsize=9)
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "margin_hist_by_visibility.png"), dpi=120)
    plt.close()
    res = {"final_refine_layer_index": layer_idx,
           "cos_ped_vs": {c: float(cos.loc["pedestrian", c]) for c in CLASS_NAMES if c != "pedestrian"},
           "n_ped_tp": len(ped), "confused_rate": float(ped.confused.mean()),
           "runner_up_counts_confused": ped[ped.confused].runner_up.value_counts().to_dict(),
           "logit_check_max_abs_err": None}
    # sanity: logits from stored h reproduce stored final logits (fp16 storage of h)
    from sdxai.concepts.data import gather as g2
    stored = g2(ped, 6, what="layer_cls")
    res["logit_check_max_abs_err"] = float(np.abs(stored - logits).max())
    json.dump(res, open(os.path.join(a.out, "class_directions.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
