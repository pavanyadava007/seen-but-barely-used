"""Feature access aligned with instances.parquet rows.

Decoder layer numbering follows the plan: layer L in 1..6 = input of refine module L
(val "full" files store all 6 at index L-1; train "light" files store layers 2,4,6 at 0,1,2).
Layer 1 precedes InstanceBank.update and lives in a different index space - never used here.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, Iterable, List, Optional

import h5py
import numpy as np
import pandas as pd

EXTRACT = os.path.join(os.environ.get("SDXAI_WORK", "work_dirs/xai"), "extract")
LIGHT_LAYERS = {2: 0, 4: 1, 6: 2}
SPARSE_FIELDS = ("layer_feat", "layer_cls_h")  # stored per slot in det/slots when extracted with --sparse-thr


def _files(split: str) -> List[str]:
    tag = "test" if split == "val" else "train"
    return sorted(glob.glob(os.path.join(EXTRACT, f"{tag}_*.h5")))


def layer_index(split: str, layer: int) -> int:
    return layer - 1 if split == "val" else LIGHT_LAYERS[layer]


def _rows(g, what: str, li: int, idx):
    """Rows `idx` (slot indices) of det/<what> at layer position li; handles sparse storage
    (det/slots present and the array holds only those slots). Missing slots -> NaN rows."""
    arr = g[f"det/{what}"]
    a = arr[li] if arr.ndim == 3 else arr[()]
    if "slots" in g["det"] and what in SPARSE_FIELDS:
        pos = {int(s): i for i, s in enumerate(g["det/slots"][()])}
        out = np.full((len(idx), a.shape[-1]), np.nan, np.float32)
        for k, s in enumerate(idx):
            if int(s) in pos:
                out[k] = a[pos[int(s)]]
        return out
    return a[list(idx)]


def gather(rows: pd.DataFrame, layer: int, what: str = "layer_feat") -> np.ndarray:
    """rows must have split, frame, pred_idx. Returns (len(rows), D) float32 in row order."""
    out = None
    for split, d in rows.groupby("split"):
        li = layer_index(split, layer) if what in ("layer_feat",) else -1
        pos = {}
        for i, (fr, p) in enumerate(zip(d.frame.values, d.pred_idx.values)):
            pos.setdefault(int(fr), []).append((d.index[i], int(p)))
        for path in _files(split):
            with h5py.File(path) as f:
                for fr in f.keys():
                    if int(fr) not in pos:
                        continue
                    idx = [p for _, p in pos[int(fr)]]
                    x = _rows(f[fr], what, li, idx)
                    if out is None:
                        out = np.zeros((len(rows), x.shape[-1]), np.float32)
                    for (ri, _), v in zip(pos[int(fr)], x):
                        out[rows.index.get_loc(ri)] = v
    return out


def all_instances(split: str, layer: int, score_thr: float = 0.1):
    """All slots with final score > score_thr: returns (X, meta DataFrame[split, frame, pred_idx, score])."""
    xs, metas = [], []
    for path in _files(split):
        with h5py.File(path) as f:
            for fr in sorted(f.keys(), key=int):
                g = f[fr]
                cls = g["det/layer_cls"][-1]
                score = 1 / (1 + np.exp(-cls.max(-1)))
                idx = np.where(score > score_thr)[0]
                if "slots" in g["det"]:  # sparse storage: only stored slots are available
                    idx = np.intersect1d(idx, g["det/slots"][()])
                li = layer_index(split, layer)
                xs.append(_rows(g, "layer_feat", li, idx).astype(np.float32))
                metas.append(pd.DataFrame({"split": split, "frame": int(fr), "pred_idx": idx,
                                           "score": score[idx], "scene_token": g.attrs["scene_token"]}))
    return np.concatenate(xs), pd.concat(metas, ignore_index=True)


def label_instances(meta: pd.DataFrame, inst: pd.DataFrame) -> pd.DataFrame:
    """Attach GT labels (TP rows of instances.parquet) to slot-level meta; unmatched -> 'unmatched'."""
    tp = inst[inst.kind == "tp"][["split", "frame", "pred_idx", "gt_class", "dist_bin", "vis_bin", "visibility",
                                  "in_corridor_w8", "critical_w8", "moving", "ttc_est", "gt_instance_token",
                                  "same_class", "pred_class"]]
    tp = tp.assign(pred_idx=tp.pred_idx.astype(int))
    m = meta.merge(tp, on=["split", "frame", "pred_idx"], how="left")
    m["label"] = m.gt_class.fillna("unmatched")
    return m
