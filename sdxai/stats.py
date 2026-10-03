"""Statistics: cluster bootstrap CIs, paired Wilcoxon, Holm-Bonferroni, Cliff's delta, rule of three.

Unit of analysis: instance nested in scene. The bootstrap resamples clusters (scenes) with
replacement. With fewer than MIN_SCENE_CLUSTERS scenes (v1.0-mini val has 2), a scene bootstrap is
degenerate, so clusters fall back to frames (instances nested in frame) and the returned `unit`
says so. Every CI records its unit.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

MIN_SCENE_CLUSTERS = 5
N_BOOT = 1000


def cluster_unit(df: pd.DataFrame, scene_col: str = "scene_token", frame_col: str = "frame") -> Tuple[pd.Series, str]:
    if scene_col in df and df[scene_col].nunique() >= MIN_SCENE_CLUSTERS:
        return df[scene_col], "scene"
    if frame_col in df:
        return df[scene_col].astype(str) + ":" + df[frame_col].astype(str) if scene_col in df else df[frame_col], "frame"
    return pd.Series(np.arange(len(df)), index=df.index), "instance"


def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray, stat: Callable[[np.ndarray], float] = np.mean,
                      n_boot: int = N_BOOT, seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    values = np.asarray(values, float)
    clusters = np.asarray(clusters)
    ok = np.isfinite(values)
    values, clusters = values[ok], clusters[ok]
    if len(values) == 0:
        return {"est": np.nan, "lo": np.nan, "hi": np.nan, "n": 0, "n_clusters": 0}
    uniq, inv = np.unique(clusters, return_inverse=True)
    groups = [np.where(inv == g)[0] for g in range(len(uniq))]
    rng = np.random.RandomState(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.randint(0, len(groups), len(groups))
        idx = np.concatenate([groups[g] for g in pick])
        boots[b] = stat(values[idx])
    return {"est": float(stat(values)), "lo": float(np.nanpercentile(boots, 100 * alpha / 2)),
            "hi": float(np.nanpercentile(boots, 100 * (1 - alpha / 2))), "n": int(len(values)),
            "n_clusters": int(len(uniq))}


def paired_cluster_bootstrap_diff(a: np.ndarray, b: np.ndarray, clusters: np.ndarray, **kw) -> Dict[str, float]:
    return cluster_bootstrap(np.asarray(a, float) - np.asarray(b, float), clusters, **kw)


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, float)[:, None]
    b = np.asarray(b, float)[None, :]
    return float(((a > b).sum() - (a < b).sum()) / (a.size * b.size)) if a.size and b.size else np.nan


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 5 or np.allclose(a, b):
        return {"W": np.nan, "p": np.nan, "n": int(len(a)), "cliffs_delta": cliffs_delta(a, b)}
    res = wilcoxon(a, b)
    return {"W": float(res.statistic), "p": float(res.pvalue), "n": int(len(a)), "cliffs_delta": cliffs_delta(a, b)}


def holm(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.empty(m)
    running = 0.0
    for rank, j in enumerate(idx):
        running = max(running, (m - rank) * p[ok][j])
        adj[j] = min(1.0, running)
    out[ok] = adj
    return out


def rule_of_three(n: int) -> float:
    """95 % upper bound on an event rate after n trials with zero events."""
    return 3.0 / n if n > 0 else np.nan


def rank_auc(score: np.ndarray, effect: np.ndarray, top_frac: float = 0.2) -> float:
    """AUC of `score` for identifying the agents whose masking `effect` is in the top `top_frac`
    of the frame (ties broken by effect > 0). Returns NaN if one class is empty."""
    score, effect = np.asarray(score, float), np.asarray(effect, float)
    ok = np.isfinite(score) & np.isfinite(effect)
    score, effect = score[ok], effect[ok]
    if len(score) < 3:
        return np.nan
    k = max(1, int(np.ceil(top_frac * len(effect))))
    thr = np.sort(effect)[::-1][k - 1]
    pos = (effect >= thr) & (effect > 0)
    if pos.sum() == 0 or (~pos).sum() == 0:
        return np.nan
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(pos, score))
