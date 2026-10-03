"""Deformable-weight saliency: where a query's deformable aggregation puts its weight.

For query q and decoder layers 1..5 (those sharing the final index space), the group-summed
weights w[l, q, cam, level, pt] are summed over levels; each key point is projected into each
camera and splatted with a Gaussian (sigma = 8 px) carrying that camera/point weight mass.
Also returns per-camera / per-level / per-keypoint mass shares (summary statistics).
"""
from __future__ import annotations

import numpy as np

from sdxai.attribution.common import H_IMG, W_IMG, project, splat

LAYERS = [1, 2, 3, 4, 5]


def dfa_saliency(cap: dict, q: int, proj: np.ndarray, sigma: float = 8.0):
    w = cap["dfa_w"][LAYERS, q].float().cpu().numpy()      # L, cams, levels, pts
    kp = cap["dfa_kp"][LAYERS, q].float().cpu().numpy()     # L, pts, 3
    maps = np.zeros((6, H_IMG, W_IMG), np.float32)
    for li in range(len(LAYERS)):
        uv, depth = project(kp[li], proj)                   # 6,P,2 / 6,P
        valid = (depth > 0.1) & (uv[..., 0] > 0) & (uv[..., 0] < W_IMG) & (uv[..., 1] > 0) & (uv[..., 1] < H_IMG)
        mass = w[li].sum(1)                                 # cams, pts
        maps += splat(uv, valid, mass * valid, sigma)
    tot = w.sum()
    summary = {
        "dfa_cam_share": (w.sum((0, 2, 3)) / tot).tolist(),
        "dfa_level_share": (w.sum((0, 1, 3)) / tot).tolist(),
        "dfa_kp_share": (w.sum((0, 1, 2)) / tot).tolist(),
        "dfa_fixed_kp_share": float(w.sum((0, 1, 2))[:7].sum() / tot),
    }
    return maps, summary
