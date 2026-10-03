"""ROAD faithfulness (MoRF / LeRF) and localisation metrics.

ROAD (Rong et al., 2022) with noisy linear imputation, approximated here by Jacobi diffusion:
removed pixels are repeatedly replaced by the mean of their 8 neighbours (known pixels fixed),
which converges to the discrete harmonic interpolation that NLI solves, then Gaussian noise
(sigma = 0.05 in normalised units) is added so the removed region cannot leak its shape.

Pixels are ranked jointly over the 6 cameras. Budget = 3 x (projected GT box area summed over
cameras), capped at 30 % of all pixels, removed in 20 equal steps. Target = D-RISE detection
similarity of the query (not the class probability). AUC = mean normalised score over steps 0..20
(trapezoid). MoRF lower = more faithful; LeRF higher = more faithful.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from sdxai.attribution.common import H_IMG, N_CAM, W_IMG, detection_similarity

N_STEPS = 20
BUDGET_MULT = 3.0
BUDGET_CAP = 0.30


def impute(img: torch.Tensor, remove: torch.Tensor, iters: int = 60, noise: float = 0.05,
           seed: int = 0) -> torch.Tensor:
    """img (B,6,3,H,W) normalised; remove (B,6,H,W) bool. Returns imputed images."""
    B = img.shape[0]
    x = img.clone().flatten(0, 1)                       # B*6,3,H,W
    r = remove.flatten(0, 1)[:, None].to(x.dtype)       # B*6,1,H,W
    x = x * (1 - r)
    k = torch.ones(1, 1, 3, 3, device=x.device, dtype=x.dtype)
    k[0, 0, 1, 1] = 0
    kk = k.repeat(3, 1, 1, 1)
    ones = torch.ones_like(r)
    cnt = F.conv2d(F.pad(ones, (1, 1, 1, 1)), k)        # neighbour count (edges have fewer)
    for _ in range(iters):
        avg = F.conv2d(F.pad(x, (1, 1, 1, 1)), kk, groups=3) / cnt
        x = x * (1 - r) + avg * r
    g = torch.Generator(device=x.device).manual_seed(seed)
    x = x + r * torch.randn(x.shape, generator=g, device=x.device) * noise
    return x.view_as(img)


@torch.no_grad()
def road_curves(stream, attr: np.ndarray, gt_mask: np.ndarray, ref_anchor: torch.Tensor,
                ref_prob: torch.Tensor, s0: float, qi: int, batch: int = 10,
                seed: int = 0) -> Dict[str, object]:
    """attr (6,H,W), gt_mask (6,H,W) bool. Returns MoRF/LeRF curves + AUCs for one query
    (qi indexes ref_anchor/ref_prob rows)."""
    img = stream.data["img"]
    n_pix = N_CAM * H_IMG * W_IMG
    budget = int(min(BUDGET_MULT * gt_mask.sum(), BUDGET_CAP * n_pix))
    budget = max(budget, N_STEPS)
    rng = np.random.RandomState(seed)
    a = attr.reshape(-1).astype(np.float64) + rng.uniform(0, 1e-12, n_pix)  # random tie-breaking
    out = {"budget_px": budget}
    for name, order in [("morf", np.argsort(-a)), ("lerf", np.argsort(a))]:
        order_t = torch.from_numpy(order[:budget]).to(img.device)
        scores = []
        for s in range(0, N_STEPS, batch):
            ks = list(range(s + 1, min(s + batch, N_STEPS) + 1))
            rem = torch.zeros(len(ks), n_pix, dtype=torch.bool, device=img.device)
            for j, k in enumerate(ks):
                rem[j, order_t[: int(round(k * budget / N_STEPS))]] = True
            rem = rem.view(len(ks), N_CAM, H_IMG, W_IMG)
            x = impute(img.expand(len(ks), -1, -1, -1, -1), rem, seed=seed + s)
            cls, anc = stream.forward_batch(x)
            scores.append(detection_similarity(cls, anc, ref_anchor[qi:qi + 1], ref_prob[qi:qi + 1])[:, 0])
        curve = torch.cat(scores).cpu().numpy() / max(s0, 1e-6)
        full = np.concatenate([[1.0], curve])
        out[f"{name}_curve"] = full.tolist()
        out[f"{name}_auc"] = float(np.trapz(full, dx=1.0 / N_STEPS))
    return out


def localisation(attr: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    a = np.clip(attr, 0, None)
    if a.max() <= 0 or gt_mask.sum() == 0:
        return {"pointing_game": np.nan, "energy_pg": np.nan, "iou@0.2": np.nan, "iou@0.5": np.nan}
    flat = np.unravel_index(a.argmax(), a.shape)
    res = {"pointing_game": float(gt_mask[flat]), "energy_pg": float((a * gt_mask).sum() / a.sum())}
    an = a / a.max()
    for t in [0.2, 0.5]:
        m = an > t
        inter = (m & gt_mask).sum()
        union = (m | gt_mask).sum()
        res[f"iou@{t}"] = float(inter / union) if union else np.nan
    res["gt_area_frac"] = float(gt_mask.mean())
    return res
