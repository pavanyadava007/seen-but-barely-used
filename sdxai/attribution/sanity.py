"""Sanity checks (Adebayo et al. 2018) for perception attributions.

MPRT (cascading model-parameter randomisation), top-down over the detection head:
  stage 0: original | 1: refine-6 | 2: decoder block 6 | 3: blocks 5-6 | ... | 7: blocks 1-6
  | 8: + det-head shared modules (anchor encoder, fc_before/after, instance-bank params)
  | 9: + image backbone and neck (full model)
Randomisation: every parameter with >1 element is redrawn from N(mean(p), std(p)) (scale kept),
seeded. Maps are compared to the original with Spearman rank correlation (on |map|, 4x
downsampled) and SSIM (max-normalised 6-camera mosaic). A method FAILS if its similarity after
full randomisation (stage 9) stays > 0.7 on either metric. "Smooth" MPRT: SmoothGrad-type maps
use noise averaging, and similarities are averaged over queries.

Label randomisation: retraining with shuffled labels is out of scope for a single released
checkpoint; the proxy used is TARGET randomisation - the same method explaining a different
(random non-pedestrian) class logit of the same query. High similarity = class-insensitive map.
"""
from __future__ import annotations

import contextlib
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity


def decoder_blocks(det_head) -> List[List[torch.nn.Module]]:
    blocks, cur = [], []
    for op, m in zip(det_head.operation_order, det_head.layers):
        if m is not None:
            cur.append(m)
        if op == "refine":
            blocks.append(cur)
            cur = []
    assert len(blocks) == 6
    return blocks


def stage_modules(model, stage: int) -> List[torch.nn.Module]:
    det = model.head.det_head
    blocks = decoder_blocks(det)
    mods: List[torch.nn.Module] = []
    if stage >= 1:
        mods.append(blocks[5][-1])  # refine of block 6
    if stage >= 2:
        k = min(stage - 1, 6)       # stage 2 -> block 6, stage 7 -> blocks 1-6
        for b in blocks[6 - k:]:
            mods += b
    if stage >= 8:
        mods += [det.anchor_encoder, det.fc_before, det.fc_after, det.instance_bank]
    if stage >= 9:
        mods += [model.img_backbone, model.img_neck]
    return mods


N_STAGES = 10
STAGE_NAMES = ["original", "refine6", "block6", "blocks5-6", "blocks4-6", "blocks3-6",
               "blocks2-6", "blocks1-6", "+det_shared", "+backbone(full)"]


@contextlib.contextmanager
def randomised(model, stage: int, seed: int = 0):
    saved = {}
    g = torch.Generator(device="cuda").manual_seed(seed)
    try:
        for mod in stage_modules(model, stage):
            for n, p in mod.named_parameters():
                if p.numel() > 1 and id(p) not in saved:
                    saved[id(p)] = (p, p.data.clone())
                    std = p.data.float().std()
                    p.data.copy_((torch.randn(p.shape, generator=g, device=p.device) * std
                                  + p.data.float().mean()).to(p.dtype))
        yield
    finally:
        for p, d in saved.values():
            p.data.copy_(d)


def _prep(m: np.ndarray, ds: int = 4) -> np.ndarray:
    m = np.abs(m)
    return m[:, ::ds, ::ds]


def similarity(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """NaN + degenerate=True when either map is non-finite or constant (e.g. fp16 overflow of
    a fully randomised backbone) - such stages must not be read as 'decorrelated'."""
    a, b = _prep(a), _prep(b)
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or a.std() == 0 or b.std() == 0:
        return {"spearman": float("nan"), "ssim": float("nan"), "degenerate": True}
    sp = spearmanr(a.ravel(), b.ravel()).correlation
    am = np.concatenate(list(a), 0)
    bm = np.concatenate(list(b), 0)
    am = am / am.max() if am.max() > 0 else am
    bm = bm / bm.max() if bm.max() > 0 else bm
    ss = structural_similarity(am, bm, data_range=1.0)
    return {"spearman": float(sp), "ssim": float(ss), "degenerate": False}
