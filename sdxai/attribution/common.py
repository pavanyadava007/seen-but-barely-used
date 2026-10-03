"""Shared utilities for perception-level attribution (Phase 3).

* `DetStream`   : sequential det-head-only pass over val (map/planner do not affect det state),
                  with a batched, state-tiled forward for perturbation passes at the current frame.
* similarity    : D-RISE detection similarity  max_j IoU_bev(box_j, box0) * cos(p_j, p0)
* projection    : lidar points / GT boxes -> network-input pixels (H=256, W=704) per camera
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.ops import box_iou_rotated

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
from sdxai.hooks import DetHeadHooks
from sdxai.runner import PED, SDRunner, _clone, reset_state

H_IMG, W_IMG, N_CAM = 256, 704, 6
_BANK_KEYS = ["cached_feature", "cached_anchor", "metas", "mask", "confidence",
              "temp_confidence", "instance_id", "prev_id"]


def _tile_tensor(x, B):
    if isinstance(x, torch.Tensor) and x.dim() > 0 and x.shape[0] == 1:
        return x.repeat((B,) + (1,) * (x.dim() - 1))
    return x


def tile_metas(metas: dict, B: int) -> dict:
    out = {}
    for k, v in metas.items():
        if k == "img_metas":
            out[k] = [v[0]] * B
        elif isinstance(v, torch.Tensor):
            out[k] = _tile_tensor(v, B)
        else:
            out[k] = v
    return out


class DetStream:
    """Sequential detection stream with snapshot of the det instance bank before each frame."""

    def __init__(self, runner: SDRunner, capture_dfa: bool = True):
        self.r = runner
        self.det = runner.head.det_head
        self.bank = self.det.instance_bank
        self.hooks = DetHeadHooks(self.det, capture_dfa=capture_dfa)
        self.pre = None      # bank state before current frame
        self.post = None     # bank state after the base pass of current frame
        self.data = None
        self.metas = None
        reset_state(runner.model)

    def _snap(self):
        return {k: _clone(getattr(self.bank, k)) for k in _BANK_KEYS}

    def _restore(self, s):
        for k, v in s.items():
            setattr(self.bank, k, _clone(v))
        self.det.sampler.dn_metas = None

    @torch.no_grad()
    def advance(self, idx: int, capture: bool = False) -> Tuple[dict, Optional[dict]]:
        self.data = self.r.batch(idx)
        self.metas = self.r.metas_of(self.data)
        self.pre = self._snap()
        fm = self.r.features(self.data["img"])
        if capture:
            self.hooks.start()
        det = self.det(fm, self.metas)
        cap = self.hooks.stop() if capture else None
        self.post = self._snap()
        return det, cap

    def forward_batch(self, imgs: torch.Tensor, grad: bool = False, feat_hook=None):
        """imgs: (B, 6, 3, H, W). Runs backbone + det head from the PRE-frame state tiled to B.
        Returns final (cls_logits (B,900,10), anchors (B,900,11)). State is restored after."""
        B = imgs.shape[0]
        self._restore(self.pre)
        for k in ["cached_feature", "cached_anchor", "mask", "confidence", "temp_confidence", "instance_id"]:
            setattr(self.bank, k, _tile_tensor(getattr(self.bank, k), B))
        if self.bank.metas is not None:
            self.bank.metas = tile_metas(self.bank.metas, B)
        metas = tile_metas(self.metas, B)
        try:
            with torch.set_grad_enabled(grad):
                fm = self.r.model.extract_feat(imgs)
                if feat_hook is not None:
                    fm = feat_hook(fm)
                out = self.det(fm, metas)
            return out["classification"][-1], out["prediction"][-1]
        finally:
            self._restore(self.post)


# ------------------------------------------------------------------------------------------
# similarity
# ------------------------------------------------------------------------------------------
def bev_rbox(anchor: torch.Tensor) -> torch.Tensor:
    """(N,11) encoded anchors -> (N,5) [x, y, w, l, yaw] for mmcv box_iou_rotated."""
    yaw = torch.atan2(anchor[..., 6], anchor[..., 7])
    wl = anchor[..., 3:5].exp()
    return torch.stack([anchor[..., 0], anchor[..., 1], wl[..., 0], wl[..., 1], yaw], -1)


def detection_similarity(cls: torch.Tensor, anchors: torch.Tensor, ref_anchor: torch.Tensor,
                         ref_prob: torch.Tensor, pre_filter_m: float = 4.0) -> torch.Tensor:
    """cls (B,900,10) logits, anchors (B,900,11); ref_anchor (Q,11), ref_prob (Q,10).
    Returns (B, Q) similarity = max_j IoU_bev * cos(prob_j, prob0)."""
    B = cls.shape[0]
    Q = ref_anchor.shape[0]
    prob = cls.float().sigmoid()
    out = cls.new_zeros(B, Q, dtype=torch.float32)
    rb_ref = bev_rbox(ref_anchor.float())
    for b in range(B):
        d = torch.cdist(anchors[b, :, :2].float(), ref_anchor[:, :2].float())  # 900,Q
        cand = (d < pre_filter_m).any(1).nonzero().flatten()
        if cand.numel() == 0:
            continue
        iou = box_iou_rotated(bev_rbox(anchors[b, cand].float()), rb_ref)  # C,Q
        cos = F.cosine_similarity(prob[b, cand][:, None], ref_prob[None].float(), dim=-1)
        out[b] = (iou * cos).max(0).values
    return out


# ------------------------------------------------------------------------------------------
# projection
# ------------------------------------------------------------------------------------------
def project(points: np.ndarray, proj: np.ndarray):
    """points (...,3) lidar; proj (6,4,4) -> uv (6,...,2) pixels, depth (6,...)."""
    P = np.concatenate([points, np.ones(points.shape[:-1] + (1,))], -1)
    x = np.einsum("cij,...j->c...i", proj, P)
    depth = x[..., 2]
    uv = x[..., :2] / np.clip(depth[..., None], 1e-5, None)
    return uv, depth


def gt_box_masks(gt_box: np.ndarray, proj: np.ndarray) -> Tuple[np.ndarray, List[Optional[Tuple[int, int, int, int]]]]:
    """GT box (7,) [x,y,z,l,w,h,yaw] -> per-camera boolean masks (6,H,W) of the clipped 2D
    bounding rectangle of the projected 3D box (all corners in front of the camera)."""
    corners = box3d_to_corners(gt_box[None])[0]  # 8,3
    uv, depth = project(corners, proj)  # 6,8,2
    masks = np.zeros((N_CAM, H_IMG, W_IMG), bool)
    rects = []
    for c in range(N_CAM):
        if (depth[c] <= 0.1).any():
            rects.append(None)
            continue
        x0, y0 = np.floor(uv[c].min(0)).astype(int)
        x1, y1 = np.ceil(uv[c].max(0)).astype(int)
        x0, x1 = np.clip([x0, x1], 0, W_IMG)
        y0, y1 = np.clip([y0, y1], 0, H_IMG)
        if x1 - x0 < 1 or y1 - y0 < 1:
            rects.append(None)
            continue
        masks[c, y0:y1, x0:x1] = True
        rects.append((int(x0), int(y0), int(x1), int(y1)))
    return masks, rects


def splat(uv: np.ndarray, valid: np.ndarray, mass: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """uv (6,P,2), valid (6,P), mass (6,P) -> (6,H,W) Gaussian splats."""
    out = np.zeros((N_CAM, H_IMG, W_IMG), np.float32)
    ys = np.arange(H_IMG)[:, None]
    xs = np.arange(W_IMG)[None, :]
    r = int(3 * sigma)
    for c in range(N_CAM):
        for p in np.where(valid[c])[0]:
            u, v = uv[c, p]
            x0, x1 = max(int(u) - r, 0), min(int(u) + r + 1, W_IMG)
            y0, y1 = max(int(v) - r, 0), min(int(v) + r + 1, H_IMG)
            if x0 >= x1 or y0 >= y1:
                continue
            g = np.exp(-(((xs[:, x0:x1] - u) ** 2) + ((ys[y0:y1] - v) ** 2)) / (2 * sigma ** 2))
            out[c, y0:y1, x0:x1] += mass[c, p] * g
    return out
