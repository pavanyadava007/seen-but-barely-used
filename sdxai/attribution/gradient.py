"""Gradient saliency on the final-layer pedestrian logit of a query.

grad_img     : |d logit_ped / d img| summed over RGB (vanilla gradient)
smoothgrad   : mean over n noisy copies (sigma = 0.1 * (max - min) of the input), |.| summed over RGB
grad_feat    : Grad x Activation on the multi-scale feature maps (the tensor the deformable
               aggregation samples from), summed over channels, |.|, upsampled to input size and
               summed over levels; equivalent to d logit / d sampled features pushed to the maps
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from sdxai.attribution.common import H_IMG, W_IMG
from sdxai.runner import PED


def _logit(stream, imgs, q, cls_idx=PED, feat_hook=None):
    cls, _ = stream.forward_batch(imgs, grad=True, feat_hook=feat_hook)
    return cls[:, q, cls_idx]


def grad_img(stream, q: int, cls_idx: int = PED) -> np.ndarray:
    x = stream.data["img"].clone().requires_grad_(True)
    _logit(stream, x, q, cls_idx).sum().backward()
    return x.grad[0].abs().sum(1).float().cpu().numpy()  # 6,H,W


def smoothgrad(stream, q: int, n: int = 16, sigma: float = 0.1, batch: int = 4, cls_idx: int = PED,
               seed: int = 0) -> np.ndarray:
    img = stream.data["img"]
    scale = sigma * (img.max() - img.min())
    g = torch.Generator(device=img.device).manual_seed(seed)
    acc = torch.zeros_like(img[0])
    done = 0
    while done < n:
        b = min(batch, n - done)
        noise = torch.randn((b,) + img.shape[1:], generator=g, device=img.device) * scale
        x = (img + noise).requires_grad_(True)
        _logit(stream, x, q, cls_idx).sum().backward()
        acc += x.grad.abs().sum(0)
        done += b
    return (acc / n).sum(1).float().cpu().numpy()


def grad_feat(stream, q: int, cls_idx: int = PED) -> np.ndarray:
    """Grad x Act on DAF-format feature maps: col_feats (B, cams*sum(HW), C)."""
    holder = {}

    def hook(fm):
        col, shapes, starts = fm
        col = col.detach().requires_grad_(True)
        holder["col"], holder["shapes"], holder["starts"] = col, shapes, starts
        return [col, shapes, starts]

    _logit(stream, stream.data["img"], q, cls_idx, feat_hook=hook).sum().backward()
    col, shapes, starts = holder["col"], holder["shapes"], holder["starts"]
    ga = (col.grad * col.detach()).sum(-1)[0].abs()  # cams*sumHW
    ncam, nlev = shapes.shape[:2]
    per_cam = ga.reshape(ncam, -1)
    out = torch.zeros(ncam, H_IMG, W_IMG, device=ga.device)
    for lv in range(nlev):
        h, w = shapes[0, lv].tolist()
        s = int(starts[0, lv])
        m = per_cam[:, s:s + h * w].reshape(ncam, 1, h, w)
        out += F.interpolate(m, size=(H_IMG, W_IMG), mode="bilinear", align_corners=False)[:, 0]
    return out.float().cpu().numpy()
