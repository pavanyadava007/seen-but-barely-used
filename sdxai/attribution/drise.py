"""D-RISE-lite for detection queries on 6-camera input.

Masks: per camera an 8x8 Bernoulli(p=0.5) grid, bilinearly upsampled to (H+cell_h, W+cell_w) and
randomly shifted (RISE), independently per camera. Masked pixels are set to 0 in the normalised
input (= the dataset mean colour). For every mask the detection similarity
max_j IoU_bev(box_j, box0) * cos(p_j, p0) is computed for every target query of the frame
(one mask set serves all queries in the frame). Saliency_q = sum_i (s_iq - mean_i s_iq) M_i / (N p)
(mean-centred RISE estimator: removes the constant term and mask-density noise of plain RISE,
which otherwise dominates energy-based localisation; pixel ranking is nearly unchanged).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from sdxai.attribution.common import H_IMG, N_CAM, W_IMG, detection_similarity


def make_masks(n: int, grid: int = 8, p: float = 0.5, seed: int = 0, device="cuda") -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    ch, cw = int(np.ceil(H_IMG / grid)), int(np.ceil(W_IMG / grid))
    cells = (torch.rand(n, N_CAM, 1, grid, grid, generator=g, device=device) < p).float()
    up = F.interpolate(cells.flatten(0, 1), size=((grid + 1) * ch, (grid + 1) * cw),
                       mode="bilinear", align_corners=False).view(n, N_CAM, 1, (grid + 1) * ch, (grid + 1) * cw)
    dx = torch.randint(0, cw, (n, N_CAM), generator=g, device=device)
    dy = torch.randint(0, ch, (n, N_CAM), generator=g, device=device)
    out = torch.empty(n, N_CAM, 1, H_IMG, W_IMG, device=device)
    for i in range(n):
        for c in range(N_CAM):
            out[i, c] = up[i, c, :, dy[i, c]:dy[i, c] + H_IMG, dx[i, c]:dx[i, c] + W_IMG]
    return out


@torch.no_grad()
def drise(stream, ref_anchor: torch.Tensor, ref_prob: torch.Tensor, n: int = 500, batch: int = 20,
          p: float = 0.5, seed: int = 0) -> np.ndarray:
    """Returns saliency (Q, 6, H, W) for Q reference detections of the current frame."""
    img = stream.data["img"]
    Q = ref_anchor.shape[0]
    sal = torch.zeros(Q, N_CAM, H_IMG, W_IMG, device=img.device)
    msum = torch.zeros(N_CAM, H_IMG, W_IMG, device=img.device)
    ssum = torch.zeros(Q, device=img.device)
    done = 0
    while done < n:
        b = min(batch, n - done)
        m = make_masks(b, p=p, seed=seed + done, device=img.device)  # b,6,1,H,W
        cls, anc = stream.forward_batch(img * m)
        s = detection_similarity(cls, anc, ref_anchor, ref_prob)  # b,Q
        sal += torch.einsum("bq,bchw->qchw", s, m[:, :, 0])
        msum += m[:, :, 0].sum(0)
        ssum += s.sum(0)
        done += b
    sal = sal - (ssum / n)[:, None, None, None] * msum[None]
    return (sal / (n * p)).cpu().numpy()
