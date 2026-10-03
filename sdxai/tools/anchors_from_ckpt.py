"""Write data/kmeans/*.npy from the checkpoint's anchor parameters.

Upstream builds these files with k-means on the train infos, but every one of them is only an
initial value for a parameter that the checkpoint overwrites (instance_bank.anchor of the det and
map heads, motion_anchor, plan_anchor). Writing the checkpoint values gives exactly the anchors the
model runs with and avoids hours of k-means on the full train split.
"""
import os
import sys

import numpy as np
import torch

ckpt = sys.argv[1] if len(sys.argv) > 1 else "ckpt/sparsedrive_stage2.pth"
sd = torch.load(ckpt, map_location="cpu")["state_dict"]
os.makedirs("data/kmeans", exist_ok=True)
out = {
    "data/kmeans/kmeans_det_900.npy": sd["head.det_head.instance_bank.anchor"].numpy(),
    "data/kmeans/kmeans_map_100.npy": sd["head.map_head.instance_bank.anchor"].numpy().reshape(100, 20, 2),
    "data/kmeans/kmeans_motion_6.npy": sd["head.motion_plan_head.motion_anchor"].numpy(),
    "data/kmeans/kmeans_plan_6.npy": sd["head.motion_plan_head.plan_anchor"].numpy(),
}
for p, a in out.items():
    np.save(p, a)
    print(p, a.shape)
