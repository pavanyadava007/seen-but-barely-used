"""Model/data construction and step-wise forward that mirrors upstream `tools/test.py`.

`SDRunner.step(data)` reproduces `SparseDrive.simple_test` but exposes every intermediate
(feature maps, det/map outputs, planner inputs) so analysis code can re-run parts of the
network. Temporal state (instance bank, instance queue) can be snapshotted and restored,
so perturbation passes do not corrupt the sequential stream.
"""
from __future__ import annotations

import copy
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import scatter
from mmcv.runner import load_checkpoint, wrap_fp16_model

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CFG = os.path.join(REPO, "sdxai/configs/sd_small_stage2_xai.py")
DEFAULT_CKPT = os.path.join(REPO, "ckpt/sparsedrive_stage2.pth")
CLASS_NAMES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
PED = CLASS_NAMES.index("pedestrian")  # 8


def sae_train_score_thr(version: str) -> float:
    """Plan protocol: SAE training on train slots with score > 0.1 (full trainval, ~5 M vectors after
    subsampling). On v1.0-mini that leaves 43 k vectors and the SAE is data-limited, so all slots
    (threshold 0.0, 290 k vectors) are used there."""
    return 0.0 if version == "mini" else 0.1


def seed_all(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cfg(cfg_path: str = DEFAULT_CFG, version: Optional[str] = None) -> Config:
    if version is not None:
        os.environ["SDXAI_VERSION"] = version
    cwd = os.getcwd()
    os.chdir(REPO)  # config uses repo-relative data/ckpt paths
    try:
        cfg = Config.fromfile(cfg_path)
        if cfg.get("plugin", False):
            import importlib
            importlib.import_module("projects.mmdet3d_plugin")
    finally:
        os.chdir(cwd)
    return cfg


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO, path)


def build_dataset(cfg: Config, split: str = "test"):
    from mmdet.datasets import build_dataset as _bd
    dcfg = copy.deepcopy(cfg.data[split])
    dcfg.test_mode = True
    dcfg.data_root = _abs(dcfg.data_root)
    dcfg.ann_file = _abs(dcfg.ann_file)
    if split == "train":  # analysis-time pass over train split: test pipeline, no augmentation
        dcfg.pipeline = cfg.data.test.pipeline
        dcfg.pop("with_seq_flag", None)
        dcfg.pop("sequences_split_num", None)
        dcfg.pop("keep_consistent_seq_aug", None)
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        ds = _bd(dcfg)
    finally:
        os.chdir(cwd)
    return ds


def build_model(cfg: Config, ckpt: str = DEFAULT_CKPT):
    from mmdet.models import build_detector
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        mcfg = copy.deepcopy(cfg.model)
        mcfg.train_cfg = None
        model = build_detector(mcfg, test_cfg=cfg.get("test_cfg"))
        if cfg.get("fp16", None) is not None:
            wrap_fp16_model(model)
        load_checkpoint(model, ckpt, map_location="cpu", logger="silent")
    finally:
        os.chdir(cwd)
    model.cuda().eval()
    return model


def collate_one(dataset, idx: int) -> Dict[str, Any]:
    """Dataset item -> GPU batch of size 1 exactly as MMDataParallel would feed it."""
    from mmcv.parallel import collate
    item = dataset[idx]
    batch = collate([item], samples_per_gpu=1)
    return scatter(batch, [torch.cuda.current_device()])[0]


# ----------------------------------------------------------------------------------------
# temporal state snapshot / restore
# ----------------------------------------------------------------------------------------
_BANK_KEYS = ["cached_feature", "cached_anchor", "metas", "mask", "confidence",
              "temp_confidence", "instance_id", "prev_id"]
_QUEUE_KEYS = ["metas", "prev_instance_id", "prev_confidence", "period", "instance_feature_queue",
               "anchor_queue", "prev_ego_status", "ego_period", "ego_feature_queue", "ego_anchor_queue"]


def _clone(x):
    if isinstance(x, torch.Tensor):
        return x.clone()
    if isinstance(x, list):
        return [_clone(v) for v in x]
    return x  # metas dicts are replaced, never mutated, by upstream code


def snapshot_state(model) -> Dict[str, Dict[str, Any]]:
    h = model.head
    return {
        "det_bank": {k: _clone(getattr(h.det_head.instance_bank, k)) for k in _BANK_KEYS},
        "map_bank": {k: _clone(getattr(h.map_head.instance_bank, k)) for k in _BANK_KEYS},
        "queue": {k: _clone(getattr(h.motion_plan_head.instance_queue, k)) for k in _QUEUE_KEYS},
        "det_dn": h.det_head.sampler.dn_metas,
        "map_dn": h.map_head.sampler.dn_metas,
    }


def restore_state(model, snap) -> None:
    h = model.head
    for k, v in snap["det_bank"].items():
        setattr(h.det_head.instance_bank, k, _clone(v))
    for k, v in snap["map_bank"].items():
        setattr(h.map_head.instance_bank, k, _clone(v))
    for k, v in snap["queue"].items():
        setattr(h.motion_plan_head.instance_queue, k, _clone(v))
    h.det_head.sampler.dn_metas = snap["det_dn"]
    h.map_head.sampler.dn_metas = snap["map_dn"]


def reset_state(model) -> None:
    h = model.head
    h.det_head.instance_bank.reset()
    h.map_head.instance_bank.reset()
    h.motion_plan_head.instance_queue.reset()


# ----------------------------------------------------------------------------------------
# step-wise forward
# ----------------------------------------------------------------------------------------
class SDRunner:
    def __init__(self, cfg_path: str = DEFAULT_CFG, ckpt: str = DEFAULT_CKPT,
                 version: Optional[str] = None, split: str = "test", seed: int = 0):
        seed_all(seed)
        self.cfg = load_cfg(cfg_path, version)
        self.dataset = build_dataset(self.cfg, split)
        self.model = build_model(self.cfg, ckpt)
        self.head = self.model.head

    def __len__(self) -> int:
        return len(self.dataset)

    def batch(self, idx: int) -> Dict[str, Any]:
        return collate_one(self.dataset, idx)

    @staticmethod
    def metas_of(data: Dict[str, Any]) -> Dict[str, Any]:
        # simple_test(img, **data): everything except img is the metas dict
        return {k: v for k, v in data.items() if k != "img"}

    def features(self, img: torch.Tensor):
        return self.model.extract_feat(img)

    def perception(self, feature_maps, metas):
        det_output = self.head.det_head(feature_maps, metas)
        map_output = self.head.map_head(feature_maps, metas)
        return det_output, map_output

    def planner_upstream(self, det_output, map_output, feature_maps, metas):
        h = self.head
        return h.motion_plan_head(
            det_output, map_output, feature_maps, metas,
            h.det_head.anchor_encoder, h.det_head.instance_bank.mask,
            h.det_head.instance_bank.anchor_handler,
        )

    @torch.no_grad()
    def step_upstream(self, data):
        """Exactly SparseDrive.simple_test (reference path)."""
        img = data["img"]
        metas = self.metas_of(data)
        fm = self.features(img)
        outs = self.head(fm, metas)
        return self.head.post_process(outs, metas), outs
