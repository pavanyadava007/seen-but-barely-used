"""Sequential val replay with an eager-attention planner, for planner-only interventions.

    rp = Replay(version="mini")
    for fr in rp.frames():
        base = fr.base                    # decode_plan dict of the unmodified planner
        res = fr.run(Intervention(mask_keys=[j]))
        ...
    # after each yield the unmodified base is committed to the instance queue

Each frame exposes: det/map outputs (cached, not recomputed), planner inputs P, the det head's
layer-6 input features and refine-5 anchors (for SAE splicing), refine-6 time interval, GT ego
future (cumulative, 6x2) with mask, GT future boxes/labels for collision, ego attention maps.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from sdxai import planner as PL
from sdxai.eager_attention import AttentionRecorder, use_eager_attention
from sdxai.hooks import DetHeadHooks, ego_row_selector
from sdxai.runner import CLASS_NAMES, SDRunner, reset_state

GNN_OPS = "gnn"


@dataclass
class Frame:
    i: int
    info: dict
    data: dict
    metas: dict
    det: dict
    mp: dict
    P: dict
    cap: dict
    time_interval: torch.Tensor
    gt_traj: np.ndarray
    gt_mask: np.ndarray
    fut_boxes: List[np.ndarray]
    fut_labels: List[np.ndarray]
    rp: "Replay"
    base_outs: dict = field(default_factory=dict)
    base: dict = field(default_factory=dict)
    ego_attn: Dict[str, torch.Tensor] = field(default_factory=dict)

    def run(self, iv: Optional[PL.Intervention] = None, P: Optional[dict] = None, det: Optional[dict] = None,
            rescore_drop=None, anchor_override=None, record=False):
        h = self.rp.head.motion_plan_head
        rec = self.rp.recorder
        rec.clear()
        with torch.no_grad():
            outs = PL.layers(h, P or self.P, iv, self.rp.eager)
            dec = PL.decode_plan(h, det or self.det, outs, self.metas, rescore_drop=rescore_drop,
                                 anchor_override=anchor_override)
        dec["outs"] = outs
        if record:
            dec["ego_attn"] = {k: v[0] for k, v in rec.maps.items()}
        return dec

    def plan_grad_scores(self) -> torch.Tensor:
        """||d selected_plan / d feature_j||_F for every det slot j (selected mode of the cmd)."""
        h = self.rp.head.motion_plan_head
        P = dict(self.P)
        feat = P["instance_feature"].detach().clone().requires_grad_(True)
        P["instance_feature"] = feat
        with torch.enable_grad():
            outs = PL.layers(h, P, None, self.rp.eager)
            reg = outs["planning_prediction"][-1]
            bs = reg.shape[0]
            dec = h.planning_decoder
            reg = reg.reshape(bs, 3, dec.ego_fut_mode, dec.ego_fut_ts, 2).cumsum(-2)
            traj = reg[0, self.base["cmd"], self.base["mode_idx"]].reshape(-1)  # 12
            J = torch.zeros(feat.shape[1], device=feat.device)
            for t in range(traj.numel()):
                g, = torch.autograd.grad(traj[t], feat, retain_graph=t < traj.numel() - 1)
                J += (g[0] ** 2).sum(-1)
        return J.sqrt().detach()


class Replay:
    def __init__(self, version: str = "mini", split: str = "test", capture_det: bool = True):
        self.r = SDRunner(version=version, split=split)
        self.head = self.r.head
        self.hooks = DetHeadHooks(self.head.det_head, capture_dfa=False) if capture_det else None
        self.recorder = AttentionRecorder(select=ego_row_selector())
        self._ctx = use_eager_attention(self.r.model, prefixes=["head.motion_plan_head"], record=self.recorder)
        self.eager = self._ctx.__enter__()
        self.gnn_names = [n for n in self.eager if int(n.split(".")[-1]) in
                          [i for i, op in enumerate(self.head.motion_plan_head.operation_order) if op == "gnn"]]

    def close(self):
        self._ctx.__exit__(None, None, None)

    def _time_interval(self, metas) -> torch.Tensor:
        bank = self.head.det_head.instance_bank
        if bank.cached_anchor is None:
            return metas["timestamp"].new_tensor([bank.default_time_interval]).float()
        ti = (metas["timestamp"] - bank.metas["timestamp"]).float()
        ok = (ti.abs() <= bank.max_time_interval) & (ti != 0)
        return torch.where(ok, ti, ti.new_tensor(bank.default_time_interval))

    def _fut(self, i: int):
        ann = self.r.dataset.get_ann_info(i)
        info = self.r.dataset.data_infos[i]
        gt = np.cumsum(info["gt_ego_fut_trajs"], axis=0)
        fut_boxes = ann.get("fut_boxes", [])
        # labels for future boxes (same masking as upstream get_ann_info)
        labels = []
        for k in range(1, len(fut_boxes) + 1):
            fi = self.r.dataset.data_infos[i + k]
            m = fi["num_lidar_pts"] > 0
            labels.append(fi["gt_names"][m])
        return gt, info["gt_ego_fut_masks"].astype(bool), fut_boxes, labels

    def frames(self, limit: int = -1, indices: Optional[List[int]] = None):
        reset_state(self.r.model)
        h = self.head
        n = len(self.r.dataset) if limit < 0 else min(limit, len(self.r.dataset))
        for i in range(n):
            data = self.r.batch(i)
            metas = self.r.metas_of(data)
            with torch.no_grad():
                fm = self.r.features(data["img"])
                ti = self._time_interval(metas)
                if self.hooks:
                    self.hooks.start()
                det, mp = self.r.perception(fm, metas)
                cap = self.hooks.stop() if self.hooks else {}
                P = PL.prepare(h.motion_plan_head, det, mp, fm, metas, h.det_head.anchor_encoder,
                               h.det_head.instance_bank.mask, h.det_head.instance_bank.anchor_handler)
            gt, gmask, fb, fl = self._fut(i)
            fr = Frame(i=i, info=self.r.dataset.data_infos[i], data=data, metas=metas, det=det, mp=mp, P=P,
                       cap=cap, time_interval=ti, gt_traj=gt, gt_mask=gmask, fut_boxes=fb, fut_labels=fl, rp=self)
            base = fr.run(record=True)
            fr.base, fr.base_outs, fr.ego_attn = base, base["outs"], base["ego_attn"]
            if indices is None or i in indices:
                yield fr
            with torch.no_grad():
                PL.commit(h.motion_plan_head, P, fr.base_outs, det, metas)


def ego_attention_mass(fr: Frame, names: List[str]) -> torch.Tensor:
    """Ego-row attention mass per selected key (50 det keys), mean over heads, summed over gnn layers."""
    tot = None
    for n in names:
        a = fr.ego_attn[n].mean(0)[:-1]  # drop ego key
        tot = a if tot is None else tot + a
    return tot
