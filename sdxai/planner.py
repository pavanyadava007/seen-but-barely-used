"""Re-implementation of `MotionPlanningHead.forward` split into three parts, with intervention points.

  prepare(head, ...)  -> P      stateful: instance-queue update (run exactly once per frame)
  layers(head, P, iv) -> outs   pure: top-k selection + temp_gnn/gnn/cross_gnn/ffn/refine loop
  commit(head, P, outs)         stateful: instance-queue cache (run once, with the ORIGINAL outs)
  decode_plan(...)              hierarchical selection + collision-aware rescore (upstream code)

With `iv=None` the result equals upstream `MotionPlanningHead.forward` (tests/test_planner_equivalence.py).
Upstream files are not modified; the loop below mirrors motion_planning_head.py line by line.

Interventions (`Intervention` fields, all optional):
  drop_from_topk    : det indices removed from the top-k candidate set (next instance shifts in)
  mask_keys         : det indices whose key/value rows are masked in the planner `gnn`
                      (requires eager attention; implemented via key-validity mask)
  zero_keys         : det indices whose selected feature row is zeroed in `gnn` keys/values
  replace_feature   : {det_idx: feature(256)} substituted in the instance feature before the loop
                      (affects the instance's own query AND its key/value row)
  replace_anchor    : {det_idx: anchor(11)} substituted anchor -> anchor_embed recomputed
  feature_override  : (bs, 900, 256) replaces ALL det instance features (SAE reconstructions)
  temporal_keep     : int k, keep only the k most recent history frames in temp_gnn (0 = none)
  zero_temp_gnn     : bool, mask all temporal keys in temp_gnn (motion+ego use only current)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from projects.mmdet3d_plugin.models.attention import gen_sineembed_for_position
from projects.mmdet3d_plugin.models.instance_bank import topk


@dataclass
class Intervention:
    drop_from_topk: List[int] = field(default_factory=list)
    mask_keys: List[int] = field(default_factory=list)
    zero_keys: List[int] = field(default_factory=list)
    replace_feature: Dict[int, torch.Tensor] = field(default_factory=dict)
    replace_anchor: Dict[int, torch.Tensor] = field(default_factory=dict)
    temporal_keep: Optional[int] = None
    zero_temp_gnn: bool = False
    feature_override: Optional[torch.Tensor] = None

    def is_noop(self) -> bool:
        return not (self.drop_from_topk or self.mask_keys or self.zero_keys or self.replace_feature
                    or self.replace_anchor or self.temporal_keep is not None or self.zero_temp_gnn
                    or self.feature_override is not None)


def prepare(head, det_output, map_output, feature_maps, metas, anchor_encoder, mask, anchor_handler):
    """Everything in upstream forward up to the layer loop, including the stateful queue update."""
    P = {}
    P["det_classification"] = det_output["classification"][-1].sigmoid()
    P["det_anchors"] = det_output["prediction"][-1]
    P["det_confidence"] = P["det_classification"].max(dim=-1).values
    P["instance_feature"] = det_output["instance_feature"]
    P["anchor_embed"] = det_output["anchor_embed"]

    map_classification = map_output["classification"][-1].sigmoid()
    map_confidence = map_classification.max(dim=-1).values
    _, (P["map_feat_sel"], P["map_embed_sel"]) = topk(
        map_confidence, head.num_map, map_output["instance_feature"], map_output["anchor_embed"])

    bs, num_anchor, dim = P["instance_feature"].shape
    ego_feature, ego_anchor, temp_feat, temp_anchor, temp_mask = head.instance_queue.get(
        det_output, feature_maps, metas, bs, mask, anchor_handler)
    P["ego_feature"] = ego_feature
    P["ego_anchor"] = ego_anchor
    P["ego_anchor_embed"] = anchor_encoder(ego_anchor)
    P["temp_feat"] = temp_feat            # (bs, N+1, Q, D) before flatten
    P["temp_anchor_embed"] = anchor_encoder(temp_anchor)
    P["temp_mask"] = temp_mask            # (bs, N+1, Q) True = padded (nn.MultiheadAttention)
    P["anchor_encoder"] = anchor_encoder
    P["det_anchors_raw"] = det_output["prediction"][-1]

    P["motion_anchor"] = head.get_motion_anchor(P["det_classification"], P["det_anchors"])
    plan_anchor = torch.tile(head.plan_anchor[None], (bs, 1, 1, 1, 1))
    P["motion_mode_query"] = head.motion_anchor_encoder(
        gen_sineembed_for_position(P["motion_anchor"][..., -1, :]))
    plan_pos = gen_sineembed_for_position(plan_anchor[..., -1, :])
    P["plan_mode_query"] = head.plan_anchor_encoder(plan_pos).flatten(1, 2).unsqueeze(1)
    P["bs"], P["num_anchor"], P["dim"] = bs, num_anchor, dim
    return P


def select_topk(head, P, iv: Optional[Intervention] = None):
    conf = P["det_confidence"]
    if iv is not None and iv.drop_from_topk:
        conf = conf.clone()
        conf[:, iv.drop_from_topk] = -1.0
    bs, N = conf.shape
    _, idx = torch.topk(conf, head.num_det, dim=1)
    return idx  # (bs, num_det) det indices in confidence order


def layers(head, P, iv: Optional[Intervention] = None, eager_gnn=None):
    """Pure recomputation of the planner layer loop. `eager_gnn`: dict name->EagerMHA currently
    installed (needed only for `mask_keys`). Returns dict with motion/planning lists + sel idx."""
    iv = iv or Intervention()
    bs, num_anchor, dim = P["bs"], P["num_anchor"], P["dim"]
    instance_feature = P["instance_feature"]
    anchor_embed = P["anchor_embed"]
    if iv.feature_override is not None:
        instance_feature = iv.feature_override.to(instance_feature)
    if iv.replace_feature:
        instance_feature = instance_feature.clone()
        for i, f in iv.replace_feature.items():
            instance_feature[:, i] = f.to(instance_feature)
    if iv.replace_anchor:
        anchors = P["det_anchors_raw"].clone()
        for i, a in iv.replace_anchor.items():
            anchors[:, i] = a.to(anchors)
        anchor_embed = P["anchor_encoder"](anchors)

    sel_idx = select_topk(head, P, iv)
    flat = (sel_idx + torch.arange(bs, device=sel_idx.device)[:, None] * num_anchor).reshape(-1)
    feat_sel = instance_feature.flatten(end_dim=1)[flat].reshape(bs, head.num_det, -1)
    embed_sel = anchor_embed.flatten(end_dim=1)[flat].reshape(bs, head.num_det, -1)
    if iv.zero_keys:
        rows = [(sel_idx[0] == i).nonzero().flatten() for i in iv.zero_keys]
        rows = torch.cat(rows) if rows else sel_idx.new_zeros(0)
        feat_sel = feat_sel.clone()
        feat_sel[:, rows] = 0.0

    # key validity for gnn (flash semantics: True = valid); ego key is always valid
    key_valid = None
    if iv.mask_keys:
        key_valid = torch.ones(bs, head.num_det + 1, dtype=torch.bool, device=sel_idx.device)
        for i in iv.mask_keys:
            key_valid[:, :head.num_det] &= sel_idx != i
        assert eager_gnn is not None, "mask_keys requires use_eager_attention on the planner"

    temp_feat = P["temp_feat"].flatten(0, 1)
    temp_embed = P["temp_anchor_embed"].flatten(0, 1)
    temp_mask = P["temp_mask"].flatten(0, 1)
    if iv.temporal_keep is not None or iv.zero_temp_gnn:
        temp_mask = temp_mask.clone()
        Q = temp_mask.shape[-1]
        keep = 0 if iv.zero_temp_gnn else iv.temporal_keep
        # queue order: oldest ... current(last). history slots are indices [0, Q-1)
        n_hist_drop = max(0, (Q - 1) - keep)
        temp_mask[:, :n_hist_drop] = True
        if iv.zero_temp_gnn:
            temp_mask[:, :] = True
            temp_mask[:, -1] = False  # softmax needs >=1 valid key: keep the current frame only

    instance_feature_selected = torch.cat([feat_sel, P["ego_feature"]], dim=1)
    anchor_embed_selected = torch.cat([embed_sel, P["ego_anchor_embed"]], dim=1)
    instance_feature = torch.cat([instance_feature, P["ego_feature"]], dim=1)
    anchor_embed = torch.cat([anchor_embed, P["ego_anchor_embed"]], dim=1)

    motion_classification, motion_prediction = [], []
    planning_classification, planning_prediction, planning_status = [], [], []
    for i, op in enumerate(head.operation_order):
        if head.layers[i] is None:
            continue
        elif op == "temp_gnn":
            instance_feature = head.graph_model(
                i,
                instance_feature.flatten(0, 1).unsqueeze(1),
                temp_feat,
                temp_feat,
                query_pos=anchor_embed.flatten(0, 1).unsqueeze(1),
                key_pos=temp_embed,
                key_padding_mask=temp_mask,
            )
            instance_feature = instance_feature.reshape(bs, num_anchor + 1, dim)
        elif op == "gnn":
            eager = None
            if key_valid is not None:
                eager = head.layers[i].attn
                eager.key_valid = key_valid
            try:
                instance_feature = head.graph_model(
                    i,
                    instance_feature,
                    instance_feature_selected,
                    instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=anchor_embed_selected,
                )
            finally:
                if eager is not None:
                    eager.key_valid = None
        elif op == "norm" or op == "ffn":
            instance_feature = head.layers[i](instance_feature)
        elif op == "cross_gnn":
            instance_feature = head.layers[i](
                instance_feature,
                key=P["map_feat_sel"],
                query_pos=anchor_embed,
                key_pos=P["map_embed_sel"],
            )
        elif op == "refine":
            motion_query = P["motion_mode_query"] + (instance_feature + anchor_embed)[:, :num_anchor].unsqueeze(2)
            plan_query = P["plan_mode_query"] + (instance_feature + anchor_embed)[:, num_anchor:].unsqueeze(2)
            motion_cls, motion_reg, plan_cls, plan_reg, plan_status = head.layers[i](
                motion_query,
                plan_query,
                instance_feature[:, num_anchor:],
                anchor_embed[:, num_anchor:],
            )
            motion_classification.append(motion_cls)
            motion_prediction.append(motion_reg)
            planning_classification.append(plan_cls)
            planning_prediction.append(plan_reg)
            planning_status.append(plan_status)
    return {
        "motion_classification": motion_classification,
        "motion_prediction": motion_prediction,
        "planning_classification": planning_classification,
        "planning_prediction": planning_prediction,
        "planning_status": planning_status,
        "final_instance_feature": instance_feature,
        "sel_idx": sel_idx,
    }


def commit(head, P, outs, det_output, metas):
    """Upstream post-loop caching + output dicts (identical structure to upstream)."""
    num_anchor = P["num_anchor"]
    inst = outs["final_instance_feature"]
    head.instance_queue.cache_motion(inst[:, :num_anchor], det_output, metas)
    head.instance_queue.cache_planning(inst[:, num_anchor:], outs["planning_status"][-1])
    return to_upstream_outputs(head, outs)


def to_upstream_outputs(head, outs):
    motion_output = {
        "classification": outs["motion_classification"],
        "prediction": outs["motion_prediction"],
        "period": head.instance_queue.period,
        "anchor_queue": head.instance_queue.anchor_queue,
    }
    planning_output = {
        "classification": outs["planning_classification"],
        "prediction": outs["planning_prediction"],
        "status": outs["planning_status"],
        "period": head.instance_queue.ego_period,
        "anchor_queue": head.instance_queue.ego_anchor_queue,
    }
    return motion_output, planning_output


def decode_plan(head, det_output, outs, metas, rescore_drop: Optional[List[int]] = None,
                use_rescore: Optional[bool] = None,
                anchor_override: Optional[Dict[int, torch.Tensor]] = None):
    """HierarchicalPlanningDecoder.decode (selection part) on a layers() result.

    rescore_drop: det indices whose predicted trajectories are removed from the collision-aware
    rescore (their confidence is set to 0 for the rescore only, so their motion boxes are filtered
    by upstream's own `det_confidence < score_thresh` rule).
    anchor_override: {det_idx: anchor(11)} used by the rescore's agent boxes (anchor relocation).
    Returns dict: plan_cls_full (3,6) logits after rescore, plan_reg (3,6,6,2) cumulative,
    final (6,2), mode_idx, cmd, plan_cls_cmd_raw (6,) before rescore.
    """
    dec = head.planning_decoder
    classification = outs["planning_classification"][-1]
    prediction = outs["planning_prediction"][-1]
    bs = classification.shape[0]
    classification = classification.reshape(bs, 3, dec.ego_fut_mode)
    prediction = prediction.reshape(bs, 3, dec.ego_fut_mode, dec.ego_fut_ts, 2).cumsum(dim=-2)
    motion_output = {"classification": outs["motion_classification"],
                     "prediction": outs["motion_prediction"]}
    det_out = det_output
    if rescore_drop:
        cls = det_output["classification"][-1].clone()
        cls[:, rescore_drop] = -1e4  # sigmoid -> 0  => filtered from rescore
        det_out = dict(det_output)
        det_out["classification"] = det_output["classification"][:-1] + [cls]
    if anchor_override:
        pred = det_out["prediction"][-1].clone()
        for i, a in anchor_override.items():
            pred[:, i] = a.to(pred)
        det_out = dict(det_out)
        det_out["prediction"] = det_out["prediction"][:-1] + [pred]
    saved = dec.use_rescore
    if use_rescore is not None:
        dec.use_rescore = use_rescore
    try:
        plan_cls_full, final = dec.select(det_out, motion_output, classification, prediction, metas)
    finally:
        dec.use_rescore = saved
    cmd = metas["gt_ego_fut_cmd"].argmax(dim=-1)
    b = torch.arange(bs, device=cmd.device)
    mode_idx = plan_cls_full[b, cmd].argmax(dim=-1)
    return {
        "plan_cls_full": plan_cls_full[0],
        "plan_cls_cmd_raw": classification[0, cmd[0]],
        "plan_reg": prediction[0],
        "final": final[0],
        "mode_idx": int(mode_idx[0]),
        "cmd": int(cmd[0]),
    }


def refine6_from_features(det_head, feat: torch.Tensor, anchor_in: torch.Tensor, time_interval=None):
    """Re-run the final refine module (decoder layer 6) on (possibly modified) layer-6 input
    features. anchor_in = anchors entering refine-6 (= output of refine-5). Returns
    (anchor_out, cls_logits, quality, anchor_embed_of_output)."""
    refines = [m for op, m in zip(det_head.operation_order, det_head.layers) if op == "refine"]
    enc = det_head.anchor_encoder
    embed_in = enc(anchor_in)
    ti = time_interval if time_interval is not None else feat.new_tensor([0.5] * feat.shape[0])
    anchor, cls, qt = refines[-1](feat, anchor_in, embed_in, time_interval=ti, return_cls=True)
    return anchor, cls, qt, enc(anchor)


def splice_P(mp_head, P, feat_new, anchor_new, cls_new, embed_new):
    Pn = dict(P)
    Pn["instance_feature"] = feat_new
    Pn["anchor_embed"] = embed_new
    Pn["det_classification"] = cls_new.sigmoid()
    Pn["det_anchors"] = anchor_new
    Pn["det_anchors_raw"] = anchor_new
    Pn["det_confidence"] = Pn["det_classification"].max(dim=-1).values
    Pn["motion_anchor"] = mp_head.get_motion_anchor(Pn["det_classification"], anchor_new)
    Pn["motion_mode_query"] = mp_head.motion_anchor_encoder(
        gen_sineembed_for_position(Pn["motion_anchor"][..., -1, :]))
    tf = P["temp_feat"].clone()
    n = feat_new.shape[1]
    tf[:, :n, -1] = feat_new  # queue's last slot is the current frame's det features
    Pn["temp_feat"] = tf
    return Pn


def det_with(det_output, anchor_new, cls_new):
    d = dict(det_output)
    d["classification"] = det_output["classification"][:-1] + [cls_new]
    d["prediction"] = det_output["prediction"][:-1] + [anchor_new]
    return d
