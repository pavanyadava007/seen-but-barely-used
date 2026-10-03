"""Forward hooks on the detection head (no upstream file is modified).

Captured per forward of `det_head` (one frame, batch 1):
  layer_feat[l]   (900, 256)  instance feature entering refine layer l (l = 0..5)
  layer_cls[l]    (900, 10)   classification logits of layer l
  layer_cls_h[l]  (900, 256)  penultimate activation of the cls branch (input of W_cls)
  layer_anchor[l] (900, 11)   refined anchors of layer l
  quality         (900, 2)    final-layer quality (centerness, yawness)
  dfa_w[l]        (900, cams, levels, pts)  deformable weights summed over groups
  dfa_kp[l]       (900, pts, 3)             3D key points (lidar frame)

Index space: layer 0 precedes `InstanceBank.update` (top-k + temporal concatenation), so layer-0
rows do NOT correspond to rows of layers 1..5. Only layers 1..5 share the final index space.
"""
from __future__ import annotations

from typing import Dict, List

import torch


class DetHeadHooks:
    def __init__(self, det_head, capture_dfa: bool = True):
        self.det_head = det_head
        self.capture_dfa = capture_dfa
        self.handles = []
        self.active = False
        self.buf: Dict[str, List[torch.Tensor]] = {}
        self._orig_get_weights = {}
        self._orig_kps = {}
        refines = [m for op, m in zip(det_head.operation_order, det_head.layers) if op == "refine"]
        dfas = [m for op, m in zip(det_head.operation_order, det_head.layers) if op == "deformable"]
        assert len(refines) == 6 and len(dfas) == 6
        self.refines, self.dfas = refines, dfas
        for li, ref in enumerate(refines):
            self.handles.append(ref.register_forward_pre_hook(self._pre_refine(li)))
            self.handles.append(ref.register_forward_hook(self._post_refine(li)))
            self.handles.append(ref.cls_layers[-1].register_forward_pre_hook(self._pre_wcls(li)))
        if capture_dfa:
            for li, dfa in enumerate(dfas):
                self._wrap_dfa(li, dfa)

    # ---- hook factories ----
    def _put(self, key, li, val):
        if self.active:
            self.buf.setdefault(key, [None] * 6)[li] = val.detach()[0]

    def _pre_refine(self, li):
        def f(mod, args):
            self._put("layer_feat", li, args[0])
        return f

    def _post_refine(self, li):
        def f(mod, args, out):
            anchor, cls, qt = out
            self._put("layer_anchor", li, anchor)
            self._put("layer_cls", li, cls)
            if li == 5 and qt is not None and self.active:
                self.buf["quality"] = qt.detach()[0]
        return f

    def _pre_wcls(self, li):
        def f(mod, args):
            self._put("layer_cls_h", li, args[0])
        return f

    def _wrap_dfa(self, li, dfa):
        orig_w = dfa._get_weights
        orig_k = dfa.kps_generator.forward
        hooks = self

        def get_weights(instance_feature, anchor_embed, metas=None):
            w = orig_w(instance_feature, anchor_embed, metas)
            if hooks.active:  # (bs, na, cams, levels, pts, groups) -> sum over groups
                hooks._put("dfa_w", li, w.sum(-1))
            return w

        def kps(*a, **k):
            out = orig_k(*a, **k)
            if hooks.active:
                hooks._put("dfa_kp", li, out if isinstance(out, torch.Tensor) else out[0])
            return out

        dfa._get_weights = get_weights
        dfa.kps_generator.forward = kps
        self._orig_get_weights[li] = (dfa, orig_w)
        self._orig_kps[li] = (dfa.kps_generator, orig_k)

    # ---- control ----
    def start(self):
        self.buf = {}
        self.active = True

    def stop(self) -> Dict[str, torch.Tensor]:
        self.active = False
        out = {}
        for k, v in self.buf.items():
            out[k] = torch.stack(v) if isinstance(v, list) else v
        self.buf = {}
        return out

    def remove(self):
        for h in self.handles:
            h.remove()
        for dfa, f in self._orig_get_weights.values():
            del dfa._get_weights  # instance attribute shadowing the bound method
        for kg, f in self._orig_kps.values():
            del kg.forward
        self.handles = []


def ego_row_selector(num_anchor: int = 900):
    """AttentionRecorder selector: keep only the ego query row of planner gnn maps."""
    def sel(name, attn):
        if "motion_plan_head" not in name:
            return None
        if attn.shape[2] == num_anchor + 1:  # gnn: queries = 900 det + ego
            return attn[0, :, -1, :]  # (heads, keys)
        return None
    return sel
