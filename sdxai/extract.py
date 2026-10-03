"""Phase 1 extraction: sequential pass over a split, one HDF5 per scene.

Per frame (group `/<frame_idx>`), val split ("full"):
  det/layer_feat (6,900,256) f16, det/layer_cls (6,900,10) f32, det/layer_cls_h (6,900,256) f16,
  det/layer_anchor (6,900,11) f32, det/quality (900,2), det/instance_id (900,),
  det/dfa_w (6,900,6,4,13) f16, det/dfa_kp (6,900,13,3) f16,
  det/instance_feature (900,256) f32 + det/anchor_embed (900,256) f32  (planner inputs, cached)
  plan/sel_idx (50,), plan/plan_reg (3,6,6,2), plan/plan_cls_full (3,6), plan/plan_cls_cmd_raw (6,),
  plan/final (6,2), plan/final_eager (6,2), plan/mode_idx, plan/cmd,
  plan/motion_reg (900,6,12,2) f16, plan/motion_cls (900,6) f16,
  plan/attn_ego/<layer_name> (heads, keys) ego-row attention (eager planner rerun)
  meta: sample token, scene token, timestamp, projection_mat (6,4,4), image_wh (6,2)
"train" (light) keeps layer_feat for layers {1,3,5} (= decoder 2,4,6), final cls/anchor/quality/id.

The flash (upstream) planner output is what is committed to the temporal queue; eager attention is
only used for an analysis rerun of the pure planner layers (identical inputs).
"""
from __future__ import annotations

import argparse
from typing import Optional
import json
import os
import time
from collections import defaultdict

import h5py
import numpy as np
import torch
from tqdm import tqdm

from sdxai import planner as PL
from sdxai.eager_attention import AttentionRecorder, use_eager_attention
from sdxai.hooks import DetHeadHooks, ego_row_selector
from sdxai.runner import SDRunner, reset_state


def _np(x, dtype=None):
    x = x.detach().float().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    return x.astype(dtype) if dtype is not None else x


def run(split: str, version: str, out_dir: str, mode: str, limit: int = -1,
        sparse_thr: Optional[float] = None, save_stride: int = 1) -> dict:
    """sparse_thr: if set (trainval scale), per-layer features are stored only for slots whose
    final score > sparse_thr (`det/slots` holds their indices); per-frame 900-slot arrays keep only
    the final layer; deformable weights / planner inputs / motion outputs are not stored (later
    phases recompute them live).
    save_stride: the model still runs on every frame (the temporal instance bank needs the full
    sequence) but only every save_stride-th frame of each scene is written (disk budget at trainval)."""
    os.makedirs(out_dir, exist_ok=True)
    r = SDRunner(version=version, split=split)
    h = r.head
    hooks = DetHeadHooks(h.det_head, capture_dfa=(mode == "full" and sparse_thr is None))
    infos = r.dataset.data_infos
    n = len(r.dataset) if limit < 0 else min(limit, len(r.dataset))
    files = {}
    counts = defaultdict(int)
    reset_state(r.model)
    t0 = time.time()
    for i in tqdm(range(n), desc=f"extract {split}"):
        info = infos[i]
        scene = info["scene_token"]
        if scene not in files:
            for old in list(files):  # frames are ordered by scene: close finished scene files
                if files[old] is not None:
                    files[old].close()
                    files[old] = None
            files[scene] = h5py.File(os.path.join(out_dir, f"{split}_{scene}.h5"), "w")
        f = files[scene]
        data = r.batch(i)
        metas = r.metas_of(data)
        with torch.no_grad():
            fm = r.features(data["img"])
            hooks.start()
            det, mp = r.perception(fm, metas)
            cap = hooks.stop()
            P = PL.prepare(h.motion_plan_head, det, mp, fm, metas, h.det_head.anchor_encoder,
                           h.det_head.instance_bank.mask, h.det_head.instance_bank.anchor_handler)
            outs = PL.layers(h.motion_plan_head, P)
            dec = PL.decode_plan(h.motion_plan_head, det, outs, metas)
            rec = AttentionRecorder(select=ego_row_selector())
            if mode == "full":
                with use_eager_attention(r.model, prefixes=["head.motion_plan_head"], record=rec) as eag:
                    outs_e = PL.layers(h.motion_plan_head, P, None, eag)
                dec_e = PL.decode_plan(h.motion_plan_head, det, outs_e, metas)
            PL.commit(h.motion_plan_head, P, outs, det, metas)

        k = counts[scene]
        counts[scene] += 1
        if k % save_stride:
            continue
        g = f.create_group(str(i))
        g.attrs["sample_token"] = info["token"]
        g.attrs["scene_token"] = scene
        g.attrs["timestamp"] = float(info["timestamp"]) / 1e6
        g.attrs["frame_in_scene"] = k
        gd = g.create_group("det")
        if sparse_thr is not None:
            score = cap["layer_cls"][5].float().sigmoid().max(-1).values
            slots = torch.where(score > sparse_thr)[0]
            gd["slots"] = _np(slots, np.int64)
            layers = [0, 1, 2, 3, 4, 5] if mode == "full" else [1, 3, 5]
            gd["layer_feat"] = _np(cap["layer_feat"][layers][:, slots], np.float16)
            gd["layer_cls_h"] = _np(cap["layer_cls_h"][[5]][:, slots], np.float16)
            gd["layer_cls"] = _np(cap["layer_cls"][[5]], np.float32)
            gd["layer_anchor"] = _np(cap["layer_anchor"][[5]], np.float32)
        elif mode == "full":
            gd["layer_feat"] = _np(cap["layer_feat"], np.float16)
            gd["layer_cls_h"] = _np(cap["layer_cls_h"], np.float16)
            gd["layer_cls"] = _np(cap["layer_cls"], np.float32)
            gd["layer_anchor"] = _np(cap["layer_anchor"], np.float32)
            gd["dfa_w"] = _np(cap["dfa_w"], np.float16)
            gd["dfa_kp"] = _np(cap["dfa_kp"], np.float16)
            gd["instance_feature"] = _np(det["instance_feature"][0], np.float32)
            gd["anchor_embed"] = _np(det["anchor_embed"][0], np.float32)
        else:
            gd["layer_feat"] = _np(cap["layer_feat"][[1, 3, 5]], np.float16)
            gd["layer_cls_h"] = _np(cap["layer_cls_h"][[5]], np.float16)
            gd["layer_cls"] = _np(cap["layer_cls"][[5]], np.float32)
            gd["layer_anchor"] = _np(cap["layer_anchor"][[5]], np.float32)
        gd["quality"] = _np(cap["quality"], np.float32)
        gd["instance_id"] = _np(det["instance_id"][0], np.int64)
        gp = g.create_group("plan")
        gp["sel_idx"] = _np(outs["sel_idx"][0], np.int64)
        gp["plan_reg"] = _np(dec["plan_reg"], np.float32)
        gp["plan_cls_full"] = _np(dec["plan_cls_full"], np.float32)
        gp["plan_cls_cmd_raw"] = _np(dec["plan_cls_cmd_raw"], np.float32)
        gp["final"] = _np(dec["final"], np.float32)
        gp.attrs["mode_idx"] = dec["mode_idx"]
        gp.attrs["cmd"] = dec["cmd"]
        if sparse_thr is None:
            gp["motion_reg"] = _np(outs["motion_prediction"][-1][0], np.float16)
            gp["motion_cls"] = _np(outs["motion_classification"][-1][0], np.float16)
        if mode == "full":
            gp["final_eager"] = _np(dec_e["final"], np.float32)
            ga = gp.create_group("attn_ego")
            for name, maps in rec.maps.items():
                ga[name] = _np(maps[0], np.float32)
        gm = g.create_group("meta")
        gm["projection_mat"] = _np(metas["projection_mat"][0], np.float32)
        gm["image_wh"] = _np(metas["image_wh"][0], np.float32)
    for f in files.values():
        if f is not None:
            f.close()
    hooks.remove()
    manifest = {
        "split": split, "version": version, "mode": mode, "num_frames": n, "save_stride": save_stride,
        "scenes": {s: counts[s] for s in counts}, "files": sorted(os.listdir(out_dir)),
        "seconds": time.time() - t0,
        "bytes": sum(os.path.getsize(os.path.join(out_dir, x)) for x in os.listdir(out_dir)
                     if x.startswith(split)),
    }
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=os.environ.get("SDXAI_VERSION", "mini"))
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--mode", default=None, choices=["full", "light"])
    ap.add_argument("--out", default="work_dirs/xai/extract")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--sparse-thr", type=float, default=None,
                    help="store per-layer features only for slots with score > thr (trainval scale)")
    ap.add_argument("--save-stride", type=int, default=1,
                    help="write every k-th frame of each scene (model still runs on all frames)")
    a = ap.parse_args()
    mode = a.mode or ("full" if a.split == "test" else "light")
    m = run(a.split, a.version, a.out, mode, a.limit, a.sparse_thr, a.save_stride)
    path = os.path.join(a.out, f"extract_manifest_{a.split}.json")
    json.dump({**m, "args": vars(a)}, open(path, "w"), indent=2)
    print(json.dumps({k: v for k, v in m.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
