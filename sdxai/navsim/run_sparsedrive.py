"""Run the released SparseDrive-S nuScenes checkpoint on NAVSIM scenes (OpenScene) and write trajectories.

Decoupled from the NAVSIM devkit (python 3.9): this script (python 3.8 / torch 1.13 env) writes
{token: poses (8, 3)} and `make_submission.py` (NAVSIM env) wraps them into the devkit's submission
format for `run_pdm_score_from_submission.py`.

Mapping OpenScene -> SparseDrive (nuScenes conventions):
  cameras   : CAM_F0 -> FRONT, CAM_R0 -> FRONT_RIGHT, CAM_L0 -> FRONT_LEFT, CAM_B0 -> BACK,
              CAM_L1 -> BACK_LEFT, CAM_R1 -> BACK_RIGHT (yaws 0, -55, +55, 180, +112, -112 deg vs
              nuScenes 0, -55, +55, 180, +110, -110; all 64 deg HFOV vs 65 in nuScenes)
  frame     : OpenScene lidar frame == ego frame (rear axle, ground, x fwd / y left). SparseDrive runs
              in a virtual nuScenes-style lidar frame (x right / y fwd / z up) placed where nuScenes
              mounts its lidar: (0.94, 0, 1.84) m in the ego frame.
  images    : 1920x1080 -> resize to width 704 (x0.3667) and keep the bottom 256 rows (same rule the
              nuScenes test pipeline applies to 1600x900)
  history   : the scene's 4 frames at 2 Hz run sequentially, temporal state reset per scene
  command   : OpenScene driving_command [left, straight, right, unknown] -> [right, left, straight]
  ego_status: can_bus accel / rotation rate / velocity (nuScenes can_bus layout)
  output    : final plan (6 x 0.5 s) -> ego frame, extended to 8 x 0.5 s (NAVSIM, 4 s) with the last
              step's displacement (constant velocity); heading = direction of travel.
Interventions (Phase 5 on NAVSIM): --intervention none | mask_peds (all pedestrian keys masked in the
planner gnn) | drop_peds (pedestrians removed from the planner's top-k).
"""
from __future__ import annotations

import argparse
import json
import os
SDXAI_ROOT = os.environ.get("SDXAI_ROOT", os.getcwd())  # workspace with SparseDrive/, nuscenes/, navsim_exp/, b2d/
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from sdxai import planner as PL
from sdxai.runner import PED, build_model, load_cfg, reset_state

DATA = os.environ.get("OPENSCENE_DATA_ROOT", "data/openscene")
DEVKIT = os.path.join(SDXAI_ROOT, "navsim_devkit")
CAM_MAP = [("CAM_F0", "FRONT"), ("CAM_R0", "FRONT_RIGHT"), ("CAM_L0", "FRONT_LEFT"),
           ("CAM_B0", "BACK"), ("CAM_L1", "BACK_LEFT"), ("CAM_R1", "BACK_RIGHT")]
# virtual nuScenes lidar -> OpenScene ego: e = R v + t, x_e = y_v (fwd), y_e = -x_v (left = -right)
T_V2E = np.array([[0.0, 1.0, 0.0, 0.94], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.84], [0, 0, 0, 1.0]])
FH, FW = 256, 704
N_HIST = 4


def scene_tokens(split: str):
    f = os.path.join(DEVKIT, "navsim/planning/script/config/common/train_test_split/scene_filter", f"{split}.yaml")
    cfg = yaml.safe_load(open(f))
    return cfg.get("tokens") or [], cfg.get("log_names") or []


def aug_config(W=1920, H=1080):
    resize = max(FH / H, FW / W)
    nW, nH = int(W * resize), int(H * resize)
    crop_h = nH - FH
    crop_w = int(max(0, nW - FW) / 2)
    return {"resize": resize, "resize_dims": (nW, nH), "crop": (crop_w, crop_h, crop_w + FW, crop_h + FH),
            "flip": False, "rotate": 0, "rotate_3d": 0}


def frame_inputs(fr, blobs, aug):
    imgs, l2i = [], []
    for os_cam, _ in CAM_MAP:
        c = fr["cams"][os_cam]
        im = cv2.imread(os.path.join(blobs, c["data_path"]))  # BGR like mmcv.imread
        if im is None:
            raise FileNotFoundError(c["data_path"])
        imgs.append(np.float32(im))
        cam2e = np.eye(4)
        cam2e[:3, :3] = np.array(c["sensor2lidar_rotation"])
        cam2e[:3, 3] = np.array(c["sensor2lidar_translation"])
        vp = np.eye(4)
        vp[:3, :3] = np.array(c["cam_intrinsic"])
        l2i.append(vp @ np.linalg.inv(cam2e) @ T_V2E)
    cmd_os = int(np.argmax(fr["driving_command"]))
    cmd = np.zeros(3, np.float32)
    cmd[{0: 1, 2: 0}.get(cmd_os, 2)] = 1.0
    cb = np.asarray(fr["can_bus"], np.float32)
    ego_status = np.zeros(10, np.float32)
    ego_status[0:9] = cb[7:16]
    return {"img": imgs, "lidar2img": l2i, "lidar2global": np.asarray(fr["ego2global"]) @ T_V2E,
            "timestamp": fr["timestamp"] / 1e6, "ego_status": ego_status, "gt_ego_fut_cmd": cmd,
            "aug_config": aug}


def to_navsim_poses(plan_v: np.ndarray) -> np.ndarray:
    pts = np.c_[plan_v, np.full(len(plan_v), -1.84), np.ones(len(plan_v))]
    xy = (T_V2E @ pts.T).T[:, :2]              # ego frame (rear axle), 6 x 0.5 s
    step = xy[-1] - xy[-2]
    xy = np.vstack([xy, xy[-1] + step, xy[-1] + 2 * step])  # 8 x 0.5 s (constant velocity)
    prev = np.vstack([[0.0, 0.0], xy[:-1]])
    d = xy - prev
    head = np.arctan2(d[:, 1], d[:, 0])
    for i in range(len(head)):                 # keep previous heading when (almost) standing still
        if np.linalg.norm(d[i]) < 0.05:
            head[i] = head[i - 1] if i else 0.0
    return np.c_[xy, head].astype(np.float32)


VEHICLES = ("car", "truck", "bus", "trailer", "construction_vehicle")


def group_mask(prob, group, thr=0.3):
    """Detections of a group (argmax class, max score > thr; the rescore itself only uses score >= 0.5)."""
    from sdxai.runner import CLASS_NAMES
    det = prob.max(-1).values > thr
    cls = prob.argmax(-1)
    if group == "agents":
        return det
    names = ("pedestrian",) if group == "peds" else VEHICLES
    ids = torch.tensor([CLASS_NAMES.index(n) for n in names], device=cls.device)
    return det & torch.isin(cls, ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--logs-dir", default="test", help="navsim_logs / sensor_blobs sub-directory")
    ap.add_argument("--ckpt", default=None, help="default: released nuScenes stage-2 checkpoint")
    ap.add_argument("--intervention", default="none",
                    choices=["none", "mask_peds", "drop_peds", "full_peds", "full_vehicles", "full_agents"],
                    help="mask_/drop_peds: last frame only, planner attention / top-50 only (original runs). "
                         "full_*: every history frame, masked pass committed to the planner queue, and the same "
                         "agents removed from the collision-aware rescore")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tokens, logs = scene_tokens(a.split)
    blobs = os.path.join(DATA, "sensor_blobs", a.logs_dir)
    have = set(os.listdir(blobs))
    logs = [l for l in logs if l in have] if logs else sorted(have)
    cfg = load_cfg(version="mini")
    model = build_model(cfg, a.ckpt) if a.ckpt else build_model(cfg)
    from mmdet.datasets.pipelines import Compose
    from mmcv.parallel import collate, scatter
    pipeline = Compose([t for t in cfg.data.test.pipeline if t["type"] != "LoadMultiViewImageFromFiles"])
    aug = aug_config()
    h = model.head
    eager_ctx = eager = None
    if a.intervention == "mask_peds" or a.intervention.startswith("full_"):  # mask_keys needs eager planner attention
        from sdxai.eager_attention import use_eager_attention
        eager_ctx = use_eager_attention(model, prefixes=["head.motion_plan_head"])
        eager = eager_ctx.__enter__()
    out, meta, missing = {}, {}, {}
    if os.path.exists(a.out):  # resume
        prev = pickle.load(open(a.out, "rb"))
        out, meta = prev["poses"], prev["meta"]
        missing = prev.get("missing", {})
    todo = []
    for log in logs:
        frames = pickle.load(open(os.path.join(DATA, "navsim_logs", a.logs_dir, f"{log}.pkl"), "rb"))
        idx = {f["token"]: i for i, f in enumerate(frames)}
        for tok in (t for t in tokens if t in idx):
            i = idx[tok]
            if i >= N_HIST - 1:
                todo.append((log, tok, [frames[j] for j in range(i - N_HIST + 1, i + 1)]))
    todo = [t for t in todo if t[1] not in out]
    if a.limit > 0:
        todo = todo[: a.limit]
    t0 = time.time()
    pool = ThreadPoolExecutor(16)
    def load(s):
        try:
            return [frame_inputs(f, blobs, aug) for f in s[2]]
        except FileNotFoundError as e:  # some sensor blobs are missing on the mount: skip the scene
            return str(e)
    AHEAD = 4  # scenes prefetched from the (s3-mounted) sensor blobs
    futs = {k: pool.submit(load, todo[k]) for k in range(min(AHEAD, len(todo)))}
    for k, scene in enumerate(tqdm(todo, desc=f"sparsedrive@{a.split} {a.intervention}")):
        inputs = futs.pop(k).result()
        if k + AHEAD < len(todo):
            futs[k + AHEAD] = pool.submit(load, todo[k + AHEAD])
        if isinstance(inputs, str):
            missing[scene[1]] = inputs
            continue
        reset_state(model)
        with torch.no_grad():
            for j, inp in enumerate(inputs):
                data = scatter(collate([pipeline(inp)], samples_per_gpu=1), [torch.cuda.current_device()])[0]
                metas = {kk: v for kk, v in data.items() if kk != "img"}
                fm = model.extract_feat(data["img"])
                det, mp = h.det_head(fm, metas), h.map_head(fm, metas)
                P = PL.prepare(h.motion_plan_head, det, mp, fm, metas, h.det_head.anchor_encoder,
                               h.det_head.instance_bank.mask, h.det_head.instance_bank.anchor_handler)
                iv, drop = None, None
                prob = det["classification"][-1][0].float().sigmoid()
                is_ped = (prob.argmax(-1) == PED) & (prob.max(-1).values > 0.3)
                if j == len(inputs) - 1 and a.intervention in ("mask_peds", "drop_peds") and is_ped.any():
                    peds = torch.where(is_ped)[0].tolist()
                    iv = (PL.Intervention(mask_keys=peds) if a.intervention == "mask_peds"
                          else PL.Intervention(drop_from_topk=peds))
                elif a.intervention.startswith("full_"):
                    sel = group_mask(prob, a.intervention[5:])
                    if sel.any():
                        drop = torch.where(sel)[0].tolist()
                        iv = PL.Intervention(mask_keys=drop)
                outs = PL.layers(h.motion_plan_head, P, iv, eager)
                dec = PL.decode_plan(h.motion_plan_head, det, outs, metas, rescore_drop=drop)
                PL.commit(h.motion_plan_head, P, outs, det, metas)
        out[scene[1]] = to_navsim_poses(dec["final"].float().cpu().numpy())
        meta[scene[1]] = {"log": scene[0], "n_ped": int(is_ped.sum()), "n_det": int((prob.max(-1).values > 0.3).sum()),
                          "mode": int(dec["mode_idx"]), "cmd": int(np.argmax(scene[2][-1]["driving_command"]))}
        if (k + 1) % 200 == 0:
            pickle.dump({"poses": out, "meta": meta, "missing": missing}, open(a.out, "wb"))
    pickle.dump({"poses": out, "meta": meta, "missing": missing, "args": vars(a), "seconds": time.time() - t0,
                 "n_scenes": len(out), "n_tokens_in_split": len(tokens), "n_logs_with_sensors": len(logs)},
                open(a.out, "wb"))
    if eager_ctx is not None:
        eager_ctx.__exit__(None, None, None)
    print(json.dumps({"n_scenes": len(out), "n_missing_sensor_scenes": len(missing), "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
