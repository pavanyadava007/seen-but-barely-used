"""Bench2Drive / CARLA leaderboard agent running the released SparseDrive-S nuScenes checkpoint (zero-shot).

Feasibility agent for Phase 8: no Bench2Drive training. The Bench2Drive camera rig mirrors nuScenes
(6 cameras, 1600x900, CAM_* names), so the model's own test pipeline is reused unchanged:
  * images: CARLA BGR frames == what LoadMultiViewImageFromFiles yields; ResizeCropFlipImage,
    NormalizeMultiviewImage (BGR->RGB) and NuScenesSparse4DAdaptor from the stage-2 config
  * geometry: nuScenes-style lidar frame (x right, y forward, z up) at ego (-0.39, 0, 1.84), camera
    lidar2img matrices of the Bench2Drive rig (same values Bench2DriveZoo's VAD agent uses)
  * pose (T_global): GNSS + compass, converted to a right-handed world frame as in the VAD agent
  * command: CARLA RoadOption -> nuScenes one-hot [right, left, straight]
    (LEFT -> left, RIGHT -> right, everything else incl. lane changes -> straight)
  * cadence: inference every 10 simulator ticks (20 Hz -> 2 Hz, the nuScenes keyframe rate the
    temporal instance bank / queue were trained with); between inferences the last plan is
    re-expressed in the current ego frame
  * control: Bench2DriveZoo PIDController on the planned waypoints (identical to the VAD agent)
Optional analysis hook (env SDXAI_MASK_PEDS=1): pedestrian keys are masked in the planner's gnn
(the Phase 5 intervention) so closed-loop outcomes with/without pedestrian attention can be compared.
Agent config string: "<checkpoint.pth | none>+<tag>" (none = released nuScenes stage-2 checkpoint).
"""
from __future__ import annotations

import json
import math
import os
SDXAI_ROOT = os.environ.get("SDXAI_ROOT", os.getcwd())  # workspace with SparseDrive/, nuscenes/, navsim_exp/, b2d/
import pathlib
import sys
import time

import carla
import numpy as np
import torch
from leaderboard.autoagents import autonomous_agent

REPO = os.environ.get("SDXAI_REPO", os.path.join(SDXAI_ROOT, "SparseDrive"))
ZOO = os.environ.get("B2D_ZOO", os.path.join(SDXAI_ROOT, "b2d/Bench2DriveZoo"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(ZOO, "team_code"))
from pid_controller import PIDController  # noqa: E402
from planner import RoutePlanner  # noqa: E402

CAMS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
INFER_EVERY = 10  # ticks at 20 Hz -> 2 Hz
BEV_Z, BEV_PX, BEV_FOV = 40.0, 540, 70.0  # video: top-down camera above the ego vehicle
CLASS_NAMES = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle",
               "bicycle", "pedestrian", "traffic_cone"]

# Bench2Drive rig lidar2img (1600x900), nuScenes-style lidar frame (from Bench2DriveZoo vad_b2d_agent.py)
LIDAR2IMG = {
    "CAM_FRONT": [[1.14251841e+03, 8.00000000e+02, 0.0, -9.52000000e+02],
                  [0.0, 4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
                  [0.0, 1.0, 0.0, -1.19000000e+00], [0, 0, 0, 1]],
    "CAM_FRONT_LEFT": [[6.03961325e-14, 1.39475744e+03, 0.0, -9.20539908e+02],
                       [-3.68618420e+02, 2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                       [-8.19152044e-01, 5.73576436e-01, 0.0, -8.29094072e-01], [0, 0, 0, 1]],
    "CAM_FRONT_RIGHT": [[1.31064327e+03, -4.77035138e+02, 0.0, -4.06010608e+02],
                        [3.68618420e+02, 2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                        [8.19152044e-01, 5.73576436e-01, 0.0, -8.29094072e-01], [0, 0, 0, 1]],
    "CAM_BACK": [[-5.60166031e+02, -8.00000000e+02, 0.0, -1.28800000e+03],
                 [5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
                 [1.22464680e-16, -1.0, 0.0, -1.61000000e+00], [0, 0, 0, 1]],
    "CAM_BACK_LEFT": [[-1.14251841e+03, 8.00000000e+02, 0.0, -6.84385123e+02],
                      [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                      [-9.39692621e-01, -3.42020143e-01, 0.0, -4.92889531e-01], [0, 0, 0, 1]],
    "CAM_BACK_RIGHT": [[3.60989788e+02, -1.34723223e+03, 0.0, -1.04238127e+02],
                       [4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                       [9.39692621e-01, -3.42020143e-01, 0.0, -4.92889531e-01], [0, 0, 0, 1]],
}
LIDAR2EGO = np.array([[0.0, 1.0, 0.0, -0.39], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.84], [0, 0, 0, 1.0]])
CAM_SPECS = {  # Bench2Drive rig (identical to the VAD agent's sensors)
    "CAM_FRONT": (0.80, 0.0, 0.0, 70), "CAM_FRONT_LEFT": (0.27, -0.55, -55.0, 70),
    "CAM_FRONT_RIGHT": (0.27, 0.55, 55.0, 70), "CAM_BACK": (-2.0, 0.0, 180.0, 110),
    "CAM_BACK_LEFT": (-0.32, -0.55, -110.0, 70), "CAM_BACK_RIGHT": (-0.32, 0.55, 110.0, 70),
}


def make_mp4(frame_dir, out_path, fps=10):
    """H.264 mp4 (plays in browsers / VS Code) via the static ffmpeg bundled with imageio-ffmpeg."""
    import subprocess
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-pattern_type", "glob", "-i", str(frame_dir / "*.jpg"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "26", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    str(out_path)], check=True)


def get_entry_point():
    return "SparseDriveAgent"


class SparseDriveAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.tag = path_to_conf_file.split("+")[-1]
        self.step = -1
        self.initialized = False
        self.pid = PIDController()
        self.lat_ref, self.lon_ref = 42.0, 2.0
        self.mask_peds = os.environ.get("SDXAI_MASK_PEDS", "0") == "1"
        self.video = os.environ.get("SDXAI_VIDEO", "0") == "1"
        self.label = None  # set after the checkpoint is known
        from sdxai.runner import build_model, load_cfg
        from mmdet.datasets.pipelines import Compose
        cfg = load_cfg(version="mini")
        ckpt = path_to_conf_file.split("+")[0]
        self.ckpt = ckpt if ckpt.endswith(".pth") else None  # None -> released nuScenes stage-2 checkpoint
        self.model = build_model(cfg, self.ckpt) if self.ckpt else build_model(cfg)
        self.label = os.environ.get("SDXAI_LABEL") or (
            "SparseDrive zero-shot (nuScenes ckpt)" if self.ckpt is None
            else f"SparseDrive fine-tuned on Bench2Drive ({pathlib.Path(self.ckpt).parent.name}/{pathlib.Path(self.ckpt).name})")
        self.pipeline = Compose([t for t in cfg.data.test.pipeline if t["type"] != "LoadMultiViewImageFromFiles"])
        dac = cfg.data_aug_conf
        H, W = dac["H"], dac["W"]
        fH, fW = dac["final_dim"]
        resize = max(fH / H, fW / W)
        newW, newH = int(W * resize), int(H * resize)
        crop_h = int((1 - np.mean(dac["bot_pct_lim"])) * newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        self.aug_config = {"resize": resize, "resize_dims": (newW, newH),
                           "crop": (crop_w, crop_h, crop_w + fW, crop_h + fH),
                           "flip": False, "rotate": 0, "rotate_3d": 0}
        self.plan_world = None  # last plan as world-frame points
        self.meta = []
        self.save_dir = None
        if os.environ.get("SAVE_PATH"):
            self.save_dir = pathlib.Path(os.environ["SAVE_PATH"]) / f"sparsedrive_{self.tag}"
            self.save_dir.mkdir(parents=True, exist_ok=True)
            if self.video:
                (self.save_dir / "video").mkdir(exist_ok=True)

    def _init(self):
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True

    def sensors(self):
        s = []
        for cid, (x, y, yaw, fov) in CAM_SPECS.items():
            s.append({"type": "sensor.camera.rgb", "x": x, "y": y, "z": 1.60, "roll": 0.0, "pitch": 0.0, "yaw": yaw,
                      "width": 1600, "height": 900, "fov": fov, "id": cid})
        s += [{"type": "sensor.other.imu", "x": -1.4, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
               "sensor_tick": 0.05, "id": "IMU"},
              {"type": "sensor.other.gnss", "x": -1.4, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
               "sensor_tick": 0.01, "id": "GPS"},
              {"type": "sensor.speedometer", "reading_frequency": 20, "id": "SPEED"}]
        if os.environ.get("SDXAI_VIDEO", "0") == "1":
            s.append({"type": "sensor.camera.rgb", "x": 0.0, "y": 0.0, "z": BEV_Z, "roll": 0.0, "pitch": -90.0,
                      "yaw": 0.0, "width": BEV_PX, "height": BEV_PX, "fov": BEV_FOV, "id": "BEV"})
        return s

    # ---------------------------------------------------------------- geometry helpers
    def gps_to_location(self, gps):
        R = 6378137.0
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat + 90) * math.pi / 360.0)) * (R * scale)
        mx = (lon * (math.pi * R * scale)) / 180.0
        y = scale * R * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * R / 180.0
        return np.array([x, y])

    def _ego2world(self, pos, compass):
        theta = -compass + np.pi / 2  # right-handed yaw (as the VAD agent)
        T = np.eye(4)
        c, s = np.cos(theta), np.sin(theta)
        T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        T[0, 3], T[1, 3] = pos[0], -pos[1]
        return T

    # ---------------------------------------------------------------- main step
    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        self.step += 1
        gps = input_data["GPS"][1][:2]
        speed = input_data["SPEED"][1]["speed"]
        imu = input_data["IMU"][1]
        compass = imu[-1] if not np.isnan(imu[-1]) else 0.0
        pos = self.gps_to_location(gps)
        near_node, near_command = self._route_planner.run_step(pos)
        ego2world = self._ego2world(pos, compass)
        lidar2global = ego2world @ LIDAR2EGO

        if self.step % INFER_EVERY == 0 or self.plan_world is None:
            imgs = [np.float32(input_data[c][1][:, :, :3]) for c in CAMS]  # BGR, as mmcv.imread
            cmd = np.zeros(3, np.float32)
            name = str(near_command).split(".")[-1]
            cmd[{"RIGHT": 0, "LEFT": 1}.get(name, 2)] = 1.0
            ego_status = np.zeros(10, np.float32)
            ego_status[0:3] = [imu[0], -imu[1], imu[2]]
            ego_status[3:6] = -np.asarray(imu[3:6])
            ego_status[6] = speed
            results = {"img": imgs, "lidar2img": [np.array(LIDAR2IMG[c], np.float64) for c in CAMS],
                       "lidar2global": lidar2global, "timestamp": self.step / 20.0,
                       "ego_status": ego_status, "gt_ego_fut_cmd": cmd, "aug_config": self.aug_config}
            data = self.pipeline(results)
            from mmcv.parallel import collate, scatter
            batch = scatter(collate([data], samples_per_gpu=1), [torch.cuda.current_device()])[0]
            plan_lidar, info = self._infer(batch)
            pts = np.concatenate([plan_lidar, np.zeros((len(plan_lidar), 1)), np.ones((len(plan_lidar), 1))], 1)
            self.plan_world = (lidar2global @ pts.T).T[:, :2]
            self.last_info = {**info, "step": self.step, "command": name, "speed": float(speed)}
            self.meta.append(self.last_info)

        # current-frame waypoints (lidar frame) from the stored world-frame plan
        g2l = np.linalg.inv(lidar2global)
        pts = np.concatenate([self.plan_world, np.zeros((len(self.plan_world), 1)), np.ones((len(self.plan_world), 1))], 1)
        wp = (g2l @ pts.T).T[:, :2]
        # target point for the controller (route node) in the lidar frame, as the VAD agent
        tgt = np.array([near_node[0], -near_node[1], 0.0, 1.0])
        tgt_l = (g2l @ tgt)[:2]
        steer, throttle, brake, _ = self.pid.control_pid(wp, speed, tgt_l)
        if brake < 0.05:
            brake = 0.0
        if throttle > brake:
            brake = 0.0
        control = carla.VehicleControl()
        control.steer = float(np.clip(steer, -1, 1))
        control.throttle = float(np.clip(throttle, 0, 0.75))
        control.brake = float(np.clip(brake, 0, 1))
        if self.video and self.save_dir is not None and self.step % 2 == 0:
            self._render(input_data, wp, control, speed)
        if self.save_dir is not None and self.step % 100 == 0:
            json.dump(self.meta, open(self.save_dir / "meta.json", "w"))
        return control

    def _render(self, input_data, wp, control, speed):
        """Video frame: front camera with the plan projected on the ground | top-down view with plan and
        detections (pedestrians red). Plan/detections are in the nuScenes-style lidar frame."""
        import cv2
        front = np.ascontiguousarray(input_data["CAM_FRONT"][1][:, :, :3])
        P = np.array(LIDAR2IMG["CAM_FRONT"])
        pts = np.c_[np.r_[[[0.0, 0.0]], wp], np.full(len(wp) + 1, -1.84), np.ones(len(wp) + 1)]
        uvw = (P @ pts.T).T
        ok = uvw[:, 2] > 0.5
        uv = (uvw[ok, :2] / uvw[ok, 2:3]).astype(int)
        for a, b in zip(uv[:-1], uv[1:]):
            cv2.line(front, tuple(a), tuple(b), (0, 255, 0), 8)
        for q in uv:
            cv2.circle(front, tuple(q), 10, (0, 200, 0), -1)
        front = cv2.resize(front, (960, 540))
        bev = np.ascontiguousarray(input_data["BEV"][1][:, :, :3]) if "BEV" in input_data else np.zeros((BEV_PX, BEV_PX, 3), np.uint8)
        f = BEV_PX / 2 / np.tan(np.radians(BEV_FOV / 2))
        depth = BEV_Z + 1.84 * 0  # camera height above the vehicle origin (ground-level plane)

        def to_px(x_l, y_l):  # lidar (x right, y fwd; origin 0.39 m behind ego centre) -> image px
            return int(BEV_PX / 2 + f * x_l / depth), int(BEV_PX / 2 - f * (y_l - 0.39) / depth)

        for b in getattr(self, "last_info", {}).get("boxes", []):
            c, sn = np.cos(b["yaw"]), np.sin(b["yaw"])
            corners = [(b["x"] + dx * c - dy * sn, b["y"] + dx * sn + dy * c)
                       for dx, dy in [(b["l"] / 2, b["w"] / 2), (b["l"] / 2, -b["w"] / 2),
                                      (-b["l"] / 2, -b["w"] / 2), (-b["l"] / 2, b["w"] / 2)]]
            col = (0, 0, 255) if b["cls"] == "pedestrian" else (255, 200, 0)
            cv2.polylines(bev, [np.array([to_px(*p) for p in corners], np.int32)], True, col, 2)
        path = [to_px(0.0, 0.39)] + [to_px(x, y) for x, y in wp]
        for a, b in zip(path[:-1], path[1:]):
            cv2.line(bev, a, b, (0, 255, 0), 3)
        info = getattr(self, "last_info", {})
        txt = [self.label + (" | peds masked in planner" if self.mask_peds else ""),
               f"t={self.step / 20:5.1f}s  speed={speed:4.1f} m/s  cmd={info.get('command', '')}",
               f"steer={control.steer:+.2f} thr={control.throttle:.2f} brk={control.brake:.2f}",
               f"detections={info.get('n_det', 0)}  pedestrians={info.get('n_ped', 0)}"]
        canvas = np.zeros((540, 960 + BEV_PX, 3), np.uint8)
        canvas[:, :960] = front
        canvas[:, 960:] = bev
        for k, t in enumerate(txt):
            cv2.putText(canvas, t, (12, 30 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
            cv2.putText(canvas, t, (12, 30 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.imwrite(str(self.save_dir / "video" / f"{self.step:06d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])

    def _infer(self, batch):
        from sdxai import planner as PL
        m = self.model
        h = m.head
        metas = {k: v for k, v in batch.items() if k != "img"}
        fm = m.extract_feat(batch["img"])
        det, mp = h.det_head(fm, metas), h.map_head(fm, metas)
        P = PL.prepare(h.motion_plan_head, det, mp, fm, metas, h.det_head.anchor_encoder,
                       h.det_head.instance_bank.mask, h.det_head.instance_bank.anchor_handler)
        prob = det["classification"][-1][0].float().sigmoid()
        is_ped = (prob.argmax(-1) == 8) & (prob.max(-1).values > 0.3)
        iv = None
        eager = None
        if self.mask_peds and is_ped.any():
            if not hasattr(self, "_eager_ctx"):
                from sdxai.eager_attention import use_eager_attention
                self._eager_ctx = use_eager_attention(m, prefixes=["head.motion_plan_head"])
                self._eager = self._eager_ctx.__enter__()
            eager = self._eager
            iv = PL.Intervention(mask_keys=torch.where(is_ped)[0].tolist())
        outs = PL.layers(h.motion_plan_head, P, iv, eager)
        dec = PL.decode_plan(h.motion_plan_head, det, outs, metas)
        PL.commit(h.motion_plan_head, P, outs, det, metas)
        keep = prob.max(-1).values > 0.3
        anc = det["prediction"][-1][0][keep].float().cpu().numpy()
        info_boxes = [{"x": float(a[0]), "y": float(a[1]), "l": float(np.exp(a[3])), "w": float(np.exp(a[4])),
                       "yaw": float(np.arctan2(a[6], a[7])), "cls": CLASS_NAMES[int(c)]}
                      for a, c in zip(anc, prob[keep].argmax(-1).tolist())]
        info = {"boxes": info_boxes, "n_ped": int(is_ped.sum()), "n_det": int(keep.sum()),
                "mode": int(dec["mode_idx"]), "plan": dec["final"].float().cpu().numpy().tolist()}
        return dec["final"].float().cpu().numpy(), info

    def destroy(self):
        if self.save_dir is not None:
            json.dump(self.meta, open(self.save_dir / "meta.json", "w"))
            if self.video:
                import glob
                import cv2
                frames = sorted(glob.glob(str(self.save_dir / "video" / "*.jpg")))
                if frames:
                    make_mp4(self.save_dir / "video", self.save_dir / "drive.mp4")
        del self.model
        torch.cuda.empty_cache()
