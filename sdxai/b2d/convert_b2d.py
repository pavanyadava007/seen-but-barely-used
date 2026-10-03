"""Bench2Drive clips -> SparseDrive (nuScenes-format) info pkl, for fine-tuning the nuScenes checkpoint.

Keyframes: every 5th frame (10 Hz -> 2 Hz, the rate SparseDrive is trained at); futures at 0.5 s steps.
Coordinates follow Bench2DriveZoo's prepare_B2D.py (left-handed CARLA -> right-handed; nuScenes-style
lidar frame x right / y forward / z up), boxes in SparseDrive's nuScenes convention:
  gt_boxes = [x, y, z, l, w, h, yaw] with yaw = heading in the lidar frame (no -pi/2 offset).
Labels: Bench2DriveZoo NameMapping collapsed onto the checkpoint's 10 nuScenes classes so the detection
head is kept intact (van -> car; traffic_sign / traffic_light / others -> ignored).
Visibility: boxes are kept only if visible in >= 1 camera (depth-image test, same thresholds as
prepare_B2D.py). Motion targets: 12 x 0.5 s agent futures (first offset from the box centre, then
per-step deltas) by CARLA actor id; ego future: 6 x 0.5 s per-step offsets; command from the final
lateral offset exactly as the nuScenes converter (>= 2 m right, <= -2 m left, else straight).
Map annotations (--maps DIR): per keyframe from the town geometry of sdxai/b2d/gen_sd_map.py, extracted
exactly as SparseDrive's NuscMapExtractor does for nuScenes (lidar frame, roi 30 x 60 m, dividers clipped,
ped crossings merged by _union_ped then closed contours, boundaries = contour of the drivable-area union),
reusing upstream's own map_utils functions. Without --maps they are left empty (map loss must then be off).
"""
from __future__ import annotations

import argparse
import gzip
import importlib
import types
import json
import os
SDXAI_ROOT = os.environ.get("SDXAI_ROOT", os.getcwd())  # workspace with SparseDrive/, nuscenes/, navsim_exp/, b2d/
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from pyquaternion import Quaternion

ZOO = os.path.join(SDXAI_ROOT, "b2d/Bench2DriveZoo")
CAMS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
CAM_DIR = {"CAM_FRONT": "rgb_front", "CAM_FRONT_LEFT": "rgb_front_left", "CAM_FRONT_RIGHT": "rgb_front_right",
           "CAM_BACK": "rgb_back", "CAM_BACK_LEFT": "rgb_back_left", "CAM_BACK_RIGHT": "rgb_back_right"}
STEP = 5          # 10 Hz frames per 0.5 s
FUT_TS, EGO_FUT_TS = 12, 6
MAX_DISTANCE, FILTER_Z = 75, 10
NUM_VISIBLE, NUM_OUTSIDE = 1, 7
TO_NUSC = {"car": "car", "van": "car", "truck": "truck", "bus": "bus", "bicycle": "bicycle",
           "motorcycle": "motorcycle", "pedestrian": "pedestrian", "traffic_cone": "traffic_cone",
           "barrier": "barrier", "construction_vehicle": "construction_vehicle", "trailer": "trailer"}

ROI = (30, 60)  # SparseDrive roi_size: x (right) 30 m, y (forward) 60 m
MAP_DIR = None
_MAPS = {}


def _map_utils():
    """SparseDrive's map_utils without importing the whole plugin package (CUDA ops etc.)."""
    if "sd_map_utils" not in sys.modules:
        pkg = types.ModuleType("sd_map_utils")
        pkg.__path__ = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "projects",
                                     "mmdet3d_plugin", "datasets", "map_utils")]
        sys.modules["sd_map_utils"] = pkg
    return importlib.import_module("sd_map_utils.utils"), importlib.import_module("sd_map_utils.nuscmap_extractor")


def _town_map(town):
    if town not in _MAPS:
        d = np.load(os.path.join(MAP_DIR, f"{town}_sd_map.npz"), allow_pickle=True)
        # CARLA left-handed world -> right-handed (y flipped), as prepare_B2D; keep bbox for a quick cull
        _MAPS[town] = {k: [(g * [1, -1], np.r_[(g * [1, -1]).min(0), (g * [1, -1]).max(0)]) for g in d[k]]
                       for k in ("divider", "road", "ped_crossing")}
    return _MAPS[town]


def map_annos(town, world2lidar):
    from shapely import ops
    from shapely.geometry import LineString, Polygon, box
    U, X = _map_utils()
    M = _town_map(town)
    ctr = np.linalg.inv(world2lidar)[:3, 3]
    r = 0.5 * np.hypot(*ROI) + 5

    def local(k):
        for g, bb in M[k]:
            if bb[0] - r < ctr[0] < bb[2] + r and bb[1] - r < ctr[1] < bb[3] + r:
                h = np.c_[g, np.full(len(g), ctr[2]), np.ones(len(g))]
                yield (world2lidar @ h.T).T[:, :2]
    patch = box(-ROI[0] / 2, -ROI[1] / 2, ROI[0] / 2, ROI[1] / 2)
    div = []
    for g in local("divider"):
        c = LineString(g).intersection(patch)
        if not c.is_empty and c.geom_type in ("LineString", "MultiLineString", "GeometryCollection"):
            div += [x for x in U.split_collections(c) if x.geom_type == "LineString"]
    peds = []
    for g in local("ped_crossing"):
        c = Polygon(g).buffer(0).intersection(patch)
        if not c.is_empty and c.geom_type in ("Polygon", "MultiPolygon"):
            peds += U.split_collections(c)
    peds = X.NuscMapExtractor._union_ped(None, peds) if peds else []
    ped_lines = [ln for ln in (U.get_ped_crossing_contour(p, patch) for p in peds) if ln is not None]
    roads = [Polygon(g).buffer(0) for g in local("road")]
    bnd = []
    if roads:
        area = ops.unary_union(roads).intersection(patch)
        if not area.is_empty and area.geom_type in ("Polygon", "MultiPolygon"):
            bnd = U.get_drivable_area_contour(U.split_collections(area), ROI)
    return {0: [np.array(x.coords) for x in ped_lines], 1: [np.array(x.coords) for x in div],
            2: [np.array(x.coords) for x in bnd]}


stand_to_ue4_rotate = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)
lidar_to_righthand_ego = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)
lefthand_ego_to_lidar = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)
left2right = np.eye(4)
left2right[1, 1] = -1


def name_mapping():
    src = open(os.path.join(ZOO, "adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py")).read()
    s = src.index("NameMapping = {")
    e = src.index("}", s)
    ns = {}
    exec(src[s:e + 1], ns)
    return ns["NameMapping"]


def load_anno(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def cube_vertices(center, extent):
    cx, cy, cz = center
    x, y, z = extent
    return np.array([[cx + dx, cy + dy, cz + dz] for dx in (x, -x) for dy in (y, -y) for dz in (z, -z)])


def _occluded(y, x, d, dmap):
    """prepare_B2D / vis_utils.point_is_occluded: occluded iff all 4 diagonal neighbours of the
    projected vertex have a smaller depth-map value than the vertex depth (raw 8-bit value)."""
    H, W = dmap.shape
    res = []
    for dy in (1, -1):
        for dx in (1, -1):
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W:
                res.append(dmap[yy, xx] < d)
    return all(res)


def visible(center_l, extent_wlh, yaw, sensors, depth_imgs):
    """Exact replica of prepare_B2D.py's FILTER_INVISINLE test (vertices built as there, raw depth
    comparison, > NUM_VISIBLE visible and < NUM_OUTSIDE outside the 1600x900 canvas in >= 1 camera)."""
    box2lidar = np.eye(4)
    box2lidar[:3, :3] = Quaternion(axis=[0, 0, 1], radians=yaw).rotation_matrix
    box2lidar[:3, 3] = center_l
    lidar2box = np.linalg.inv(box2lidar)
    verts = []
    for v in cube_vertices(center_l, extent_wlh):
        tmp = lidar2box @ np.r_[v, 1.0]
        tmp[:3] += center_l
        verts.append(tmp[:3])
    verts = np.array(verts)
    for cam in CAMS:
        K = sensors[cam]["intrinsic"]
        lidar2cam = sensors[cam]["lidar2cam"]
        dmap = depth_imgs[cam]
        H, W = dmap.shape
        n_vis = n_out = 0
        for vtx in verts:
            pc = lidar2cam[:3, :3] @ vtx + lidar2cam[:3, 3]
            d = pc[2]
            if d <= 0:
                continue  # prepare_B2D drops vertices behind the camera before counting
            u, v = (K @ pc)[:2] / (K @ pc)[2]
            if MAX_DISTANCE > d > 0 and 0 <= v < H and 0 <= u < W:
                if _occluded(int(v), int(u), d, dmap):
                    pass
                else:
                    n_vis += 1
            else:
                n_out += 1
        if n_vis > NUM_VISIBLE and n_out < NUM_OUTSIDE:
            return True
    return False


def frame_geometry(anno):
    sensors = {}
    lidar2ego = left2right @ np.array(anno["sensors"]["LIDAR_TOP"]["lidar2ego"]) @ left2right @ lidar_to_righthand_ego
    world2lidar = lefthand_ego_to_lidar @ np.array(anno["sensors"]["LIDAR_TOP"]["world2lidar"]) @ left2right
    for cam in CAMS:
        cam2ego = left2right @ np.array(anno["sensors"][cam]["cam2ego"]) @ stand_to_ue4_rotate
        K = np.array(anno["sensors"][cam]["intrinsic"])[:3, :3]
        lidar2cam = np.linalg.inv(cam2ego) @ lidar2ego
        sensors[cam] = {"intrinsic": K, "lidar2cam": lidar2cam, "sensor2lidar": np.linalg.inv(lidar2cam)}
    ego2world = np.linalg.inv(left2right @ np.array(anno["bounding_boxes"][0]["world2ego"]) @ left2right)
    return sensors, lidar2ego, world2lidar, ego2world


def objects(anno, world2lidar, ego_yaw):
    """Per actor: id, class (raw), centre (world RH), centre (lidar), size [l,w,h], yaw (lidar)."""
    out = {}
    ego_z = anno["bounding_boxes"][0]["location"][2]
    for npc in anno["bounding_boxes"]:
        if npc["class"] == "ego_vehicle" or npc["distance"] > MAX_DISTANCE:
            continue
        if abs(npc["location"][2] - ego_z) > FILTER_Z:
            continue
        cw = np.array([npc["center"][0], -npc["center"][1], npc["center"][2]])
        cl = (world2lidar @ np.r_[cw, 1.0])[:3]
        if "world2vehicle" in npc:
            w2v = left2right @ np.array(npc["world2vehicle"]) @ left2right
            v2l = world2lidar @ np.linalg.inv(w2v)
            yaw = float(np.arctan2(v2l[1, 0], v2l[0, 0]))
        else:
            yaw = -npc["rotation"][-1] / 180 * np.pi - ego_yaw + np.pi / 2
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
        ext = npc["extent"]  # half sizes along CARLA x (length), y (width), z
        out[str(npc["id"])] = {"type": npc.get("type_id", npc["class"]), "cw": cw, "cl": cl,
                               "lwh": np.array([2 * ext[0], 2 * ext[1], 2 * ext[2]]),
                               "ext_wlh": np.array([ext[1], ext[0], ext[2]]), "yaw": yaw,
                               "speed": npc.get("speed", None), "cls": npc["class"]}
    return out


def ego_status(anno):
    s = np.zeros(10, np.float32)
    acc = anno["acceleration"]
    s[0:3] = [acc[0], -acc[1], acc[2]]
    s[3:6] = -np.asarray(anno["angular_velocity"])
    s[6] = anno["speed"]
    s[9] = anno["steer"]
    return s


def convert_clip(args):
    root, clip, nm = args
    cdir = os.path.join(root, clip)
    frames = sorted(int(f.split(".")[0]) for f in os.listdir(os.path.join(cdir, "anno")))
    annos = {i: load_anno(os.path.join(cdir, "anno", f"{i:05d}.json.gz")) for i in frames}
    fset = set(frames)
    infos = []
    town = clip.split("_")[1]
    for t in frames:
        if t % STEP or not all(os.path.exists(os.path.join(cdir, "camera", CAM_DIR[c], f"{t:05d}.jpg")) for c in CAMS):
            continue
        a = annos[t]
        sensors, lidar2ego, world2lidar, ego2world = frame_geometry(a)
        ego_yaw = -np.nan_to_num(a["theta"], nan=np.pi) + np.pi / 2
        depth = {c: cv2.imread(os.path.join(cdir, "camera", CAM_DIR[c].replace("rgb_", "depth_"), f"{t:05d}.png"))[:, :, 0]
                 for c in CAMS}
        objs = objects(a, world2lidar, ego_yaw)
        # velocities from the previous 2 Hz keyframe (as prepare_B2D does for non-vehicles)
        prev = objects(annos[t - STEP], world2lidar, ego_yaw) if (t - STEP) in fset else {}
        boxes, names, vel, ids, fut, fmask = [], [], [], [], [], []
        for oid, o in objs.items():
            cls = TO_NUSC.get(nm.get(o["type"], ""), None)
            if cls is None:
                continue
            if not visible(o["cl"], o["ext_wlh"], o["yaw"], sensors, depth):
                continue
            if oid in prev:
                v = (o["cl"][:2] - prev[oid]["cl"][:2]) / (STEP / 10.0)
            else:
                v = np.zeros(2)
            # future centres in the current lidar frame
            traj = np.zeros((FUT_TS, 2))
            mask = np.zeros(FUT_TS)
            last = o["cl"][:2]
            for k in range(1, FUT_TS + 1):
                tk = t + k * STEP
                if tk not in fset:
                    break
                fo = next((b for b in annos[tk]["bounding_boxes"] if str(b["id"]) == oid), None)
                if fo is None:
                    break
                cw = np.array([fo["center"][0], -fo["center"][1], fo["center"][2]])
                p = (world2lidar @ np.r_[cw, 1.0])[:2]
                traj[k - 1] = p - last
                last = p
                mask[k - 1] = 1
            boxes.append(np.r_[o["cl"], o["lwh"], o["yaw"]])
            names.append(cls)
            vel.append(v)
            ids.append(int(oid))
            fut.append(traj)
            fmask.append(mask)
        # ego future (lidar frame of the current keyframe)
        ego_pts = [np.zeros(2)]
        ego_mask = np.zeros(EGO_FUT_TS)
        for k in range(1, EGO_FUT_TS + 1):
            tk = t + k * STEP
            if tk not in fset:
                ego_pts.append(ego_pts[-1])
                continue
            e2w = np.linalg.inv(left2right @ np.array(annos[tk]["bounding_boxes"][0]["world2ego"]) @ left2right)
            lidar_now_to_world = ego2world @ lidar2ego
            p = np.linalg.inv(lidar_now_to_world) @ e2w @ lidar2ego @ np.array([0, 0, 0, 1.0])
            ego_pts.append(p[:2])
            ego_mask[k - 1] = 1
        ego_pts = np.array(ego_pts)
        fx = ego_pts[-1, 0]
        cmd = np.array([1, 0, 0]) if fx >= 2 else (np.array([0, 1, 0]) if fx <= -2 else np.array([0, 0, 1]))
        l2e_q = Quaternion(matrix=lidar2ego[:3, :3])
        e2g_q = Quaternion(matrix=ego2world[:3, :3], atol=1e-6)
        cams = {}
        for c in CAMS:
            s2l = sensors[c]["sensor2lidar"]
            cams[c] = {"data_path": os.path.join(cdir, "camera", CAM_DIR[c], f"{t:05d}.jpg"), "type": c,
                       "sensor2lidar_rotation": s2l[:3, :3], "sensor2lidar_translation": s2l[:3, 3],
                       "cam_intrinsic": sensors[c]["intrinsic"]}
        n = len(boxes)
        infos.append({
            # upstream marks a new temporal sequence by an empty sweep list (never loaded here)
            "lidar_path": "", "token": f"{clip}/{t:05d}", "sweeps": [] if not infos else ["prev_keyframe"],
            "scene_token": clip,
            "lidar2ego_translation": lidar2ego[:3, 3].tolist(), "lidar2ego_rotation": list(l2e_q),
            "ego2global_translation": ego2world[:3, 3].tolist(), "ego2global_rotation": list(e2g_q),
            "timestamp": int(t * 1e5), "map_location": town,
            "map_annos": map_annos(town, world2lidar) if MAP_DIR else {0: [], 1: [], 2: []},
            "cams": cams,
            "gt_boxes": np.array(boxes, np.float32).reshape(n, 7), "gt_names": np.array(names),
            "gt_velocity": np.array(vel, np.float32).reshape(n, 2),
            "num_lidar_pts": np.ones(n, np.int64), "num_radar_pts": np.zeros(n, np.int64),
            "valid_flag": np.ones(n, bool), "instance_inds": ids,
            "gt_agent_fut_trajs": np.array(fut, np.float32).reshape(n, FUT_TS, 2),
            "gt_agent_fut_masks": np.array(fmask, np.float32).reshape(n, FUT_TS),
            "gt_ego_fut_trajs": np.diff(ego_pts, axis=0).astype(np.float32),
            "gt_ego_fut_masks": ego_mask.astype(np.float32), "gt_ego_fut_cmd": cmd.astype(np.float32),
            "ego_status": ego_status(a),
        })
    return infos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(SDXAI_ROOT, "b2d/data/v1"))
    ap.add_argument("--clips", required=True, help="json file with {split_name: [clip, ...]}")
    ap.add_argument("--key", default="pilot_train")
    ap.add_argument("--val-frac", type=float, default=0.1, help="held-out clips (by clip) for a sanity val set")
    ap.add_argument("--out", default=os.path.join(SDXAI_ROOT, "b2d/data/infos_sd"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--maps", default=None, help="dir with <Town>_sd_map.npz (gen_sd_map.py); omit = empty maps")
    a = ap.parse_args()
    global MAP_DIR
    MAP_DIR = a.maps
    os.makedirs(a.out, exist_ok=True)
    clips = [c for c in json.load(open(a.clips))[a.key] if os.path.exists(os.path.join(a.root, c, ".done"))]
    nm = name_mapping()
    rng = np.random.RandomState(0)
    val = set(rng.choice(clips, max(1, int(a.val_frac * len(clips))), replace=False))
    res = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for c, inf in zip(clips, ex.map(convert_clip, [(a.root, c, nm) for c in clips])):
            res[c] = inf
    # The dataset sorts infos by timestamp: give every clip a disjoint time window (1000 s apart,
    # > the instance bank's 2 s max interval, so temporal state resets between clips).
    for k, c in enumerate(clips):
        for i in res[c]:
            i["timestamp"] += int(k * 1e9)
    for split, sel in [("train", [c for c in clips if c not in val]), ("val", sorted(val))]:
        infos = [i for c in sel for i in res[c]]
        infos.sort(key=lambda x: (x["scene_token"], x["timestamp"]))
        pickle.dump({"infos": infos, "metadata": {"version": "b2d"}},
                    open(os.path.join(a.out, f"b2d_infos_{split}.pkl"), "wb"))
        n_obj = sum(len(i["gt_names"]) for i in infos)
        print(split, "clips", len(sel), "keyframes", len(infos), "objects", n_obj,
              "peds", int(sum(np.sum(np.asarray(i["gt_names"], dtype=str) == "pedestrian") for i in infos)))


if __name__ == "__main__":
    main()
