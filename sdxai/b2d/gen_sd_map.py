"""World-frame map geometry for one CARLA town, from which convert_b2d.py derives SparseDrive's three
map classes per frame with SparseDrive's own nuScenes extraction utilities (map_utils/utils.py):
- divider: lane markings from Bench2Drive's public tools/gen_hdmap.py (imported, unmodified) with a
  painted type (Broken/Solid/SolidSolid/...); 'Center' lines and unpainted edges are not dividers.
  gen_hdmap records a shared marking once per adjacent lane (~57% near-duplicates on Town03) and in
  short pieces, so lines are de-duplicated (>= 80% within 0.3 m of a longer kept line), merged end to
  end (shapely linemerge) and pieces < 1 m dropped.
- road: road-surface polygons, one strip per Driving-lane segment (junctions included), widened outwards
  over adjacent Shoulder/Parking/Bidirectional lanes. As nuScenes' road_segment + lane union, their union
  is the drivable area whose contour gives the boundaries (get_drivable_area_contour).
- ped_crossing: CARLA crosswalk polygons (Map.get_crosswalks()). On the large towns (12/13/15) that list is
  empty: their crosswalks are OpenDRIVE objects named '*Crosswalk*' with type "-1", which CARLA's parser
  skips. Their <outline> corners are then placed from the road's reference line, obtained from a CARLA lane
  waypoint at s moved back by the lane-centre offset (laneOffset polynomial + half lane width).
Coordinates are CARLA world (x, y) - left-handed; the converter flips y exactly like Zoo's prepare_B2D.
Output: <out>/<Town>_sd_map.npz with object arrays 'divider' (lines), 'road', 'ped_crossing' (rings)."""
import argparse
import os
SDXAI_ROOT = os.environ.get("SDXAI_ROOT", os.getcwd())  # workspace with SparseDrive/, nuscenes/, navsim_exp/, b2d/
import sys
import time
from collections import defaultdict

import carla
import numpy as np

sys.path.insert(0, os.path.join(SDXAI_ROOT, "b2d/Bench2Drive/tools"))
from gen_hdmap import LankMarkingGettor  # noqa: E402  (upstream, unmodified)

PAINTED = {"Broken", "Solid", "SolidSolid", "SolidBroken", "BrokenSolid", "BrokenBroken", "BottsDots"}


def crosswalks(cmap):
    pts, polys, cur = cmap.get_crosswalks(), [], []
    for p in pts:  # polygons are listed back-to-back, each closed by repeating its first vertex
        cur.append((p.x, p.y))
        if len(cur) > 2 and np.allclose(cur[0], cur[-1], atol=1e-3):
            polys.append(np.array(cur))
            cur = []
    return polys


def xodr_crosswalks(cmap):
    import math
    import xml.etree.ElementTree as ET
    root = ET.fromstring(cmap.to_opendrive())
    out = []
    for road in root.iter("road"):
        objs = [o for o in road.iter("object") if "crosswalk" in o.get("name", "").lower()
                and o.find("outline") is not None]
        if not objs:
            continue
        rid = int(road.get("id"))
        offs = sorted((float(o.get("s")), [float(o.get(k)) for k in "abcd"]) for o in road.iter("laneOffset"))

        def lane_offset(s):
            s0, (a, b, c, d) = ([x for x in offs if x[0] <= s] or [(0.0, [0.0] * 4)])[-1]
            ds = s - s0
            return a + b * ds + c * ds ** 2 + d * ds ** 3
        for o in objs:
            s, t, hdg = float(o.get("s")), float(o.get("t")), float(o.get("hdg"))
            ref = None
            for lane in (-1, 1):
                w = cmap.get_waypoint_xodr(rid, lane, s)
                if w is None:
                    continue
                # OpenDRIVE frame (y = -y_carla); road heading (positive lanes drive against it)
                yaw = -math.radians(w.transform.rotation.yaw) + (math.pi if lane > 0 else 0.0)
                nx, ny = -math.sin(yaw), math.cos(yaw)  # left normal
                tc = lane_offset(s) + (-0.5 if lane < 0 else 0.5) * w.lane_width
                x, y = w.transform.location.x, -w.transform.location.y
                ref = (x - tc * nx, y - tc * ny, yaw, nx, ny)
                break
            if ref is None:
                continue
            x0, y0, yaw, nx, ny = ref
            cx, cy, h = x0 + t * nx, y0 + t * ny, yaw + hdg
            pts = [(cx + float(c.get("u")) * math.cos(h) - float(c.get("v")) * math.sin(h),
                    cy + float(c.get("u")) * math.sin(h) + float(c.get("v")) * math.cos(h))
                   for c in o.find("outline").iter("cornerLocal")]
            if len(pts) > 2:
                out.append(np.array([(px, -py) for px, py in pts]))  # back to CARLA (left-handed) world
    return out


def roads(cmap, step=1.0):
    D = carla.LaneType.Driving
    EXT = {carla.LaneType.Shoulder, carla.LaneType.Parking, carla.LaneType.Bidirectional}

    def extent(w, get):
        e, nb = 0.5 * w.lane_width, get(w)
        while nb is not None and nb.lane_type in EXT:
            e += nb.lane_width
            nb = get(nb)
        return e

    strips = defaultdict(list)
    for w in cmap.generate_waypoints(step):
        if w.lane_type != D:
            continue
        loc, r = w.transform.location, w.transform.get_right_vector()
        el = extent(w, lambda x: x.get_left_lane())
        er = extent(w, lambda x: x.get_right_lane())
        strips[(w.road_id, w.section_id, w.lane_id)].append(
            (w.s, loc.x - el * r.x, loc.y - el * r.y, loc.x + er * r.x, loc.y + er * r.y))
    out = []
    for pts in strips.values():
        pts.sort()
        a = np.array([p[1:] for p in pts])
        cut = np.where(np.linalg.norm(np.diff(a[:, :2], axis=0), axis=1) > 3 * step)[0] + 1
        for seg in np.split(a, cut):
            if len(seg) > 1:  # left edge forward, right edge back: a closed ring
                out.append(np.concatenate([seg[:, :2], seg[::-1, 2:], seg[:1, :2]]))
    return out


def dividers(cmap):
    d = {}
    LankMarkingGettor.get_lanemarkings(cmap, d, precision=0.5)  # upstream default 0.05 m: hours on the large towns
    out = []
    for road in d.values():
        for lane_id, lanes in road.items():
            if lane_id == "Trigger_Volumes":
                continue
            for ln in lanes:
                if ln["Type"] in PAINTED and len(ln["Points"]) > 1:
                    out.append(np.array([p[0][:2] for p in ln["Points"]]))
    return out


def clean_dividers(lines, tol=0.3, cover=0.8, min_len=1.0):
    from shapely import ops
    from shapely.geometry import LineString
    from shapely.strtree import STRtree
    L = sorted((LineString(g) for g in lines if len(g) > 1), key=lambda x: -x.length)
    B = [ln.buffer(tol, resolution=2) for ln in L]
    tree, idx = STRtree(B), {id(g): i for i, g in enumerate(B)}
    keep = np.zeros(len(L), bool)
    for i, ln in enumerate(L):
        if not any(keep[idx[id(o)]] and o.intersection(ln).length >= cover * ln.length
                   for o in tree.query(ln)):
            keep[i] = True
    merged = ops.linemerge([ln for ln, k in zip(L, keep) if k])
    merged = list(merged.geoms) if merged.geom_type == "MultiLineString" else [merged]
    return [np.array(m.coords)[:, :2] for m in merged if m.length >= min_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", required=True)
    ap.add_argument("--port", type=int, default=34000)
    ap.add_argument("--out", default=os.path.join(SDXAI_ROOT, "b2d/data/maps_sd"))
    ap.add_argument("--clean-only", action="store_true", help="re-clean dividers of an existing npz (no CARLA)")
    a = ap.parse_args()
    f = os.path.join(a.out, f"{a.town}_sd_map.npz")
    if a.clean_only:
        d = dict(np.load(f, allow_pickle=True))
        n0 = len(d["divider"])
        d["divider"] = np.array(clean_dividers(list(d["divider"])) + [None], dtype=object)[:-1]
        np.savez_compressed(f, **d)
        print(a.town, "dividers", n0, "->", len(d["divider"]))
        return
    c = carla.Client("localhost", a.port)
    c.set_timeout(600)
    t0 = time.time()
    cmap = c.load_world(a.town).get_map()
    res = {"ped_crossing": crosswalks(cmap) or xodr_crosswalks(cmap), "divider": clean_dividers(dividers(cmap)), "road": roads(cmap)}
    np.savez_compressed(f,
                        **{k: np.array(v + [None], dtype=object)[:-1] for k, v in res.items()})
    print(a.town, {k: len(v) for k, v in res.items()}, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
