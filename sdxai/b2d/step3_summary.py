"""Summarise step-3 closed-loop runs (Bench2Drive leaderboard result.json) -> eval_out/step3_summary.{json,md}.
Agents: zs = released nuScenes checkpoint, e3 = pilot fine-tune without map labels, map = pilot fine-tune
with CARLA HD-map labels. Pedestrian routes are run twice (r1, r2) to expose CARLA run-to-run variation.
Success = route completed (score_route 100) with no penalty (score_penalty 1), as Bench2Drive's success rate."""
import glob
import json
import os
SDXAI_ROOT = os.environ.get("SDXAI_ROOT", os.getcwd())  # workspace with SparseDrive/, nuscenes/, navsim_exp/, b2d/
from collections import defaultdict

import numpy as np

E = os.path.join(SDXAI_ROOT, "b2d/eval_out")
AG = {"pzs": "zs", "pe3": "e3", "pmp": "map", "zsv": "zs", "ftv": "e3", "mpv": "map"}


def load(tag):
    f = os.path.join(E, tag, "result.json")
    if not os.path.exists(f):
        return None
    recs = json.load(open(f))["_checkpoint"]["records"]
    if not recs:
        return None
    r = recs[0]
    video = glob.glob(os.path.join(E, tag, "*", "drive.mp4"))
    return {"status": r["status"], "ds": r["scores"]["score_composed"], "rc": r["scores"]["score_route"],
            "success": r["scores"]["score_route"] >= 100 and r["scores"]["score_penalty"] >= 1,
            "ped_coll": len(r["infractions"]["collisions_pedestrian"]),
            "veh_coll": len(r["infractions"]["collisions_vehicle"]),
            "layout_coll": len(r["infractions"]["collisions_layout"]),
            "scenario": r["scenario_name"], "video": os.path.relpath(video[0], E) if video else None}


def agg(rs):
    rs = [r for r in rs if r]
    if not rs:
        return {"n": 0}
    return {"n": len(rs), "DS": round(float(np.mean([r["ds"] for r in rs])), 2),
            "RC": round(float(np.mean([r["rc"] for r in rs])), 2),
            "success_rate": round(float(np.mean([r["success"] for r in rs])), 3),
            "ped_collisions": sum(r["ped_coll"] for r in rs), "veh_collisions": sum(r["veh_coll"] for r in rs),
            "layout_collisions": sum(r["layout_coll"] for r in rs)}


def boot_ci(d, n=10000, seed=0):
    d = np.asarray(d, float)
    if len(d) < 2:
        return None
    rng = np.random.RandomState(seed)
    m = rng.choice(d, (n, len(d))).mean(1)
    return [round(float(np.percentile(m, 2.5)), 2), round(float(np.percentile(m, 97.5)), 2)]


def main():
    ped = sorted({os.path.basename(x).split("_")[1]: os.path.basename(x).split("_")[3].replace(".xml", "")
                  for x in glob.glob(os.path.join(E, "..", "eval_routes_ped", "*.xml"))}.items())
    dev = sorted(os.path.basename(x).split("_")[1] for x in glob.glob(os.path.join(E, "..", "eval_routes", "*.xml")))
    R = {a: {(rid, rep): load(f"{p}_r{rep}_{rid}") for rid, _ in ped for rep in (1, 2)}
         for p, a in [("pzs", "zs"), ("pe3", "e3"), ("pmp", "map")]}
    out = {"pedestrian_routes": {a: agg(list(v.values())) for a, v in R.items()}}
    types = sorted({t for _, t in ped})
    out["pedestrian_by_type"] = {t: {a: agg([R[a][(rid, rep)] for rid, tt in ped if tt == t for rep in (1, 2)])
                                     for a in R} for t in types}
    out["rep_variation"] = {}
    for a in R:
        d = [abs(R[a][(rid, 1)]["ds"] - R[a][(rid, 2)]["ds"]) for rid, _ in ped if R[a][(rid, 1)] and R[a][(rid, 2)]]
        flips = sum(R[a][(rid, 1)]["success"] != R[a][(rid, 2)]["success"] for rid, _ in ped
                    if R[a][(rid, 1)] and R[a][(rid, 2)])
        out["rep_variation"][a] = {"pairs": len(d), "mean_abs_DS_diff": round(float(np.mean(d)), 2) if d else None,
                                   "success_flips": int(flips)}
    out["paired_DS_diff"] = {}
    for a, b in [("map", "zs"), ("map", "e3"), ("e3", "zs")]:
        d = [R[a][k]["ds"] - R[b][k]["ds"] for k in R[a] if R[a][k] and R[b][k]]
        out["paired_DS_diff"][f"{a}-{b}"] = {"n": len(d), "mean": round(float(np.mean(d)), 2) if d else None,
                                             "ci95": boot_ci(d)}
    out["dev10"] = {a: agg([load(f"{p}_{rid}") for rid in dev]) for p, a in [("zsv", "zs"), ("ftv", "e3"), ("mpv", "map")]}
    out["runs"] = {f"{a}_r{k[1]}_{k[0]}": v for a, vv in R.items() for k, v in vv.items() if v}
    json.dump(out, open(os.path.join(E, "step3_summary.json"), "w"), indent=1)
    md = ["# Step 3 closed loop (Bench2Drive, CARLA 0.9.15)", "",
          "| set | agent | n | DS | RC | success | ped coll | veh coll | layout coll |", "|---|---|---|---|---|---|---|---|---|"]
    for s in ("pedestrian_routes", "dev10"):
        for a, v in out[s].items():
            if v["n"]:
                md.append(f"| {s} | {a} | {v['n']} | {v['DS']} | {v['RC']} | {v['success_rate']} | "
                          f"{v['ped_collisions']} | {v['veh_collisions']} | {v['layout_collisions']} |")
    md += ["", "Paired DS differences (pedestrian routes, route x repetition):", ""]
    md += [f"- {k}: mean {v['mean']} (95% bootstrap CI {v['ci95']}, n={v['n']})" for k, v in out["paired_DS_diff"].items()]
    md += ["", "Repetition variation (same agent, same route, r1 vs r2):", ""]
    md += [f"- {a}: mean |dDS| {v['mean_abs_DS_diff']}, success flips {v['success_flips']} / {v['pairs']}"
           for a, v in out["rep_variation"].items()]
    open(os.path.join(E, "step3_summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
