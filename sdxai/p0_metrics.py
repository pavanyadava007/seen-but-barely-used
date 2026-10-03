"""Parse the metrics dict printed by upstream tools/test.py and write it as JSON.

Usage:
  python -m sdxai.p0_metrics test.log metrics.json
  python -m sdxai.p0_metrics --compare a.json b.json summary.json
"""
import ast
import json
import math
import sys

KEY_METRICS = {
    "NDS": "img_bbox_NuScenes/NDS",
    "mAP": "img_bbox_NuScenes/mAP",
    "AP_ped_dist2.0": "img_bbox_NuScenes/pedestrian_AP_dist_2.0",
    "AMOTA": "img_bbox_NuScenes/amota",
    "mAP_map": "mAP_normal",
    "EPA_ped": "pedestrian_EPA",
    "plan_L2_avg": "L2",
    "plan_col_avg": "obj_box_col",
}
# Released stage-2 checkpoint, full val ("Results of released checkpoint" table in upstream README).
README_FULL_VAL = {"NDS": 0.5257, "mAP_map": 0.5656, "AMOTA": 0.372, "plan_L2_avg": 0.61, "plan_col_avg": 0.00097}


def parse_log(path: str) -> dict:
    last = None
    for line in open(path):
        line = line.strip()
        if line.startswith("{'img_bbox_NuScenes/") or line.startswith("{'ped_crossing'"):
            last = line
    if last is None:
        raise RuntimeError(f"no metrics dict found in {path}")
    last = last.replace("nan", "None")
    full = ast.literal_eval(last)
    return {"key": {k: full.get(v) for k, v in KEY_METRICS.items()}, "full": full}


def main(argv):
    if argv[0] == "--compare":
        a, b = (json.load(open(p)) for p in argv[1:3])
        diff = {k: (a["key"][k] - b["key"][k]) for k in a["key"]
                if isinstance(a["key"][k], float) and isinstance(b["key"][k], float)}
        summary = {
            "run1": a["key"], "run2": b["key"], "run_to_run_abs_diff": diff,
            "readme_full_val": README_FULL_VAL,
            "note": "mini-val (81 samples, 2 scenes) is not comparable to README full-val numbers. "
                    "run1/run2 are identical invocations; their spread is the nondeterminism floor "
                    "from atomicAdd accumulation in the deformable-aggregation CUDA kernel.",
        }
        json.dump(summary, open(argv[3], "w"), indent=2)
        print(json.dumps(summary, indent=2))
        return
    m = parse_log(argv[0])
    json.dump(m, open(argv[1], "w"), indent=2, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)
    print(json.dumps(m["key"], indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
