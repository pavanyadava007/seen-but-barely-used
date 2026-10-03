"""NAVSIM complete group removal vs. the unchanged planner: paired per-scene differences of EPDMS and of the
no-at-fault-collision sub-score, 95 % CIs from a log-clustered bootstrap (2,000 resamples). Writes navsim_groups_ci.json."""
import glob
import json
import os
SDXAI_ROOT = os.environ.get("SDXAI_ROOT", os.getcwd())  # workspace with SparseDrive/, nuscenes/, navsim_exp/, b2d/
import pickle

import numpy as np
import pandas as pd

E = os.path.join(SDXAI_ROOT, "navsim_exp")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "navsim_groups_ci.json")


def load(k):
    d = pd.read_csv(sorted(glob.glob(f"{E}/score_navtest38_{k}/*.csv"))[-1])
    return d[d.token.str.len() == 16].set_index("token")


meta = pickle.load(open(f"{E}/sd_navtest38_none.pkl", "rb"))["meta"]
base, res = load("none"), {}
for k in ("full_peds", "full_vehicles", "full_agents"):
    d = load(k)
    t = base.index.intersection(d.index)
    df = pd.DataFrame({"ds": d.loc[t, "score"] - base.loc[t, "score"],
                       "dnc": d.loc[t, "no_at_fault_collisions"] - base.loc[t, "no_at_fault_collisions"],
                       "log": [meta[x]["log"] for x in t]})
    logs = df.log.unique()
    g = {lg: df[df.log == lg] for lg in logs}
    rng = np.random.RandomState(0)
    bs = np.array([(s.ds.mean(), s.dnc.mean()) for s in
                   (pd.concat([g[lg] for lg in rng.choice(logs, len(logs))]) for _ in range(2000))])
    res[k] = {"n": len(df), "logs": len(logs), "d_epdms": df.ds.mean(),
              "lo": np.percentile(bs[:, 0], 2.5), "hi": np.percentile(bs[:, 0], 97.5),
              "d_nc": df.dnc.mean(), "nc_lo": np.percentile(bs[:, 1], 2.5), "nc_hi": np.percentile(bs[:, 1], 97.5),
              "nc_fixed": int((df.dnc > 0).sum()), "nc_new": int((df.dnc < 0).sum())}
json.dump(res, open(OUT, "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
