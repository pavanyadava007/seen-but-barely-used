"""Compare NAVSIM one-stage scores of SparseDrive with and without pedestrian interventions.
Writes navsim_exp/navsim_interventions.json (+ .md): all scenes and scenes where >=1 pedestrian was detected."""
import json
import os
SDXAI_ROOT = os.environ.get("SDXAI_ROOT", os.getcwd())  # workspace with SparseDrive/, nuscenes/, navsim_exp/, b2d/
import pickle

import pandas as pd

E = os.path.join(SDXAI_ROOT, "navsim_exp")
COLS = ["no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound",
        "ego_progress", "score"]


def load(iv):
    d = [f for f in os.listdir(f"{E}/score_navtest38_{iv}") if f.endswith(".csv")]
    df = pd.read_csv(f"{E}/score_navtest38_{iv}/{sorted(d)[-1]}")
    return df[df["token"].str.len() == 16].set_index("token")  # drop aggregate row(s)


def main():
    meta = pickle.load(open(f"{E}/sd_navtest38_none.pkl", "rb"))["meta"]
    ped = {t for t, m in meta.items() if m["n_ped"] > 0}
    res, dfs = {}, {iv: load(iv) for iv in ["none", "mask_peds", "drop_peds"]}
    common = set.intersection(*[set(d.index) for d in dfs.values()])
    for iv, df in dfs.items():
        for name, toks in [("all", common), ("ped_scenes", common & ped)]:
            sub = df.loc[sorted(toks), [c for c in COLS if c in df.columns]]
            res.setdefault(iv, {})[name] = {"n": len(sub), **{c: round(float(sub[c].mean()), 4) for c in sub}}
    base = dfs["none"].loc[sorted(common & ped)]
    for iv in ["mask_peds", "drop_peds"]:
        d = dfs[iv].loc[sorted(common & ped)]
        res[iv]["ped_scenes"]["new_collisions_vs_none"] = int(((base["no_at_fault_collisions"] == 1) & (d["no_at_fault_collisions"] < 1)).sum())
        res[iv]["ped_scenes"]["fixed_collisions_vs_none"] = int(((base["no_at_fault_collisions"] < 1) & (d["no_at_fault_collisions"] == 1)).sum())
    json.dump(res, open(f"{E}/navsim_interventions.json", "w"), indent=1)
    rows = [f"| {iv} | {s} | " + " | ".join(str(v) for v in r.values()) + " |" for iv, x in res.items() for s, r in x.items()]
    open(f"{E}/navsim_interventions.md", "w").write("\n".join(rows) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
