"""Phase 5 summary: rank-AUC tables with CIs, intervention effect tables, sanity, acceptance, plots."""
from __future__ import annotations

import argparse
import json
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from sdxai import stats as ST
from sdxai.instances import GT_RANGE
from sdxai.runner import REPO

RANKERS = ["attn", "grad", "physics", "random"]
EFFECT = "mask_dl2_avg"
TOP_FRAC = 0.2


def attach_strata(ag: pd.DataFrame, inst: pd.DataFrame, version: str) -> pd.DataFrame:
    """GT-match pedestrian agents (replay slots) to val GT pedestrians -> Phase 2 strata."""
    infos = pickle.load(open(os.path.join(REPO, "data/infos", "mini" if version == "mini" else "",
                                          "nuscenes_infos_val.pkl"), "rb"))["infos"]
    val = inst[(inst.split == "val") & inst.gt_idx.notna()].set_index(["frame", "gt_idx"])
    cols = ["critical", "critical_w8", "in_corridor_w8", "dist_bin", "vis_bin", "gt_dist", "visibility", "ttc_est"]
    out = []
    # frame indices follow the dataset order (sorted by timestamp), not the pkl order: look up by sample token
    by_token = {x["token"]: x for x in infos}
    for f, d in ag.groupby("frame"):
        info = by_token[d.sample_token.iloc[0]]
        names, boxes = info["gt_names"], info["gt_boxes"]
        pts = info["num_lidar_pts"] + info["num_radar_pts"]
        rng = np.linalg.norm(boxes[:, :2], axis=-1)
        gidx = np.where((names == "pedestrian") & (pts > 0) & (rng <= GT_RANGE))[0]
        d = d.copy()
        for c in cols + ["gt_idx"]:
            d[c] = np.nan
        peds = d[d.is_ped]
        if len(peds) and len(gidx):
            D = np.linalg.norm(peds[["pred_x", "pred_y"]].values[:, None] - boxes[None, gidx, :2], axis=-1)
            r, c = linear_sum_assignment(D)
            for rr, cc in zip(r, c):
                if D[rr, cc] <= 2.0 and (f, float(gidx[cc])) in val.index:
                    row = val.loc[(f, float(gidx[cc]))]
                    row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
                    ix = peds.index[rr]
                    d.loc[ix, "gt_idx"] = gidx[cc]
                    for col in cols:
                        d.loc[ix, col] = row[col]
        out.append(d)
    return pd.concat(out)


def frame_level_auc(ag: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f, d in ag.groupby("frame"):
        for r in RANKERS:
            rows.append({"frame": f, "ranker": r, "auc": ST.rank_auc(d[f"score_{r}"].values, d[EFFECT].values, TOP_FRAC),
                         "spearman": spearmanr(d[f"score_{r}"], d[EFFECT]).correlation})
    return pd.DataFrame(rows)


def clusters(d: pd.DataFrame) -> np.ndarray:
    """Bootstrap cluster per row: scene if the data has enough scenes (trainval), else scene:frame (mini)."""
    return ST.cluster_unit(d)[0].values


def pooled_auc_ci(d: pd.DataFrame, ranker: str, n_boot: int = 1000, seed: int = 0) -> dict:
    """AUC over pooled agents of a stratum: positives = top-20 % masking effect (effect > 0).
    CI: cluster bootstrap over scenes (frames when there are too few scenes, i.e. on mini)."""
    est = ST.rank_auc(d[f"score_{ranker}"].values, d[EFFECT].values, TOP_FRAC)
    frames = d.frame.unique()
    cl = clusters(d)
    keys = np.unique(cl)
    groups = {k: d[cl == k] for k in keys}
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(keys, len(keys), replace=True)
        s = pd.concat([groups[k] for k in pick])
        boots.append(ST.rank_auc(s[f"score_{ranker}"].values, s[EFFECT].values, TOP_FRAC))
    boots = np.array(boots, float)
    return {"auc": est, "lo": float(np.nanpercentile(boots, 2.5)), "hi": float(np.nanpercentile(boots, 97.5)),
            "n_agents": len(d), "n_frames": len(frames), "n_pos": int((d[EFFECT] >= np.sort(d[EFFECT])[::-1][max(1, int(np.ceil(TOP_FRAC * len(d)))) - 1]).sum())}


def effect_table(df: pd.DataFrame, prefix: str, group: pd.Series, label: str) -> pd.DataFrame:
    rows = []
    for g, d in df.groupby(group):
        if f"{prefix}_dl2_avg" not in d or d[f"{prefix}_dl2_avg"].notna().sum() == 0:
            continue
        cl = clusters(d)
        row = {"intervention": label, "group": g, "n": int(d[f"{prefix}_dl2_avg"].notna().sum())}
        for m in ["dl2_1s", "dl2_2s", "dl2_3s", "dl2_avg", "mode_changed", "col_flip", "d_ttc_plan",
                  "d_min_clear_ped", "d_entropy", "d_rescore_penalty"]:
            c = f"{prefix}_{m}"
            if c in d:
                ci = ST.cluster_bootstrap(d[c].values, cl)
                row[m], row[m + "_lo"], row[m + "_hi"] = ci["est"], ci["lo"], ci["hi"]
        flips = d.get(f"{prefix}_col_flip")
        if flips is not None:
            k = int(np.nansum(flips))
            row["col_flips"] = k
            row["col_flip_rule_of_three_ub"] = ST.rule_of_three(int(flips.notna().sum())) if k == 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="work_dirs/xai/interventions")
    ap.add_argument("--instances", default="work_dirs/xai/instances/instances.parquet")
    ap.add_argument("--version", default=os.environ.get("SDXAI_VERSION", "mini"))
    a = ap.parse_args()
    d = a.dir
    inst = pd.read_parquet(a.instances)
    ag = attach_strata(pd.read_parquet(os.path.join(d, "planner_mask.parquet")), inst, a.version)
    agr = pd.read_parquet(os.path.join(d, "planner_mask_randomised.parquet"))
    ag.to_parquet(os.path.join(d, "interventions.parquet"), index=False)

    # frame-level AUC
    fl = frame_level_auc(ag)
    flr = frame_level_auc(agr)
    fl_sum = []
    for tag, t in [("trained", fl), ("planner_randomised", flr)]:
        for r, x in t.groupby("ranker"):
            ci = ST.cluster_bootstrap(x.auc.values, clusters(x))
            sp = ST.cluster_bootstrap(x.spearman.values, clusters(x))
            fl_sum.append({"model": tag, "ranker": r, "auc": ci["est"], "lo": ci["lo"], "hi": ci["hi"],
                           "n_frames": ci["n"], "spearman": sp["est"]})
    fl_sum = pd.DataFrame(fl_sum)
    # pedestrian strata (pooled)
    peds = ag[ag.is_ped]
    strata = {"all_agents": ag, "ped_all": peds,
              "ped_critical": peds[peds.critical == True],
              "ped_critical_w8": peds[peds.critical_w8 == True],
              "ped_near(<15)": peds[peds.dist_bin == "<15"],
              "ped_occluded(1-2)": peds[peds.vis_bin == "occluded(1-2)"],
              "ped_visible(3-4)": peds[peds.vis_bin == "visible(3-4)"]}
    st_rows = []
    for sname, sd in strata.items():
        if len(sd) < 10:
            continue
        for r in RANKERS:
            st_rows.append({"stratum": sname, "ranker": r, **pooled_auc_ci(sd, r)})
    st_df = pd.DataFrame(st_rows)
    # randomised planner, same strata definition for all agents / peds
    for sname, sd in {"all_agents": agr, "ped_all": agr[agr.is_ped]}.items():
        for r in RANKERS:
            st_rows.append({"stratum": sname + "|planner_randomised", "ranker": r, **pooled_auc_ci(sd, r)})
    st_df = pd.DataFrame(st_rows)
    fl_sum.to_csv(os.path.join(d, "auc_frame_level.csv"), index=False)
    st_df.to_csv(os.path.join(d, "auc_table.csv"), index=False)

    # effects
    eff = pd.concat([
        effect_table(ag, "mask", ag.is_ped.map({True: "pedestrian", False: "non-pedestrian"}), "mask_keys"),
        effect_table(peds, "mean", peds.critical_w8.map({True: "ped critical_w8", False: "ped other"}).fillna("ped unmatched"), "mean_feature"),
        effect_table(peds, "drop", peds.critical_w8.map({True: "ped critical_w8", False: "ped other"}).fillna("ped unmatched"), "drop_from_topk"),
        effect_table(peds, "mask", peds.critical_w8.map({True: "ped critical_w8", False: "ped other"}).fillna("ped unmatched"), "mask_keys"),
    ])
    eff.to_csv(os.path.join(d, "effects_table.csv"), index=False)
    # relocation / rescore
    extra = {}
    rel_path = os.path.join(d, "anchor_relocate.parquet")
    if os.path.exists(rel_path):
        rel = pd.read_parquet(rel_path)
        rel = rel.dropna(subset=["swap_dl2_avg"])
        geo = np.maximum(rel.into_dl2_avg.values, rel.out_dl2_avg.values)
        extra["relocation"] = {
            "n_agents": len(rel),
            "into_dl2_avg": ST.cluster_bootstrap(rel.into_dl2_avg.values, clusters(rel)),
            "out_dl2_avg": ST.cluster_bootstrap(rel.out_dl2_avg.values, clusters(rel)),
            "swap_dl2_avg": ST.cluster_bootstrap(rel.swap_dl2_avg.values, clusters(rel)),
            "geometry_minus_appearance": ST.paired_cluster_bootstrap_diff(geo, rel.swap_dl2_avg.values, clusters(rel)),
            "wilcoxon_geometry_vs_appearance": ST.paired_wilcoxon(geo, rel.swap_dl2_avg.values),
            "mode_change_rate": {k: float(rel[f"{k}_mode_changed"].mean()) for k in ["into", "out", "swap"]},
        }
    rs_path = os.path.join(d, "rescore_ablate.parquet")
    if os.path.exists(rs_path):
        rs = pd.read_parquet(rs_path)
        rf = pd.read_parquet(os.path.join(d, "rescore_frames.parquet"))
        extra["rescore"] = {
            "frames": len(rf), "base_reproduced_all": bool(rf.base_reproduced.all()),
            "frames_with_any_penalty": int((rf.n_penalised_modes > 0).sum()),
            "frames_where_rescore_changes_selection": int(rf.rescore_changes_selection.sum()),
            "ped_agents_considered": len(rs),
            "ped_agents_changing_any_penalty": int((rs.n_modes_penalty_changed > 0).sum()) if len(rs) else 0,
            "ped_agents_changing_selection": int(rs.drop_mode_changed.sum()) if len(rs) else 0,
            "mean_dl2_avg_when_dropped": float(rs.drop_dl2_avg.mean()) if len(rs) else float("nan"),
        }
    # random-instance masking check (risk table)
    extra["masking_effect_by_class"] = ag.groupby("pred_class")[EFFECT].agg(["mean", "median", "count"]).round(5).to_dict()
    extra["frac_agents_zero_effect(<1e-4 m)"] = float((ag[EFFECT] < 1e-4).mean())

    # acceptance
    from sdxai.instances import analysis_stratum
    cname = "ped_" + analysis_stratum()["flag"]
    crit = st_df[st_df.stratum == cname].set_index("ranker")
    use = cname if len(crit) and crit.n_agents.min() >= 10 else "ped_all"
    t = st_df[st_df.stratum == use].set_index("ranker")
    acc = {"stratum_used": use,
           "attn": t.loc["attn"][["auc", "lo", "hi"]].to_dict(),
           "physics": t.loc["physics"][["auc", "lo", "hi"]].to_dict(),
           "random": t.loc["random"][["auc", "lo", "hi"]].to_dict(),
           "attn_gt_physics_nonoverlap": bool(t.loc["attn", "lo"] > t.loc["physics", "hi"]),
           "attn_gt_random_nonoverlap": bool(t.loc["attn", "lo"] > t.loc["random", "hi"]),
           "randomised_planner_attn_auc_all_agents": st_df[(st_df.stratum == "all_agents|planner_randomised")
                                                           & (st_df.ranker == "attn")].auc.iloc[0],
           "ci_unit": ST.cluster_unit(ag)[1], **extra}
    acc["pass"] = acc["attn_gt_physics_nonoverlap"] and acc["attn_gt_random_nonoverlap"]
    json.dump(acc, open(os.path.join(d, "p5_acceptance.json"), "w"), indent=2, default=float)

    # plots
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for i, (sname, g) in enumerate(st_df[~st_df.stratum.str.contains("randomised")].groupby("stratum", sort=False)):
        for j, r in enumerate(RANKERS):
            row = g[g.ranker == r]
            if len(row):
                row = row.iloc[0]
                ax[0].errorbar(i + (j - 1.5) * 0.18, row.auc, yerr=[[row.auc - row.lo], [row.hi - row.auc]],
                               fmt="o", color=f"C{j}", label=r if i == 0 else None, capsize=2)
    ax[0].axhline(0.5, color="k", lw=0.6, ls="--")
    labels = list(st_df[~st_df.stratum.str.contains("randomised")].stratum.unique())
    ax[0].set_xticks(range(len(labels)))
    ax[0].set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    ax[0].set_ylabel("rank-AUC (top-20 % masking effect)")
    ax[0].legend(fontsize=7)
    ax[1].hist(np.log10(ag[EFFECT].clip(lower=1e-6)), bins=50, alpha=0.6, label="trained")
    ax[1].hist(np.log10(agr[EFFECT].clip(lower=1e-6)), bins=50, alpha=0.6, label="planner randomised")
    ax[1].set_xlabel("log10 masking effect dl2_avg [m]")
    ax[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(d, "auc_ci.png"), dpi=120)
    plt.close()
    print(fl_sum.round(3).to_string(index=False))
    print(st_df.round(3).to_string(index=False))
    print(eff[["intervention", "group", "n", "dl2_avg", "dl2_avg_lo", "dl2_avg_hi", "mode_changed", "col_flips"]].round(4).to_string(index=False))
    print(json.dumps(acc, indent=2, default=float))


if __name__ == "__main__":
    main()
