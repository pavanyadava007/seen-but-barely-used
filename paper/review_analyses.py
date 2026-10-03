"""Analyses requested by the review, computed from the saved trainval results (no new model runs).
Writes review_results.json (read by build_numbers.py) and prints a summary."""
import glob
import json
import os
SDXAI_ROOT = os.environ.get("SDXAI_ROOT", os.getcwd())  # workspace with SparseDrive/, nuscenes/, navsim_exp/, b2d/
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import binomtest, spearmanr

R = SDXAI_ROOT
X = f"{R}/SparseDrive_tv/work_dirs/xai"
sys.path.insert(0, f"{R}/SparseDrive_tv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "review_results.json")
res = {}
rng = np.random.RandomState(0)


def boot(df, val, cluster="scene_token", stat=np.mean, n=1000):
    g = {k: d[val].values for k, d in df.groupby(cluster)}
    keys = list(g)
    est = float(stat(df[val].values))
    bs = [stat(np.concatenate([g[k] for k in rng.choice(keys, len(keys))])) for _ in range(n)]
    return [est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def boot_diff(df, val, flag, cluster="scene_token", n=1000):
    """difference of means between flag==1 and flag==0, scene-clustered."""
    g = {k: d for k, d in df.groupby(cluster)}
    keys = list(g)
    f = lambda d: d[d[flag] == 1][val].mean() - d[d[flag] == 0][val].mean()  # noqa: E731
    bs = [f(pd.concat([g[k] for k in rng.choice(keys, len(keys))])) for _ in range(n)]
    return [float(f(df)), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]


inst = pd.read_parquet(f"{X}/instances/instances.parquet")
val = inst[inst.split == "val"]

# ---------------------------------------------------------------- planner: classes, matched, tail
pm = pd.read_parquet(f"{X}/interventions/planner_mask.parquet")
pm = pm[pm.tag == "trained"].copy()
pm["cm"] = 100 * pm.mask_dl2_avg
res["per_class"] = {c: {"n": int(len(d)), "mean_cm": float(d.cm.mean()), "median_cm": float(d.cm.median())}
                    for c, d in pm.groupby("pred_class")}
GTCOLS = ["gt_class", "gt_dist", "lateral_to_path", "in_corridor_w8", "critical_w8"]


def posmatch(ag, max_d=2.0):
    """Link planner agents to GT objects by position (Hungarian, <= max_d m) within the same sample. The planner run
    and the instance table are separate runs, so their query slots differ: joining on slot == pred_idx mislinks rows."""
    gt = val[val.gt_idx.notna()].drop_duplicates(["sample_token", "gt_idx"])
    gts = {t: d for t, d in gt.groupby("sample_token")}
    out = []
    for t, d in ag.groupby("sample_token"):
        d = d.copy()
        for c in GTCOLS + ["gt_match_dist"]:
            d[c] = np.nan
        g = gts.get(t)
        if g is not None and len(d):
            D = np.linalg.norm(d[["pred_x", "pred_y"]].values[:, None] - g[["gt_x", "gt_y"]].values[None], axis=-1)
            r, c = linear_sum_assignment(D)
            for rr, cc in zip(r, c):
                if D[rr, cc] <= max_d:
                    for col in GTCOLS:
                        d.iloc[rr, d.columns.get_loc(col)] = g.iloc[cc][col]
                    d.iloc[rr, d.columns.get_loc("gt_match_dist")] = D[rr, cc]
        out.append(d)
    return pd.concat(out)


pmg = posmatch(pm)
# how many rows the old slot join linked to a GT object more than 0.5 m from the planner agent
tp_old = val[val.kind == "tp"][["frame", "pred_idx", "gt_class", "gt_x", "gt_y"]].assign(pred_idx=lambda d: d.pred_idx.astype(int))
old = pm.merge(tp_old, left_on=["frame", "slot"], right_on=["frame", "pred_idx"])
old["off"] = np.hypot(old.pred_x - old.gt_x, old.pred_y - old.gt_y)
res["slot_join_audit"] = {c: {"n": int((old.gt_class == c).sum()), "frac_gt_0p5m": float((old[old.gt_class == c].off > 0.5).mean())}
                          for c in ("pedestrian", "car")}
j = pmg[pmg.gt_class.isin(["pedestrian", "car"])].copy()
j["in_corridor_w8"] = j.in_corridor_w8.astype(bool)
j["is_ped"] = (j.gt_class == "pedestrian").astype(int)
# coarsened exact matching on distance, lateral offset, confidence, corridor
j["dbin"] = pd.cut(j.gt_dist, [0, 10, 20, 30, 50], labels=False)
j["lbin"] = pd.cut(j.lateral_to_path.clip(0, 50), [-.1, 2.5, 5, 10, 20, 51], labels=False)
j["sbin"] = pd.cut(j.pred_score, [0, .4, .6, .8, 1.01], labels=False)
j["stratum"] = j[["dbin", "lbin", "sbin", "in_corridor_w8"]].astype(str).agg("-".join, axis=1)
ok = j.groupby("stratum").is_ped.transform(lambda s: s.nunique() == 2)
m = j[ok]
w = m.groupby("stratum").apply(lambda d: pd.Series({"d": d[d.is_ped == 1].cm.mean() - d[d.is_ped == 0].cm.mean(),
                                                     "r": d[d.is_ped == 1].cm.mean() / max(d[d.is_ped == 0].cm.mean(), 1e-6),
                                                     "n": len(d)}))
sc = {k: d for k, d in m.groupby("scene_token")}


def matched_stat(df):
    ww = df.groupby("stratum").apply(lambda d: pd.Series({
        "p": d[d.is_ped == 1].cm.mean(), "c": d[d.is_ped == 0].cm.mean(), "n": len(d)})).dropna()
    return np.average(ww.p, weights=ww.n), np.average(ww.c, weights=ww.n)


p_m, c_m = matched_stat(m)
bs = []
for _ in range(500):
    d = pd.concat([sc[k] for k in rng.choice(list(sc), len(sc))])
    d = d[d.groupby("stratum").is_ped.transform(lambda s: s.nunique() == 2)]
    a, b = matched_stat(d)
    bs.append(a - b)
res["matched_ped_vs_car"] = {"n_ped": int(m.is_ped.sum()), "n_car": int((m.is_ped == 0).sum()), "n_strata": int(m.stratum.nunique()),
                             "ped_cm": float(p_m), "car_cm": float(c_m), "diff_cm": float(p_m - c_m),
                             "diff_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                             "raw_ped_cm": float(j[j.is_ped == 1].cm.mean()), "raw_car_cm": float(j[j.is_ped == 0].cm.mean())}
peds = pm[pm.is_ped]
res["tail"] = {"n": int(len(peds)), "median_cm": float(peds.cm.median()), "p99_cm": float(np.percentile(peds.cm, 99)),
               "max_m": float(peds.cm.max() / 100), "n_gt_50cm": int((peds.cm > 50).sum()),
               "n_gt_74cm": int((peds.cm > 74.5).sum()), "n_mode_switch": int(peds.mask_mode_changed.sum()),
               "mean_cm_mode_switch": float(peds[peds.mask_mode_changed == 1].cm.mean()),
               "max_cm_no_switch": float(peds[peds.mask_mode_changed == 0].cm.max()),
               "max_frame": int(peds.loc[peds.cm.idxmax(), "frame"])}
ms = pmg[pmg.is_ped & (pmg.mask_mode_changed == 1)]
res["mode_switch_cases"] = {"n": int(len(ms)), "gt_matched": int(ms.gt_class.notna().sum()),
                            "median_dist_m": float(ms.gt_dist.median()) if ms.gt_dist.notna().any() else None,
                            "in_corridor": int(ms.in_corridor_w8.fillna(False).astype(bool).sum()),
                            "attn_rank_median": float(ms.rank_in_topk.median())}
au = pd.read_csv(f"{X}/interventions/auc_table.csv").set_index(["stratum", "ranker"])
res["randomised_like_for_like"] = {"all_trained": float(au.loc[("all_agents", "attn"), "auc"]),
                                   "all_random": float(au.loc[("all_agents|planner_randomised", "attn"), "auc"]),
                                   "ped_trained": float(au.loc[("ped_all", "attn"), "auc"]),
                                   "ped_random": float(au.loc[("ped_all|planner_randomised", "attn"), "auc"])}
cw = pmg[pmg.is_ped & (pmg.gt_class == "pedestrian") & (pmg.critical_w8 == True)]  # noqa: E712
res["physics_corr_corridor"] = {"spearman_phys_effect": float(spearmanr(cw.score_physics, cw.cm).correlation),
                                "spearman_phys_attn": float(spearmanr(cw.score_physics, cw.score_attn).correlation), "n": int(len(cw))}

# ---------------------------------------------------------------- class confusion and recall (val only)
pmar = pd.read_parquet(f"{X}/concepts/class_directions/ped_margins.parquet")
res["confusion"] = {s: {"n": int(len(d)), "rate": float(d.confused.mean())} for s, d in pmar.groupby("split")}
vp = val[val.gt_class == "pedestrian"]
res["val_ped_recall"] = float((vp.kind == "tp").mean())

# ---------------------------------------------------------------- temporal: strata, interaction, first appearance
tr = pd.read_parquet(f"{X}/temporal/temporal_recall_rows.parquet")
tr["occ"] = (tr.visibility <= 2).astype(int)
res["memory"] = {}
for tag in ["full", "history0", "history1", "history2", "history3"]:
    d = tr[tr.tag == tag]
    res["memory"][tag] = {"occluded": boot(d[d.occ == 1], "detected"), "visible": boot(d[d.occ == 0], "detected")}
pv = tr[tr.tag.isin(["full", "history0"])].pivot_table(index=["frame", "gt_idx", "scene_token", "occ"], columns="tag", values="detected").reset_index()
pv["loss"] = pv.full - pv.history0
res["memory"]["loss_did"] = boot_diff(pv, "loss", "occ")
res["memory"]["rel_loss"] = {o: float(1 - pv[pv.occ == o].history0.mean() / pv[pv.occ == o].full.mean()) for o in (0, 1)}
# first appearance: frames since the GT instance first appears (within range) in the val sequence
vv = val[val.gt_class == "pedestrian"][["frame", "gt_idx", "gt_instance_token", "scene_token"]].dropna()
vv = vv.sort_values("frame")
vv["k_seen"] = vv.groupby("gt_instance_token").cumcount()
fa = tr[tr.tag == "full"].merge(vv, on=["frame", "gt_idx", "scene_token"])
fa["kbin"] = pd.cut(fa.k_seen, [-1, 0, 1, 3, 1000], labels=["0", "1", "2-3", ">=4"])
res["first_appearance"] = {f"{o}_{k}": boot(d, "detected") + [int(len(d))]
                           for (o, k), d in fa.groupby(["occ", "kbin"]) if len(d) >= 20}
# a pedestrian "first seen" in the first keyframe of a scene is not newly appearing (the scene just started):
# repeat without scene starts, and test the occlusion x first-sighting interaction with a scene bootstrap
fis = val[["frame", "frame_in_scene"]].drop_duplicates("frame")
fa = fa.merge(fis, on="frame")
fn = fa[fa.frame_in_scene > 0]
res["first_appearance_nostart"] = {f"{o}_{k}": boot(d, "detected") + [int(len(d))]
                                   for (o, k), d in fn.groupby(["occ", "kbin"]) if len(d) >= 20}


def fa_inter(d):
    g = d[d.kbin.isin(["0", ">=4"])].groupby(["occ", "kbin"], observed=True).detected.mean()
    return (g.get((1, "0"), np.nan) - g.get((1, ">=4"), np.nan)) - (g.get((0, "0"), np.nan) - g.get((0, ">=4"), np.nan))


for nm, d in (("all", fa), ("nostart", fn)):
    sc_ = {k: x for k, x in d.groupby("scene_token")}
    bs_ = [fa_inter(pd.concat([sc_[k] for k in rng.choice(list(sc_), len(sc_))])) for _ in range(1000)]
    res[f"fa_interaction_{nm}"] = [float(fa_inter(d)), float(np.nanpercentile(bs_, 2.5)), float(np.nanpercentile(bs_, 97.5))]

# ---------------------------------------------------------------- concepts: matched occlusion TCAV, moving vs predicted speed
from sdxai.concepts.data import gather  # noqa: E402
from sdxai.concepts import ped_concepts as PC  # noqa: E402
from sdxai.concepts.tcav import fit_cav  # noqa: E402
os.chdir(f"{R}/SparseDrive_tv")
lab = pd.read_parquet(f"{X}/concepts/ped_concepts/labels.parquet")
G = np.load(f"{X}/concepts/tcav/grads.npz")["L6"]
trn = lab[lab.split == "train"].reset_index(drop=True)
Xf = gather(trn, 6)
okr = np.isfinite(Xf).all(1) & (np.abs(Xf).sum(1) > 0)
trn, Xf = trn[okr].reset_index(drop=True), Xf[okr]
trn["occ"] = (trn.visibility <= 2).astype(int)
trn["hb"] = pd.qcut(trn.box_h, 4, labels=False, duplicates="drop")
trn["db"] = pd.cut(trn.gt_dist, [0, 10, 20, 30, 50], labels=False)
tc = {}
for name, sel in {"raw": None, "matched": "match", "residualised": "resid"}.items():
    if sel == "match":  # equal numbers of occluded / visible per distance x box-height cell
        idx = []
        for _, d in trn.groupby(["db", "hb"]):
            a_, b_ = d[d.occ == 1].index.values, d[d.occ == 0].index.values
            k = min(len(a_), len(b_))
            if k:
                idx += list(rng.choice(a_, k, replace=False)) + list(rng.choice(b_, k, replace=False))
        dd, XX = trn.loc[idx], Xf[idx]
    elif sel == "resid":
        C = np.c_[trn.gt_dist, trn.gt_dist ** 2, trn.box_h, trn.box_h ** 2]
        from sklearn.linear_model import LinearRegression
        XX = Xf - LinearRegression().fit(C, Xf).predict(C)
        dd = trn
    else:
        dd, XX = trn, Xf
    Xp, Xn = XX[(dd.occ == 1).values], XX[(dd.occ == 0).values]
    cs, rs = [], []
    for _ in range(30):
        v, _ = fit_cav(Xp, Xn, rng)
        cs.append(float((G @ v > 0).mean()))
        pool = np.r_[Xp, Xn]
        perm = rng.permutation(len(pool))
        vr, _ = fit_cav(pool[perm[:len(Xp)]], pool[perm[len(Xp):]], rng)
        rs.append(float((G @ vr > 0).mean()))
    from scipy.stats import ttest_ind
    tc[name] = {"tcav": float(np.mean(cs)), "random": float(np.mean(rs)), "p": float(ttest_ind(cs, rs, equal_var=False).pvalue),
                "n_pos": int(len(Xp)), "n_neg": int(len(Xn))}
res["occlusion_tcav_controlled"] = tc

anc = gather(lab, 6, what="layer_anchor") if False else None  # layer_anchor stored for all slots; read below
lab2 = lab.copy()
A = gather(lab2, 6, what="layer_anchor")
lab2["pred_speed"] = np.hypot(A[:, 8], A[:, 9])
okm = np.isfinite(lab2.pred_speed)
lab2 = lab2[okm].reset_index(drop=True)
Xa = gather(lab2, 6)
oka = np.isfinite(Xa).all(1) & (np.abs(Xa).sum(1) > 0)
lab2, Xa = lab2[oka].reset_index(drop=True), Xa[oka]
fp, fn, _, _ = PC.concept_table()["moving_vs_standing"]
pos, neg = fp(lab2).fillna(False).astype(bool), fn(lab2).fillna(False).astype(bool)
sub = lab2[pos ^ neg].copy()
sub["y"] = pos[pos ^ neg].astype(int).values
Xs = Xa[(pos ^ neg).values]
mv = {}
for cname, conf in {"gt_speed": ["gt_speed", "gt_dist", "visibility"], "pred_speed": ["pred_speed", "gt_dist", "visibility"]}.items():
    r_ = PC.evaluate(sub, Xs, PC.conf_matrix(sub, conf))
    mv[cname] = {k: r_[k] for k in ("bacc_feat", "bacc_conf", "bacc_resid", "bacc_shuf", "bacc_resid_ci")}
res["moving_confound"] = mv

pr = json.load(open(f"{X}/concepts/ped_concepts/probes.json"))
strict = {}
for c in pr:
    if c.get("bacc_resid") is None or (isinstance(c.get("bacc_resid"), float) and np.isnan(c["bacc_resid"])):
        strict[c["concept"]] = "too few"
        continue
    lo = c["bacc_resid_ci"][0]
    floor = max(0.5, c["bacc_shuf"]) + 0.05
    strict[c["concept"]] = {"resid": c["bacc_resid"], "resid_lo": lo, "shuf": c["bacc_shuf"], "floor": floor,
                            "verdict": "beyond confounds" if lo > floor else ("weak" if lo > max(0.5, c["bacc_shuf"]) else "not beyond")}
res["concepts_strict"] = strict

# ---------------------------------------------------------------- CARLA route-clustered
cs_ = json.load(open(f"{R}/b2d/eval_out/step3_summary.json"))
runs = pd.DataFrame([{"key": k, **v} for k, v in cs_["runs"].items()])
runs["agent"] = runs.key.str.split("_").str[0]
runs["route"] = runs.key.str.split("_").str[-1]
runs["succ"] = runs.success.astype(float)
rm = runs.groupby(["agent", "route"])[["ds", "rc", "ped_coll", "veh_coll", "layout_coll", "succ"]].mean().reset_index()
carla = {"per_agent": {a: {c: float(d[c].mean()) for c in ["ds", "rc", "ped_coll", "veh_coll", "layout_coll", "succ"]}
                       for a, d in rm.groupby("agent")}}
for a_, b_ in [("map", "zs"), ("e3", "zs"), ("map", "e3")]:
    A_ = rm[rm.agent == a_].set_index("route")
    B_ = rm[rm.agent == b_].set_index("route")
    common = A_.index.intersection(B_.index)
    out = {}
    for c in ["ds", "rc", "ped_coll"]:
        d = (A_.loc[common, c] - B_.loc[common, c]).values
        bsd = [rng.choice(d, len(d)).mean() for _ in range(10000)]
        out[c] = [float(d.mean()), float(np.percentile(bsd, 2.5)), float(np.percentile(bsd, 97.5))]
    carla[f"{a_}-{b_}"] = out
zs = runs[runs.agent == "zs"]
carla["zs_ped_coll_events"] = int(zs.ped_coll.sum())
carla["zs_runs_with_ped_coll"] = int((zs.ped_coll > 0).sum())
carla["n_runs_per_agent"] = int(len(zs))
carla["per_scenario"] = {f"{a}|{s}": {"runs": int(len(d)), "rc": float(d.rc.mean()), "ped_coll_runs": int((d.ped_coll > 0).sum()),
                                      "ped_coll_events": int(d.ped_coll.sum())}
                         for (a, s), d in runs.assign(scen=runs.scenario.str.split("_").str[0]).groupby(["agent", "scen"])}
res["carla"] = carla

# ---------------------------------------------------------------- NAVSIM: sign tests, collision fraction, funnel
nv = json.load(open(f"{R}/navsim_exp/navsim_interventions.json"))
res["navsim_sign"] = {k: {"fixed": nv[k]["ped_scenes"]["fixed_collisions_vs_none"], "new": nv[k]["ped_scenes"]["new_collisions_vs_none"],
                          "p": float(binomtest(nv[k]["ped_scenes"]["fixed_collisions_vs_none"],
                                               nv[k]["ped_scenes"]["fixed_collisions_vs_none"] + nv[k]["ped_scenes"]["new_collisions_vs_none"]).pvalue)}
                      for k in ("mask_peds", "drop_peds")}
f = sorted(glob.glob(f"{R}/navsim_exp/score_navtest38_none/*.csv"))[-1]
sc0 = pd.read_csv(f)
sc0 = sc0[sc0.token.str.len() == 16]
import pickle  # noqa: E402
meta = pickle.load(open(f"{R}/navsim_exp/sd_navtest38_none.pkl", "rb"))["meta"]
pedtok = {t for t, mm in meta.items() if mm["n_ped"] > 0}
s_p = sc0[sc0.token.isin(pedtok)]
res["navsim_collision_fraction"] = {"ped_scenes": int(len(s_p)), "any_at_fault": float((s_p.no_at_fault_collisions < 1).mean()),
                                    "all_scenes_any": float((sc0.no_at_fault_collisions < 1).mean())}
res["navsim_funnel"] = {"logs_with_sensors": 38, "cached": 2129, "inferred": 1934, "scored_tokens": 1717,
                        "logs_scored": len({mm["log"] for mm in meta.values()}), "results": int(len(sc0))}

# ---------------------------------------------------------------- attribution facts
p3 = json.load(open(f"{X}/attribution/p3_acceptance.json"))
res["mprt_last_stage"] = {k: v["last_valid_stage"] for k, v in p3["mprt_verdict"].items()}
res["mprt_last_ssim"] = {k: v["last_ssim"] for k, v in p3["mprt_verdict"].items()}

json.dump(res, open(OUT, "w"), indent=1, default=float)
print(json.dumps({k: res[k] for k in ["matched_ped_vs_car", "tail", "mode_switch_cases", "confusion", "val_ped_recall",
                                      "occlusion_tcav_controlled", "moving_confound", "first_appearance"]}, indent=1, default=float)[:6000])
print(json.dumps({"memory_did": res["memory"]["loss_did"], "rel": res["memory"]["rel_loss"], "carla": {k: v for k, v in carla.items() if "-" in k},
                  "nav": res["navsim_sign"], "navcol": res["navsim_collision_fraction"], "strict": res["concepts_strict"]}, indent=1, default=float)[:5000])
