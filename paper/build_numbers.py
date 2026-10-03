"""Turn the experiment outputs into LaTeX for the paper: numbers.tex (one macro per reported number) and
tables/*.tex. Every value is read from a result file, never typed by hand.

Source rule: the full nuScenes v1.0-trainval run (SparseDrive_tv/work_dirs/xai) is used wherever its output
exists; otherwise the v1.0-mini run (SparseDrive/work_dirs/xai) and the value is marked with \\mini (a dagger).
status.txt lists which results are still mini. Re-run this script after the trainval chain finishes and
rebuild the PDF; nothing else changes.
"""
import glob
import json
import os
SDXAI_ROOT = os.environ.get("SDXAI_ROOT", os.getcwd())  # workspace with SparseDrive/, nuscenes/, navsim_exp/, b2d/

import numpy as np
import pandas as pd

R = SDXAI_ROOT
TV = f"{R}/SparseDrive_tv/work_dirs/xai"
MI = f"{R}/SparseDrive/work_dirs/xai"
OUT = os.path.dirname(os.path.abspath(__file__))
macros, status = [], []


def src(rel):
    """(path, is_mini) for a result file; trainval preferred."""
    if os.path.exists(os.path.join(TV, rel)):
        return os.path.join(TV, rel), False
    return os.path.join(MI, rel), True


def J(rel):
    p, m = src(rel)
    return (json.load(open(p)) if os.path.exists(p) else None), m


def T(rel):
    p, m = src(rel)
    if not os.path.exists(p):
        return None, m
    return (pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)), m


def mac(name, val, fmt="{:.3f}", mini=False):
    s = fmt.format(val) if isinstance(val, (int, float, np.floating, np.integer)) else str(val)
    tag = "\\mini" if mini else ""
    macros.append("\\newcommand{\\" + name + "}{" + s + tag + "}")


def note(what, mini):
    status.append(f"{'MINI     ' if mini else 'TRAINVAL '} {what}")


def th(n):
    """integer with a LaTeX-safe thousands separator"""
    return f"{int(n):,}".replace(",", "{,}")


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def boot(df, val, cluster="scene_token", n=1000, seed=0):
    rng_ = np.random.RandomState(seed)
    g = {k: d[val].values for k, d in df.groupby(cluster)}
    keys = list(g)
    bs = [np.mean(np.concatenate([g[k] for k in rng_.choice(keys, len(keys))])) for _ in range(n)]
    return [float(df[val].mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def write(name, body):
    open(os.path.join(OUT, "tables", name), "w").write(body)


# ---------------------------------------------------------------- P0 baseline (always trainval when present)
p0, m = J("p0_baseline/metrics_run1.json")
k = p0["key"]
for n, key, f in [("NDS", "NDS", "{:.3f}"), ("mAP", "mAP", "{:.3f}"), ("mAPmap", "mAP_map", "{:.3f}"),
                  ("AMOTA", "AMOTA", "{:.3f}"), ("PlanLtwo", "plan_L2_avg", "{:.2f}"), ("APped", "AP_ped_dist2.0", "{:.3f}"),
                  ("EPAped", "EPA_ped", "{:.3f}")]:
    mac(n, k[key], f, m)
mac("PlanCol", 100 * k["plan_col_avg"], "{:.2f}", m)
note("P0 baseline", m)
p0s, m0 = J("p0_baseline/p0_summary.json")
if p0s:
    note("P0 run-to-run summary", m0)

# ---------------------------------------------------------------- P2 instances / strata
p2, m = J("instances/p2_acceptance.json")
mac("NvalPed", f"{p2['val_ped_gt']:,}".replace(",", "{,}"), mini=m)
mac("NcritStrict", p2["val_ped_critical_gt"], "{:d}", m)
mac("NcritWeight", f"{p2['widened_stratum']['val_ped_gt']:,}".replace(",", "{,}"), mini=m)
note("P2 instances", m)

# ---------------------------------------------------------------- P4 class directions
cd, m = J("concepts/class_directions/class_directions.json")
if cd:
    cos = cd["cos_ped_vs"]
    cmax = max(cos, key=cos.get)
    mac("NpedTP", f"{cd['n_ped_tp']:,}".replace(",", "{,}"), mini=m)
    mac("ConfRate", 100 * cd["confused_rate"], "{:.2f}", m)
    mac("CosMax", cos[cmax], "{:.2f}", m)
    mac("CosMaxCls", cmax.replace("_", " "))
    ru = cd["runner_up_counts_confused"]
    mac("RunnerUp", max(ru, key=ru.get).replace("_", " "))
    note("P4 class directions", m)

# ---------------------------------------------------------------- P4 probes (table)
pr, m = T("concepts/probes/probe_results.csv")
names = {"class": ("object class (10)", 10), "dist_bin": ("distance bin (3)", 3), "visibility": ("visibility level (4)", 4),
         "in_corridor_w8": ("in widened corridor (2)", 2), "in_ego_corridor": ("in strict corridor (2)", 2), "moving": ("moving (2)", 2),
         "ttc<4s": ("TTC $<$ 4\\,s (2)", 2)}
rows = []
for t, (lab, k_cls) in names.items():
    r2 = pr[(pr.task == t) & (pr.layer == 2)]
    r6 = pr[(pr.task == t) & (pr.layer == 6)]
    if len(r6):
        r2, r6 = r2.iloc[0], r6.iloc[0]
        rows.append(f"{lab} & {pct(r2.bacc_scene_split)} & {pct(r6.bacc_scene_split)} & {100 / k_cls:.1f} & "
                    f"{r6.selectivity_pts:.1f} \\\\")
write("tab_probes.tex", "\n".join(rows) + "\n")
mac("ProbeN", f"{int(pr.n.iloc[0]):,}".replace(",", "{,}"), mini=m)
r6 = pr[(pr.task == "class") & (pr.layer == 6)].iloc[0]
mac("ProbeClassAcc", pct(r6.acc_scene_split), mini=m)
mac("ProbeClassSel", f"{r6.selectivity_pts:.1f}", mini=m)
rv = pr[(pr.task == "visibility") & (pr.layer == 6)].iloc[0]
mac("ProbeVisBacc", pct(rv.bacc_scene_split), mini=m)
mac("ProbesMini", "\\mini" if m else "")
note("P4 probes", m)

# ---------------------------------------------------------------- P4 SAE + Overcomplete (table)
sae, m = J("concepts/sae/sae_metrics.json")
sae = sae if isinstance(sae, list) else list(sae.values()) if isinstance(sae, dict) and "name" not in sae else [sae]
prim = [s for s in sae if s["name"] == "sae_n2048_k64"][0]
mac("SAEntrain", f"{prim['n_train']:,}".replace(",", "{,}"), mini=m)
mac("SAEev", prim["explained_variance"], "{:.3f}", m)
mac("SAEdead", 100 * prim.get("dead_frac", 0), "{:.1f}", m)
note("P4 SAE training", m)
acc, ma = T("concepts/sae_downstream/sae_acceptance.csv")
rows = []
if acc is not None:
    keep = acc[acc.model.str.match(r"^(sae_|oc_TopK_n2048_k64|oc_TopK_n8192_k64|oc_BatchTopK|oc_MP|oc_OMP|oc_RATopK|oc_PCA|oc_ICA|oc_DictLearning|oc_KMeans|oc_ArchetypalAnalysis)")]
    for _, r in keep.iterrows():
        nm = r.model.replace("oc_", "OC-").replace("_", "\\_")
        ok = "\\cmark" if r.pass_all else "\\xmark"
        rows.append(f"{nm} & {r.EV:.3f} & {r.L0:.0f} & {r.dAP_ped_pts:+.2f} & {ok} \\\\")
    mac("NdictPass", int(acc.pass_all.sum()), "{:d}", ma)
    note("P4 dictionaries downstream (EV, L0, dAP)", ma)
write("tab_sae.tex", "\n".join(rows) + "\n")
ap, ma = T("concepts/sae_downstream/ap_ped.csv")
if ap is not None:
    mac("AblateAP", -ap.set_index("model").loc["ablate_top10_ped", "dAP_ped_pts"], "{:.1f}", ma)
    mac("SteerAPfour", ap.set_index("model").loc["add_ped_alpha4.0", "dAP_ped_pts"], "{:+.1f}", ma)
ps, ma = T("concepts/sae_downstream/plan_summary.csv")
if ps is not None:
    mac("AblatePlan", ps.set_index("tag").loc["ablate_top10_ped", "dl2_avg"], "{:.2f}", ma)
st, ma = T("concepts/sae_downstream/steering.csv")
if st is not None:
    mac("SteerMax", 100 * st[st.kind == "add"].rate.max(), "{:.1f}", ma)
    mac("SteerAblate", 100 * st[st.kind == "ablate"].rate.max(), "{:.1f}", ma)
    mac("SteerPlan", ps.set_index("tag").loc["add_ped_alpha4.0", "dl2_avg"], "{:.2f}", ma) if ps is not None else None
    note("P4 SAE ablation / steering", ma)
cat, ma = J("concepts/sae_downstream/catalogue_sae_n2048_k64.json")
if cat:
    mac("NpedUnits", cat["summary"]["n_ped_latents(purity>=0.75)"], "{:d}", ma)
    mac("UnitsAlive", th(cat["summary"]["n_alive"]), mini=ma)
    note("P4 SAE catalogue", ma)

# ---------------------------------------------------------------- P4 TCAV (table)
tc, m = T("concepts/tcav/tcav_results.csv")
rows = []
for _, r in tc.sort_values(["concept", "layer"]).iterrows():
    sig = "\\textbf{yes}" if r.significant else "no"
    pb = f"{r.p_bonferroni:.1e}".replace("e-0", "e-").replace("e-", "\\text{e-}") if r.p_bonferroni < 1 else "1"
    rows.append(f"{r.concept.replace('_', ' ')} & {int(r.layer)} & {r.tcav_mean:.2f}\\,$\\pm$\\,{r.tcav_std:.2f} & "
                f"{r.random_mean:.2f}\\,$\\pm$\\,{r.random_std:.2f} & ${pb}$ & {sig} \\\\")
write("tab_tcav.tex", "\n".join(rows) + "\n")
o6 = tc[(tc.concept == "occluded") & (tc.layer == 6)].iloc[0]
mac("TcavOcc", o6.tcav_mean, "{:.2f}", m)
mac("TcavOccRand", o6.random_mean, "{:.2f}", m)
mac("TcavOccP", f"{o6.p_bonferroni:.0e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}", mini=m)
mac("TcavN", int(o6.n_val_peds), "{:d}", m)
mac("TcavMini", "\\mini" if m else "")
note("P4 TCAV", m)

# ---------------------------------------------------------------- P3 attribution (table)
at, m = T("attribution/attr_table.csv")
p3, _ = J("attribution/p3_acceptance.json")
rc, _ = T("attribution/method_rank_correlation.csv")
mp = p3["mprt_verdict"]
names = {"dfa": "Deformable attention", "grad": "Gradient", "smoothgrad": "SmoothGrad", "grad_feat": "Grad$\\times$feature",
         "drise": "D-RISE", "random": "Random"}
cw = at[at.stratum == "critical_w8"].set_index("method")
al = at[at.stratum == "all"].set_index("method")
rows = []
for k2, lab in names.items():
    r_ = cw.loc[k2]
    mv = mp.get(k2, {})
    mprt = "--" if k2 == "random" else ("fail" if mv.get("FAIL(>0.7)") else "pass")
    rows.append(f"{lab} & {r_.morf_auc:.2f} & {r_.lerf_auc:.2f} & {al.loc[k2].pointing_game:.2f} & {al.loc[k2].energy_pg:.2f} & {mprt} \\\\")
write("tab_attr.tex", "\n".join(rows) + "\n")
mac("AttrN", p3["n_targets"], "{:d}", m)
mac("AttrNcrit", int(cw.loc["drise"].n), "{:d}", m)
mac("RoadRel", 100 * p3["relative_improvement_vs_random"], "{:.0f}", m)
rc = rc.set_index(rc.columns[0])
mac("RhoDfaGrad", rc.loc["dfa", "grad"], "{:.2f}", m)
mac("RhoGradDrise", rc.loc["grad", "drise"], "{:.2f}", m)
mac("RhoGradSmooth", rc.loc["grad", "smoothgrad"], "{:.2f}", m)
mac("AttrMini", "\\mini" if m else "")
note("P3 attribution", m)

# ---------------------------------------------------------------- P5 planner (table + macros)
p5, m = J("interventions/p5_acceptance.json")
au, _ = T("interventions/auc_table.csv")
strata = {"all_agents": "all agents", "ped_all": "pedestrians", "ped_near(<15)": "pedestrians $<$15\\,m",
          "ped_occluded(1-2)": "pedestrians, occluded", "ped_critical_w8": "pedestrians, corridor"}
rows = []
for s_, lab in strata.items():
    d = au[au.stratum == s_].set_index("ranker")
    if len(d) == 0:
        continue
    f_ = lambda r: f"{d.loc[r].auc:.2f} [{d.loc[r].lo:.2f}, {d.loc[r].hi:.2f}]"  # noqa: E731
    rows.append(f"{lab} & {th(d.loc['attn'].n_agents)} & {f_('attn')} & {f_('grad')} & {f_('physics')} & {f_('random')} \\\\")
write("tab_auc.tex", "\n".join(rows) + "\n")
for n, key in [("AucAttn", "attn"), ("AucPhys", "physics"), ("AucRand", "random")]:
    mac(n, p5[key]["auc"], "{:.2f}", m)
    mac(n + "Lo", p5[key]["lo"], "{:.2f}", m)
    mac(n + "Hi", p5[key]["hi"], "{:.2f}", m)
crit = au[(au.stratum == "ped_critical_w8") & (au.ranker == "grad")].iloc[0]
mac("AucGrad", crit.auc, "{:.2f}", m)
mac("AucRandPlanner", p5["randomised_planner_attn_auc_all_agents"], "{:.2f}", m)
ef, _ = T("interventions/effects_table.csv")
e_ = ef.set_index(["intervention", "group"])
r_ = e_.loc[("mask_keys", "ped critical_w8")]
mac("MaskCritCm", 100 * r_.dl2_avg, "{:.1f}", m); mac("MaskCritLo", 100 * r_.dl2_avg_lo, "{:.1f}", m)
mac("MaskCritHi", 100 * r_.dl2_avg_hi, "{:.1f}", m)
r_ = e_.loc[("mask_keys", "pedestrian")]
mac("MaskPedCm", 100 * r_.dl2_avg, "{:.1f}", m)
mac("MaskPedN", f"{int(r_.n):,}".replace(",", "{,}"), mini=m)
mac("MaskPedFlips", int(r_.col_flips), "{:d}", m)
mac("MaskPedFlipRate", 100 * r_.col_flips / r_.n, "{:.3f}", m)  # observed rate; rule of three only for 0 events
rl = p5.get("relocation", {})
if rl:
    for k3, n3 in [("into_dl2_avg", "RelocInto"), ("swap_dl2_avg", "RelocSwap")]:
        if k3 in rl:
            mac(n3, 100 * rl[k3]["est"], "{:.1f}", m)
mac("PlannerMini", "\\mini" if m else "")
note("P5 planner interventions", m)

# ---------------------------------------------------------------- P6 temporal
tr, m = T("temporal/temporal_recall_rows.parquet")
occ = tr[tr.visibility <= 2]
rec = occ.groupby("tag").detected.mean()
for n, tag in [("RecHnone", "history0"), ("RecHone", "history1"), ("RecHtwo", "history2"), ("RecHthree", "history3"),
               ("RecFull", "full"), ("RecRerun", "full_rerun"), ("RecDetTemp", "det_temp_gnn_zero_values")]:
    mac(n, rec[tag], "{:.2f}", m)
mac("RecNocc", th((occ.tag == "full").sum()), mini=m)
tp, _ = T("temporal/temporal_plan_rows.parquet")
g = tp.groupby("tag").dl2_avg.mean()
mac("PlanNoQueue", g["planner_zero_temp_gnn"], "{:.2f}", m)
mac("PlanHnone", g["history0"], "{:.2f}", m)
mac("PlanRerun", 100 * g["full_rerun"], "{:.1f}", m)
mac("TempMini", "\\mini" if m else "")
note("P6 temporal ablation", m)

# ---------------------------------------------------------------- P4b fine-grained pedestrian concepts
pc, m = T("concepts/ped_concepts/probes.csv")
tcp, _ = T("concepts/ped_concepts/tcav.csv")
plp, _ = T("concepts/ped_concepts/planner.csv")
lab = {"moving_vs_standing": "moving vs standing", "sitting_vs_standing": "sitting/lying vs standing",
       "worker_vs_adult": "construction worker vs adult", "crossing_vs_alongside": "crossing towards path vs alongside",
       "road_vs_walkway": "on road vs walkway", "crossing_zone_vs_walkway": "on zebra crossing vs walkway",
       "group_vs_alone": "in a group vs alone", "emerging_vs_visible": "emerging from occlusion vs visible",
       "child_vs_adult": "child vs adult", "night_vs_day": "night vs day"}
short = {"encoded beyond confounds": "beyond confounds", "encoded, but explained by confounds": "confound-explained",
         "not encoded": "not encoded", "not testable (too few examples)": "too few"}
rows = []
if pc is not None:
    tci = tcp.set_index("concept") if tcp is not None and len(tcp) else pd.DataFrame()
    for _, r in pc.iterrows():
        f2 = lambda x: "--" if pd.isna(x) else f"{x:.2f}"  # noqa: E731
        tv = "--"
        if r.concept in tci.index:
            t_ = tci.loc[r.concept]
            tv = f"{t_.tcav_mean:.2f}{'$^*$' if t_.significant else ''}"
        sv = json.load(open(os.path.join(OUT, "review_results.json")))["concepts_strict"].get(r.concept)
        verdict = sv["verdict"] if isinstance(sv, dict) else "too few"
        rows.append(f"{lab[r.concept]} & {th(r.n_pos)} & {f2(r.bacc_feat)} & {f2(r.bacc_conf)} & {f2(r.bacc_resid)} & "
                    f"{tv} & {verdict} \\\\")
    note("P4b fine-grained concepts", m)
    mac("ConceptsMini", "\\mini" if m else "")
write("tab_concepts.tex", "\n".join(rows) + "\n")

# ---------------------------------------------------------------- NAVSIM
nv = json.load(open(f"{R}/navsim_exp/navsim_interventions.json"))
rows = []
for k4, l4 in [("none", "unchanged"), ("mask_peds", "pedestrians hidden from planner"), ("drop_peds", "pedestrians removed")]:
    a_, p_ = nv[k4]["all"], nv[k4]["ped_scenes"]
    rows.append(f"{l4} & {a_['score']:.3f} & {p_['score']:.3f} & {p_['no_at_fault_collisions']:.3f} & "
                f"{p_['time_to_collision_within_bound']:.3f} & {p_.get('fixed_collisions_vs_none', '--')}/{p_.get('new_collisions_vs_none', '--')} \\\\")
write("tab_navsim.tex", "\n".join(rows) + "\n")
mac("NavN", nv["none"]["all"]["n"], "{:d}")
mac("NavNped", f"{nv['none']['ped_scenes']['n']:,}".replace(",", "{,}"))
mac("NavEPDMS", nv["none"]["all"]["score"], "{:.3f}")
mac("NavPedNone", nv["none"]["ped_scenes"]["score"], "{:.3f}")
mac("NavPedMask", nv["mask_peds"]["ped_scenes"]["score"], "{:.3f}")
mac("NavPedDrop", nv["drop_peds"]["ped_scenes"]["score"], "{:.3f}")
mac("NavMaxDelta", max(abs(nv[k]["ped_scenes"]["score"] - nv["none"]["ped_scenes"]["score"]) for k in ("mask_peds", "drop_peds")), "{:.3f}")

# ---------------------------------------------------------------- CARLA closed loop
cs_p = f"{R}/b2d/eval_out/step3_summary.json"
rows = []
if os.path.exists(cs_p):
    cs = json.load(open(cs_p))
    names_a = {"zs": "nuScenes checkpoint (zero-shot)", "e3": "+ B2D fine-tune, no map labels", "map": "+ B2D fine-tune, map labels"}
    for a_, l_ in names_a.items():
        v = cs["pedestrian_routes"].get(a_, {})
        if v.get("n"):
            rows.append(f"{l_} & {v['n']} & {v['DS']:.1f} & {v['RC']:.1f} & {100 * v['success_rate']:.1f} & {v['ped_collisions']} & {v['veh_collisions']} & {v['layout_collisions']} \\\\")
    rv = cs.get("rep_variation", {})
    mac("CarlaRepDS", np.nanmean([x["mean_abs_DS_diff"] for x in rv.values() if x.get("mean_abs_DS_diff") is not None]), "{:.1f}")
    pdd = cs.get("paired_DS_diff", {}).get("map-zs", {})
    if pdd.get("mean") is not None:
        mac("CarlaMapZs", pdd["mean"], "{:+.1f}")
        mac("CarlaMapZsLo", pdd["ci95"][0], "{:+.1f}"); mac("CarlaMapZsHi", pdd["ci95"][1], "{:+.1f}")
    ncar = sum(cs["pedestrian_routes"][a]["n"] for a in cs["pedestrian_routes"] if cs["pedestrian_routes"][a].get("n"))
    mac("CarlaNruns", ncar, "{:d}")
write("tab_carla.tex", "\n".join(rows) + "\n")


# ================================================================ review fixes (Oct 3)
RV = json.load(open(os.path.join(OUT, "review_results.json")))
f1 = lambda x: f"{x:.1f}".replace("-0.0", "0.0") if f"{x:.1f}" == "-0.0" else f"{x:.1f}"  # noqa: E731
f2 = lambda x: "0.00" if f"{x:.2f}" == "-0.00" else f"{x:.2f}"  # noqa: E731
mp_ = RV["matched_ped_vs_car"]
mac("MatchPed", f2(mp_["ped_cm"])); mac("MatchCar", f2(mp_["car_cm"])); mac("MatchDiff", f2(mp_["diff_cm"]))
mac("MatchDiffLo", f2(mp_["diff_ci"][0])); mac("MatchDiffHi", f2(mp_["diff_ci"][1]))
mac("MatchNped", th(mp_["n_ped"])); mac("MatchNcar", th(mp_["n_car"])); mac("MatchNstrata", mp_["n_strata"], "{:d}")
pc_ = RV["per_class"]
mac("ClsPedMean", f2(100 * 0 + pc_["pedestrian"]["mean_cm"])); mac("ClsPedMed", f"{pc_['pedestrian']['median_cm']:.2f}")
mac("ClsCarMean", f2(pc_["car"]["mean_cm"])); mac("ClsBusMean", f2(pc_["bus"]["mean_cm"])); mac("ClsCarMed", f2(pc_["car"]["median_cm"]))
tl = RV["tail"]
mac("TailMed", f2(tl["median_cm"])); mac("TailPnn", f1(tl["p99_cm"])); mac("TailMax", f2(tl["max_m"]))
mac("TailNfifty", tl["n_gt_50cm"], "{:d}"); mac("TailNmode", tl["n_mode_switch"], "{:d}")
mac("TailModeMean", f2(tl["mean_cm_mode_switch"] / 100)); mac("TailNoSwitchMax", f1(tl["max_cm_no_switch"]))
mac("TailModeRate", f2(100 * tl["n_mode_switch"] / tl["n"]))
msw = RV["mode_switch_cases"]
mac("ModeSwMatched", msw["gt_matched"], "{:d}"); mac("ModeSwCorr", msw["in_corridor"], "{:d}"); mac("ModeSwDist", f1(msw["median_dist_m"]))
mac("ModeSwRank", f"{msw['attn_rank_median']:.0f}")
cf = RV["confusion"]
mac("ConfVal", f2(100 * cf["val"]["rate"])); mac("ConfValN", th(cf["val"]["n"])); mac("ConfTrain", f2(100 * cf["train"]["rate"]))
mac("ValPedRecall", f2(RV["val_ped_recall"]))
rl = RV["randomised_like_for_like"]
for k_, n_ in [("all_trained", "RandAllT"), ("all_random", "RandAllR"), ("ped_trained", "RandPedT"), ("ped_random", "RandPedR")]:
    mac(n_, f2(rl[k_]))
_au = pd.read_csv(os.path.join(TV, "interventions", "auc_table.csv")).set_index(["stratum", "ranker"])
_r = _au.loc[("ped_critical", "attn")]
mac("AucStrict", f2(_r["auc"])); mac("AucStrictLo", f2(_r.iloc[1])); mac("AucStrictHi", f2(_r.iloc[2])); mac("AucStrictN", int(_r.iloc[3]), "{:d}")
mac("PhysRho", f2(RV["physics_corr_corridor"]["spearman_phys_effect"]))
ot = RV["occlusion_tcav_controlled"]
mac("OccMatched", f2(ot["matched"]["tcav"])); mac("OccMatchedR", f2(ot["matched"]["random"]))
mac("OccResid", f2(ot["residualised"]["tcav"])); mac("OccResidR", f2(ot["residualised"]["random"]))
mvc = RV["moving_confound"]["pred_speed"]
mac("MovPredResid", f2(mvc["bacc_resid"])); mac("MovPredLo", f2(mvc["bacc_resid_ci"][0])); mac("MovPredHi", f2(mvc["bacc_resid_ci"][1]))
mac("MovPredConf", f2(mvc["bacc_conf"]))
mem = RV["memory"]
for tg, n_ in [("full", "Full"), ("history0", "Hnone")]:
    for st_, s2 in [("occluded", "Occ"), ("visible", "Vis")]:
        v = mem[tg][st_]
        mac(f"Mem{n_}{s2}", f2(v[0])); mac(f"Mem{n_}{s2}Lo", f2(v[1])); mac(f"Mem{n_}{s2}Hi", f2(v[2]))
mac("MemDid", f2(mem["loss_did"][0])); mac("MemDidLo", f2(mem["loss_did"][1])); mac("MemDidHi", f2(mem["loss_did"][2]))
mac("MemRelOcc", f"{100 * mem['rel_loss']['1']:.0f}"); mac("MemRelVis", f"{100 * mem['rel_loss']['0']:.0f}")
fa = RV["first_appearance"]
for k_, n_ in [("1_0", "FaOccZero"), ("1_>=4", "FaOccFour"), ("0_0", "FaVisZero"), ("0_>=4", "FaVisFour")]:
    mac(n_, f2(fa[k_][0])); mac(n_ + "N", th(fa[k_][3]))
fan = RV["first_appearance_nostart"]
for k_, n_ in [("1_0", "FanOccZero"), ("1_>=4", "FanOccFour"), ("0_0", "FanVisZero"), ("0_>=4", "FanVisFour")]:
    mac(n_, f2(fan[k_][0])); mac(n_ + "N", th(fan[k_][3]))
for nm_, n_ in (("all", "FaInt"), ("nostart", "FanInt")):
    v = RV[f"fa_interaction_{nm_}"]
    mac(n_, f"{v[0]:+.2f}"); mac(n_ + "Lo", f"{v[1]:+.2f}"); mac(n_ + "Hi", f"{v[2]:+.2f}")
ja = RV["slot_join_audit"]
mac("JoinPed", f"{100 * ja['pedestrian']['frac_gt_0p5m']:.0f}"); mac("JoinCar", f"{100 * ja['car']['frac_gt_0p5m']:.0f}")
cr = RV["carla"]
for pair, n_ in [("map-zs", "MapZs"), ("e3-zs", "EthZs"), ("map-e3", "MapEth")]:
    for met, m2 in [("rc", "Rc"), ("ped_coll", "Pc"), ("ds", "Ds")]:
        v = cr[pair][met]
        fmt = "{:+.2f}" if met == "ped_coll" else "{:+.1f}"
        mac(f"Cr{n_}{m2}", fmt.format(v[0])); mac(f"Cr{n_}{m2}Lo", fmt.format(v[1])); mac(f"Cr{n_}{m2}Hi", fmt.format(v[2]))
mac("CrZsEvents", cr["zs_ped_coll_events"], "{:d}"); mac("CrZsRuns", cr["zs_runs_with_ped_coll"], "{:d}")
ns_ = RV["navsim_sign"]
mac("NavSignMaskP", f2(ns_["mask_peds"]["p"])); mac("NavSignDropP", f2(ns_["drop_peds"]["p"]))
mac("NavAnyFault", f"{100 * RV['navsim_collision_fraction']['any_at_fault']:.1f}")
nf = RV["navsim_funnel"]
mac("NavLogs", nf["logs_scored"], "{:d}"); mac("NavCached", th(nf["cached"])); mac("NavInferred", th(nf["inferred"]))
strict = RV["concepts_strict"]
mac("NStrictBeyond", sum(isinstance(v, dict) and v["verdict"] == "beyond confounds" for v in strict.values()), "{:d}")

# ---- planner rescore ablation (Phase 5, already in the report)
rs_ = open(os.path.join(TV, "report", "report.md")).read()
import re  # noqa: E402
mm = re.search(r"ped_agents_considered = (\d+), ped_agents_changing_any_penalty = (\d+), ped_agents_changing_selection = (\d+), mean_dl2_avg_when_dropped = ([0-9.e-]+)", rs_)
if mm:
    mac("RsN", th(int(mm.group(1)))); mac("RsPen", int(mm.group(2)), "{:d}"); mac("RsSel", int(mm.group(3)), "{:d}")
    mac("RsCm", f2(100 * float(mm.group(4))))

# ---- Phase 5b sustained / joint removal
sgp = os.path.join(TV, "interventions", "sustained_groups.parquet")
rows = []
if os.path.exists(sgp):
    sg = pd.read_parquet(sgp)
    lab = {"peds": "pedestrians", "veh": "vehicles", "agents": "all agents"}
    for g_ in ("peds", "veh", "agents"):
        cells = []
        # all three conditions on the same frames: those where the group is detected in the current frame
        # (complete removal also runs on the other frames, where only its history differs; reported separately)
        present = set(sg[sg.cond == f"{g_}_cur"].frame)
        absent = sg[(sg.cond == f"{g_}_full") & ~sg.frame.isin(present)]
        mac(f"Sg{g_.capitalize()}Absent", f2(100 * absent.dl2_avg.mean()) if len(absent) else "0")
        for c_ in ("cur", "cur_rs", "full"):
            d = sg[(sg.cond == f"{g_}_{c_}") & sg.frame.isin(present)]
            v = boot(d.assign(cm=100 * d.dl2_avg), "cm") if len(d) else [np.nan] * 3
            cells.append(f"{v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}]")
            if c_ == "full":
                mac(f"Sg{g_.capitalize()}Full", f1(v[0])); mac(f"Sg{g_.capitalize()}FullLo", f1(v[1])); mac(f"Sg{g_.capitalize()}FullHi", f1(v[2]))
                mac(f"Sg{g_.capitalize()}Mode", f1(100 * d.mode_changed.mean()))
                mac(f"Sg{g_.capitalize()}Long", f2(d.d_long_3s.mean()))
                cr_ = d.dropna(subset=["col"])
                mac(f"Sg{g_.capitalize()}Col", f1(100 * cr_.col.mean())); mac(f"Sg{g_.capitalize()}ColFlip", f1(100 * cr_.col_flip.mean()))
                mac(f"Sg{g_.capitalize()}Clear", f2(d.d_min_clear_ped.mean()))
            if c_ == "cur":
                mac(f"Sg{g_.capitalize()}Cur", f1(v[0]))
        n_fr = len(present)
        rows.append(f"{lab[g_]} & {th(n_fr)} & " + " & ".join(cells) + " \\\\")
    b0 = sg.drop_duplicates("frame").dropna(subset=["base_col"])
    mac("SgBaseCol", f1(100 * b0.base_col.mean())); mac("SgNframes", th(sg.frame.nunique()))
    cpf = sg[(sg.cond == "peds_full") & sg.has_corridor_ped]
    mac("SgPedsFullCorr", f1(100 * cpf.dl2_avg.mean())); mac("SgPedsFullCorrMode", f1(100 * cpf.mode_changed.mean()))
    mac("SgPedsPerFrame", f1(sg[sg.cond == "peds_cur"].n_peds.mean())); mac("SgVehPerFrame", f1(sg[sg.cond == "veh_cur"].n_veh.mean()))
    mac("SgPedsNfr", th(sg[sg.cond == "peds_cur"].frame.nunique()))
    note("P5b sustained removal", False)
else:
    note("P5b sustained removal PENDING", True)
write("tab_sustained.tex", "\n".join(rows) + "\n")
s2p = os.path.join(TV, "interventions", "single_corridor_2x2.parquet")
if os.path.exists(s2p):
    s2 = pd.read_parquet(s2p)
    for c_, n_ in (("attn", "SiAttn"), ("rescore", "SiRs"), ("both", "SiBoth")):
        d = s2[s2.cond == c_]
        v = boot(d.assign(cm=100 * d.dl2_avg), "cm")
        mac(n_, f1(v[0])); mac(n_ + "Lo", f1(v[1])); mac(n_ + "Hi", f1(v[2])); mac(n_ + "Mode", f1(100 * d.mode_changed.mean()))
    mac("SiN", th(s2[s2.cond == "both"].shape[0]))

# ---- NAVSIM full_* groups + constant velocity
nrows = []
for k_, l_ in (("none", "unchanged"), ("full_peds", "pedestrians removed"),
               ("full_vehicles", "vehicles removed"), ("full_agents", "all agents removed"), ("cv", "constant velocity")):
    fs = sorted(glob.glob(f"{R}/navsim_exp/score_navtest38_{k_}/*.csv"))
    if not fs:
        note(f"NAVSIM {k_} PENDING", True)
        continue
    d = pd.read_csv(fs[-1])
    d = d[d.token.str.len() == 16]
    meta_ = __import__("pickle").load(open(f"{R}/navsim_exp/sd_navtest38_none.pkl", "rb"))["meta"]
    pt = {t for t, mm_ in meta_.items() if mm_["n_ped"] > 0}
    dp = d[d.token.isin(pt)]
    nrows.append(f"{l_} & {d.score.mean():.3f} & {dp.score.mean():.3f} & {dp.no_at_fault_collisions.mean():.3f} & {dp.ego_progress.mean():.3f} \\\\")
    mac("Nav" + k_.replace("_", "").capitalize(), f"{d.score.mean():.3f}")
write("tab_navsim_groups.tex", "\n".join(nrows) + "\n")
ngp = os.path.join(OUT, "navsim_groups_ci.json")
if os.path.exists(ngp):
    for k_, v_ in json.load(open(ngp)).items():
        n_ = "NavD" + k_[5:].capitalize()
        mac(n_, f"{v_['d_epdms']:+.3f}"); mac(n_ + "Lo", f"{v_['lo']:+.3f}"); mac(n_ + "Hi", f"{v_['hi']:+.3f}")
        mac(n_ + "Nc", f"{v_['d_nc']:+.3f}"); mac(n_ + "NcLo", f"{v_['nc_lo']:+.3f}"); mac(n_ + "NcHi", f"{v_['nc_hi']:+.3f}")
        mac(n_ + "Fix", v_["nc_fixed"], "{:d}"); mac(n_ + "New", v_["nc_new"], "{:d}")

# ---- SAE controls + seeds
scp = os.path.join(TV, "concepts", "sae_controls")
if os.path.exists(os.path.join(scp, "steering_controls.csv")):
    apc = pd.read_csv(os.path.join(scp, "ap_by_class.csv"))
    def dap(tag, cls_):
        v = apc[(apc.tag == tag) & (apc["class"] == cls_)].dAP_pts
        return float(v.iloc[0]) if len(v) else np.nan
    mac("CtlPedPed", f1(dap("ablate_ped10", "pedestrian"))); mac("CtlPedCar", f1(dap("ablate_ped10", "car")))
    mac("CtlCarPed", f1(dap("ablate_car10", "pedestrian"))); mac("CtlCarCar", f1(dap("ablate_car10", "car")))
    rp_ = [dap(f"ablate_rand10_s{k}", "pedestrian") for k in range(3)]
    rc_ = [dap(f"ablate_rand10_s{k}", "car") for k in range(3)]
    mac("CtlRandPed", f2(np.mean(rp_))); mac("CtlRandCar", f2(np.mean(rc_)))
    stc = pd.read_csv(os.path.join(scp, "steering_controls.csv"))
    g_ = lambda d_, a_, c_: float(stc[(stc.dir == d_) & (stc.alpha == a_)][c_].iloc[0])  # noqa: E731
    mac("StZero", f2(100 * g_("ped", 0.0, "flip_rate"))); mac("StPedFour", f2(100 * g_("ped", 4.0, "flip_rate")))
    mac("StPedFourLogit", f2(g_("ped", 4.0, "d_ped_logit"))); mac("StRandFourLogit", f2(g_("random", 4.0, "d_ped_logit")))
    mac("StRandFour", f2(100 * g_("random", 4.0, "flip_rate")))
    mj = json.load(open(os.path.join(scp, "meta.json")))
    mac("StNormRatio", f2(mj["steer_norm_ratio_alpha1"]))
    plc = pd.read_csv(os.path.join(scp, "plan.csv")).set_index("tag")
    mac("CtlPlanPed", f1(100 * plc.loc["ablate_ped10", "dl2_avg"])); mac("CtlPlanCar", f1(100 * plc.loc["ablate_car10", "dl2_avg"]))
    mac("CtlPlanRand", f2(np.mean([plc.loc[f"ablate_rand10_s{k}", "dl2_avg"] for k in range(3)])))
    mac("CtlPlanRandAll", ", ".join(f"{100 * plc.loc[f'ablate_rand10_s{k}', 'dl2_avg']:.1f}" for k in range(3)), "{}")
    note("P4c SAE controls", False)
else:
    note("P4c SAE controls PENDING", True)
ssp = os.path.join(OUT, "sae_seed_stability.json")
if os.path.exists(ssp):
    ss = json.load(open(ssp))
    mac("SeedAll", f2(ss["mean_cos_all"])); mac("SeedPed", f2(ss["mean_cos_ped"]))
    mac("SeedPedMatched", ss["ped_matched_09"], "{:d}"); mac("SeedPedN", ss["n_ped"], "{:d}")
    mac("SeedFigUnits", ss["fig_units_matched"], "{:d}")
    scp = os.path.join(OUT, "sae_seed_concepts.json")
    if os.path.exists(scp):
        sc = json.load(open(scp))
        mac("SeedNped", "/".join(str(sc["n_ped"][k]) for k in ("0", "1", "2")), "{}")
        mac("SeedDir", f2(sc["dir_cos"])); mac("SeedSub", f2(sc["subspace_cos"])); mac("SeedSubRand", f2(sc["subspace_rand"]))
    note("P4 SAE seed stability", False)
else:
    note("P4 SAE seed stability PENDING", True)

open(os.path.join(OUT, "numbers.tex"), "w").write(
    "% generated by build_numbers.py - do not edit\n" + "\n".join(macros) + "\n")
import re as _re
used = set(_re.findall(r"\\([A-Z][A-Za-z]+)", open(os.path.join(OUT, "main.tex")).read()))
defined = set(_re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", "\n".join(macros)))
missing = sorted(used - defined - {"RQ", "Cref", "Delta", "IEEEPARstart", "Phi", "Sigma", "Theta"})
open(os.path.join(OUT, "pending.tex"), "w").write(
    "% placeholders for results still running (build_numbers.py)\n" +
    "".join("\\providecommand{\\%s}{\\textbf{??}}\n" % m_ for m_ in missing))
if missing:
    status.append("PENDING  macros: " + " ".join(missing))
open(os.path.join(OUT, "status.txt"), "w").write("\n".join(status) + "\n")
print("\n".join(status))
print(f"{len(macros)} macros")
