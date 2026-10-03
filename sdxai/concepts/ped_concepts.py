"""Phase 4b: fine-grained pedestrian concepts, each tested separately from its confounds.

Population: GT pedestrians matched to a detection query (instances.parquet kind == "tp"), layer-6 query
features (decoder layer 6 input, as in Phase 4). Labels come from public nuScenes annotations, tracks and
the nuScenes map expansion; nothing is hand-labelled.

Concepts (positive vs negative set, confounds controlled for, tier):
  moving_vs_standing     attribute pedestrian.moving vs pedestrian.standing       speed, distance, visibility
  sitting_vs_standing    pedestrian.sitting_lying_down vs pedestrian.standing     distance, visibility, box height
  worker_vs_adult        category construction_worker vs adult                    distance, visibility, speed
  crossing_vs_alongside  ahead, walking > 0.5 m/s towards the ego axis vs along it  distance, lateral offset, speed
  road_vs_walkway        on drivable area (not a crossing) vs on a walkway          distance, lateral offset, visibility
  crossing_zone_vs_walkway  on a marked ped_crossing vs on a walkway              distance, lateral offset, visibility
  group_vs_alone         another GT pedestrian within 1.5 m vs none within 4 m     distance, visibility, crowd density
  emerging_vs_visible    visible now (v3-4) but v1-2 one keyframe earlier vs v3-4 both   distance, speed
  child_vs_adult   (weak) category child vs adult                                  distance, visibility, box height
  night_vs_day     (weak) scene description contains "night"                      distance, visibility

Per concept:
 1 encoded?      logistic probe, balanced accuracy. Held-out protocol: train split -> val split when val has
                 >= 20 distinct people on each side, else 5-fold scene-grouped CV on train+val ("scene_cv").
                 Controls: the same probe on shuffled labels; a probe on the confounds alone.
 2 beyond confounds?  features residualised on the confounds (linear + squares, fit on the training fold)
                 before probing; scene-bootstrap 95 % CI of the held-out balanced accuracy.
 3 separate?     cosine between concept directions (probe weights in raw feature space) and with confound
                 directions; LEACE (Belrose et al. 2023) linear erasure of concept A, then probe concept B:
                 retention = (bacc_B after erasing A - 0.5) / (bacc_B - 0.5).
 4 decision?     TCAV on the pedestrian logit, reusing Phase-4 gradients (tcav/grads.npz, layer 6, val
                 pedestrians): 30 bootstrap CAVs vs 30 random CAVs, Welch t-test, Bonferroni over concepts.
 5 planner?      val pedestrians in the Phase-5 planner rows: attention percentile within the frame and plan
                 change when masked, positive minus negative, distance-stratified, Mann-Whitney p.
 6 picture       camera crops of positive examples.
"""
from __future__ import annotations

import argparse
import json
import os
SDXAI_ROOT = os.environ.get("SDXAI_ROOT", os.getcwd())  # workspace with SparseDrive/, nuscenes/, navsim_exp/, b2d/

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from sdxai.concepts.data import gather

DATAROOT = os.path.join(SDXAI_ROOT, "nuscenes")
MAX_PER_CLASS = 6000
MIN_VAL_PEOPLE = 20
N_CAV = 30


# ------------------------------------------------------------------ labels
class MapIndex:
    """Point-in-layer tests for drivable_area / ped_crossing / walkway with one STRtree per layer (the devkit's
    layers_on_point scans every record per call, too slow at trainval scale). Same polygons as the devkit."""

    def __init__(self, nmap):
        from shapely.strtree import STRtree
        self.trees = {}
        for layer in ("drivable_area", "ped_crossing", "walkway"):
            polys = []
            for rec in getattr(nmap, layer):
                toks = rec["polygon_tokens"] if "polygon_tokens" in rec else [rec["polygon_token"]]
                polys += [nmap.extract_polygon(t) for t in toks]
            polys = [p if p.is_valid else p.buffer(0) for p in polys]
            self.trees[layer] = (STRtree(polys), polys)

    def flags(self, x, y):
        from shapely.geometry import Point
        pt = Point(x, y)
        return {k: any(p.contains(pt) for p in tree.query(pt)) for k, (tree, _) in self.trees.items()}


def build_labels(inst: pd.DataFrame, version: str) -> pd.DataFrame:
    from nuscenes.map_expansion.map_api import NuScenesMap
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes("v1.0-mini" if version == "mini" else "v1.0-trainval", DATAROOT, verbose=False)
    peds_all = inst[inst.gt_class == "pedestrian"]
    d = peds_all[peds_all.kind == "tp"].copy()
    d["pred_idx"] = d.pred_idx.astype(int)
    d = d[d.gt_ann_token.apply(lambda t: isinstance(t, str))].reset_index(drop=True)
    maps, loc_of_scene = {}, {}
    rec = []
    for t, sc in zip(d.gt_ann_token, d.scene_token):
        a = nusc.get("sample_annotation", t)
        if sc not in loc_of_scene:
            loc_of_scene[sc] = nusc.get("log", nusc.get("scene", sc)["log_token"])["location"]
        loc = loc_of_scene[sc]
        if loc not in maps:
            maps[loc] = MapIndex(NuScenesMap(DATAROOT, loc))
        x, y = a["translation"][:2]
        on = maps[loc].flags(x, y)
        vprev = None
        if a["prev"]:
            vprev = int(nusc.get("visibility", nusc.get("sample_annotation", a["prev"])["visibility_token"])["token"])
        rec.append({"ped_attrs": ",".join(nusc.get("attribute", k)["name"] for k in a["attribute_tokens"]),
                    "box_h": a["size"][2], "on_drivable": bool(on["drivable_area"]),
                    "on_crossing": bool(on["ped_crossing"]), "on_walkway": bool(on["walkway"]), "vis_prev": vprev,
                    "night_desc": "night" in nusc.get("scene", sc)["description"].lower()})
    d = pd.concat([d, pd.DataFrame(rec)], axis=1)
    # neighbours: all GT pedestrians (detected or not) in the same frame
    near1, near4 = [], []
    for (sp, fr), g in d.groupby(["split", "frame"]):
        allp = peds_all[(peds_all.split == sp) & (peds_all.frame == fr)][["gt_x", "gt_y"]].values
        for i, (x, y) in zip(g.index, g[["gt_x", "gt_y"]].values):
            dist = np.hypot(allp[:, 0] - x, allp[:, 1] - y)
            dist = dist[dist > 1e-3]
            near1.append((i, int((dist < 1.5).sum()))); near4.append((i, int((dist < 4.0).sum())))
    d.loc[[i for i, _ in near1], "n_within_1p5"] = [n for _, n in near1]
    d.loc[[i for i, _ in near4], "n_within_4"] = [n for _, n in near4]
    ahead, spd = d.s_along_path > 0, d.gt_speed
    d["_toward"] = ahead & (spd > 0.5) & (d.gt_vx.abs() > d.gt_vy.abs()) & (d.gt_x * d.gt_vx < 0)
    d["_along"] = ahead & (spd > 0.5) & (d.gt_vy.abs() > 2 * d.gt_vx.abs())
    d["abs_x"] = d.gt_x.abs()
    return d


def concept_table():
    A = lambda s: (lambda d: d.ped_attrs.str.contains(s, regex=False))  # noqa: E731
    C = lambda s: (lambda d: d.gt_category == f"human.pedestrian.{s}")  # noqa: E731
    vis = lambda d: d.visibility  # noqa: E731
    return {
        "moving_vs_standing": (A("pedestrian.moving"), A("pedestrian.standing"), ["gt_speed", "gt_dist", "visibility"], "main"),
        "sitting_vs_standing": (A("sitting_lying_down"), A("pedestrian.standing"), ["gt_dist", "visibility", "box_h"], "main"),
        "worker_vs_adult": (C("construction_worker"), C("adult"), ["gt_dist", "visibility", "gt_speed"], "main"),
        "crossing_vs_alongside": (lambda d: d._toward, lambda d: d._along, ["gt_dist", "lateral_to_path", "gt_speed"], "main"),
        "road_vs_walkway": (lambda d: d.on_drivable & ~d.on_crossing & ~d.on_walkway,
                            lambda d: d.on_walkway & ~d.on_drivable, ["gt_dist", "lateral_to_path", "visibility"], "main"),
        "crossing_zone_vs_walkway": (lambda d: d.on_crossing, lambda d: d.on_walkway & ~d.on_drivable,
                                     ["gt_dist", "lateral_to_path", "visibility"], "main"),
        "group_vs_alone": (lambda d: d.n_within_1p5 >= 1, lambda d: d.n_within_4 == 0,
                           ["gt_dist", "visibility", "crowd_density"], "main"),
        "emerging_vs_visible": (lambda d: (vis(d) >= 3) & (d.vis_prev <= 2), lambda d: (vis(d) >= 3) & (d.vis_prev >= 3),
                                ["gt_dist", "gt_speed"], "main"),
        "child_vs_adult": (C("child"), C("adult"), ["gt_dist", "visibility", "box_h"], "weak"),
        "night_vs_day": (lambda d: d.night_desc, lambda d: ~d.night_desc, ["gt_dist", "visibility"], "weak"),
    }


# ------------------------------------------------------------------ helpers
def conf_matrix(d, cols):
    c = d[cols].astype(float).fillna(d[cols].astype(float).median()).values
    return np.c_[c, c ** 2]


def fit_probe(X, y, seed=0):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced", random_state=seed).fit(sc.transform(X), y)
    return sc, clf


def residualise(Xtr, Ctr, Xte, Cte):
    lr = LinearRegression().fit(Ctr, Xtr)
    return Xtr - lr.predict(Ctr), Xte - lr.predict(Cte)


def splits(sub: pd.DataFrame):
    """[(train_idx, test_idx)], protocol name."""
    va = sub[sub.split == "val"]
    ppl = va.groupby("y").gt_instance_token.nunique()
    if (sub.split == "train").any() and len(ppl) == 2 and ppl.min() >= MIN_VAL_PEOPLE:
        return [(np.where(sub.split == "train")[0], np.where(sub.split == "val")[0])], "train->val"
    groups = sub.scene_token.values
    k = min(5, len(np.unique(groups)))
    if k < 2:
        return [], "too_few_scenes"
    return list(GroupKFold(k).split(sub, sub.y, groups)), "scene_cv"


def boot_ci(y, p, groups, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if len(np.unique(y[s])) == 2:
            vals.append(balanced_accuracy_score(y[s], p[s]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [None, None]


def evaluate(sub, X, C, seed=0, transform=None):
    """Held-out balanced accuracy for features / confounds-only / residualised / shuffled labels."""
    folds, proto = splits(sub)
    y = sub.y.values
    out = {k: np.full(len(y), -1) for k in ("feat", "conf", "resid", "shuf")}
    rng = np.random.RandomState(seed)
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        tr = balance(tr, y, rng)
        Xtr, Xte = X[tr], X[te]
        if transform is not None:
            Xtr, Xte = transform(Xtr), transform(Xte)
        sc, clf = fit_probe(Xtr, y[tr]); out["feat"][te] = clf.predict(sc.transform(Xte))
        sc, clf = fit_probe(C[tr], y[tr]); out["conf"][te] = clf.predict(sc.transform(C[te]))
        Rtr, Rte = residualise(Xtr, C[tr], Xte, C[te])
        sc, clf = fit_probe(Rtr, y[tr]); out["resid"][te] = clf.predict(sc.transform(Rte))
        ys = rng.permutation(y[tr])
        sc, clf = fit_probe(Xtr, ys); out["shuf"][te] = clf.predict(sc.transform(Xte))
    m = out["feat"] >= 0
    if m.sum() == 0 or len(np.unique(y[m])) < 2:
        return None
    g = sub.scene_token.values[m]
    res = {"protocol": proto, "n_test": int(m.sum())}
    for k, p in out.items():
        res[f"bacc_{k}"] = float(balanced_accuracy_score(y[m], p[m]))
    res["bacc_resid_ci"] = boot_ci(y[m], out["resid"][m], g)
    res["bacc_feat_ci"] = boot_ci(y[m], out["feat"][m], g)
    return res


def balance(idx, y, rng, cap=MAX_PER_CLASS):
    parts = []
    for c in (0, 1):
        ic = idx[y[idx] == c]
        if len(ic) > cap:
            ic = rng.choice(ic, cap, replace=False)
        parts.append(ic)
    return np.concatenate(parts)


def direction(X, y, rng):
    idx = balance(np.arange(len(y)), y, rng)
    sc, clf = fit_probe(X[idx], y[idx])
    v = clf.coef_[0] / sc.scale_
    return v / np.linalg.norm(v)


class LEACE:
    """Least-squares concept erasure (Belrose et al. 2023), binary concept, closed form."""

    def __init__(self, X, y, eps=1e-4):
        self.mu = X.mean(0)
        Xc = X - self.mu
        S = Xc.T @ Xc / len(X)
        w, V = np.linalg.eigh(S)
        w = np.clip(w, eps * w.max(), None)
        W = V @ np.diag(w ** -0.5) @ V.T
        Wp = V @ np.diag(w ** 0.5) @ V.T
        u = W @ (Xc.T @ (y - y.mean()) / len(X))
        u = u / (np.linalg.norm(u) + 1e-12)
        self.P = np.eye(len(u)) - Wp @ np.outer(u, u) @ W

    def __call__(self, X):
        return (X - self.mu) @ self.P.T + self.mu


# ------------------------------------------------------------------ main
def link_by_position(sub, pm, max_d=2.0):
    from scipy.optimize import linear_sum_assignment
    pms = {f: d for f, d in pm.groupby("frame")}
    out = []
    for f, d in sub.groupby("frame"):
        p = pms.get(f)
        if p is None:
            continue
        D = np.linalg.norm(d[["gt_x", "gt_y"]].values[:, None] - p[["pred_x", "pred_y"]].values[None], axis=-1)
        r, c = linear_sum_assignment(D)
        keep = D[r, c] <= max_d
        out.append(pd.concat([d.iloc[r[keep]].reset_index(drop=True),
                              p.iloc[c[keep]][["attn_pct", "mask_dl2_avg"]].reset_index(drop=True)], axis=1))
    return pd.concat(out) if out else sub.iloc[:0].assign(attn_pct=[], mask_dl2_avg=[])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=os.environ.get("SDXAI_VERSION", "mini"))
    ap.add_argument("--work", default="work_dirs/xai")
    ap.add_argument("--gallery", type=int, default=8)
    a = ap.parse_args()
    out = os.path.join(a.work, "concepts", "ped_concepts")
    os.makedirs(os.path.join(out, "gallery"), exist_ok=True)
    inst = pd.read_parquet(os.path.join(a.work, "instances", "instances.parquet"))
    lab_path = os.path.join(out, "labels.parquet")
    if os.path.exists(lab_path):
        d = pd.read_parquet(lab_path)
    else:
        d = build_labels(inst, a.version)
        d.to_parquet(lab_path, index=False)
    X = gather(d, 6)
    ok = np.isfinite(X).all(1) & (np.abs(X).sum(1) > 0)  # train frames not stored (save-stride) -> dropped
    d, X = d[ok].reset_index(drop=True), X[ok]
    print(f"pedestrian TPs with layer-6 features: {len(d)} (train {int((d.split == 'train').sum())}, val {int((d.split == 'val').sum())})")

    rng = np.random.RandomState(0)
    concepts = concept_table()
    rows, subsets, dirs = [], {}, {}
    for name, (fp, fn, conf, tier) in concepts.items():
        pos, neg = fp(d).fillna(False).astype(bool), fn(d).fillna(False).astype(bool)
        pos, neg = pos & ~neg, neg & ~pos
        sub = d[pos | neg].copy()
        sub["y"] = pos[pos | neg].astype(int).values
        Xs = X[(pos | neg).values]
        cnt = {s: {int(k): int(v) for k, v in sub[sub.split == s].groupby("y").gt_instance_token.nunique().items()}
               for s in ("train", "val")}
        row = {"concept": name, "tier": tier, "confounds": ",".join(conf), "n_pos": int(sub.y.sum()),
               "n_neg": int((1 - sub.y).sum()), "people_train": cnt["train"], "people_val": cnt["val"],
               "n_scenes_pos": int(sub[sub.y == 1].scene_token.nunique())}
        if row["n_pos"] >= 20 and row["n_neg"] >= 20 and sub[sub.y == 1].scene_token.nunique() >= 2:
            C = conf_matrix(sub, conf)
            r = evaluate(sub, Xs, C)
            if r:
                row.update(r)
                subsets[name] = (sub, Xs, C)
                dirs[name] = direction(Xs, sub.y.values, rng)
        rows.append(row)
        print(name, {k: row.get(k) for k in ("n_pos", "n_neg", "protocol", "bacc_feat", "bacc_conf", "bacc_resid", "bacc_shuf")})
    res = pd.DataFrame(rows)

    # 3a cosines between concept directions and with confound directions (ridge on raw features)
    names = list(dirs)
    cos = pd.DataFrame([[float(dirs[i] @ dirs[j]) for j in names] for i in names], index=names, columns=names)
    conf_dirs = {}
    for c in ["gt_dist", "gt_speed", "visibility", "box_h", "lateral_to_path"]:
        v = d[c].astype(float)
        m = v.notna().values
        w = LinearRegression().fit(StandardScaler().fit_transform(X[m]), (v[m] - v[m].mean()) / (v[m].std() + 1e-9)).coef_
        w = w / StandardScaler().fit(X[m]).scale_
        conf_dirs[c] = w / np.linalg.norm(w)
    cos_conf = pd.DataFrame([[float(abs(dirs[i] @ conf_dirs[c])) for c in conf_dirs] for i in names], index=names,
                            columns=list(conf_dirs))
    cos.to_csv(os.path.join(out, "direction_cosines.csv")); cos_conf.to_csv(os.path.join(out, "confound_cosines.csv"))

    # 3b LEACE erasure matrix: erase A (fit on A's rows), probe B; only concepts whose probe is clearly
    # above chance (bacc >= 0.6), since retention is a ratio to (bacc - 0.5)
    enc = [n for n in names if res.set_index("concept").loc[n, "bacc_feat"] >= 0.6]
    er = []
    for A in enc:
        subA, XA, _ = subsets[A]
        eraser = LEACE(XA, subA.y.values.astype(float))
        for B in enc:
            subB, XB, CB = subsets[B]
            base = res.set_index("concept").loc[B, "bacc_feat"]
            r = evaluate(subB, XB, CB, transform=eraser)
            after = r["bacc_feat"] if r else np.nan
            er.append({"erase": A, "probe": B, "bacc_before": base, "bacc_after": after,
                       "retention": (after - 0.5) / (base - 0.5) if base > 0.5 else np.nan})
    er = pd.DataFrame(er)
    er.to_csv(os.path.join(out, "erasure.csv"), index=False)

    # 4 TCAV reusing Phase-4 gradients (layer 6)
    tc = []
    gpath = os.path.join(a.work, "concepts", "tcav", "grads.npz")
    if os.path.exists(gpath):
        G = np.load(gpath)["L6"]
        for name in names:
            sub, Xs, _ = subsets[name]
            tr = (sub.split == "train").values if (sub.split == "train").sum() >= 40 else np.ones(len(sub), bool)
            Xp, Xn = Xs[tr & (sub.y == 1).values], Xs[tr & (sub.y == 0).values]
            if len(Xp) < 10 or len(Xn) < 10:
                continue
            cs, rs = [], []
            for _ in range(N_CAV):
                cs.append(float((G @ direction(np.r_[Xp[rng.choice(len(Xp), len(Xp))], Xn[rng.choice(len(Xn), len(Xn))]],
                                                np.r_[np.ones(len(Xp)), np.zeros(len(Xn))], rng) > 0).mean()))
                pool = np.r_[Xp, Xn]; perm = rng.permutation(len(pool))
                rs.append(float((G @ direction(pool[perm], np.r_[np.ones(len(Xp)), np.zeros(len(Xn))], rng) > 0).mean()))
            t = ttest_ind(cs, rs, equal_var=False)
            tc.append({"concept": name, "tcav_mean": np.mean(cs), "tcav_std": np.std(cs), "random_mean": np.mean(rs),
                       "random_std": np.std(rs), "p": float(t.pvalue), "n_val_ped_grads": len(G)})
    tc = pd.DataFrame(tc)
    if len(tc):
        tc["p_bonferroni"] = np.minimum(1, tc.p * len(tc)); tc["significant"] = tc.p_bonferroni < 0.05
    tc.to_csv(os.path.join(out, "tcav.csv"), index=False)

    # 5 planner: attention percentile and masking effect, pos - neg, distance-stratified
    pl = []
    ppath = os.path.join(a.work, "interventions", "planner_mask.parquet")
    if os.path.exists(ppath):
        pm = pd.read_parquet(ppath)
        pm = pm[pm.tag == "trained"].copy()
        pm["attn_pct"] = pm.groupby("frame").score_attn.rank(pct=True)
        pm = pm[pm.is_ped][["frame", "pred_x", "pred_y", "attn_pct", "mask_dl2_avg"]]
        for name in names:
            sub = subsets[name][0]
            # the planner run and the instance table are separate runs whose query slots differ, so planner agents are
            # linked to labelled pedestrians by position (Hungarian, <= 2 m) rather than by slot index
            j = link_by_position(sub[sub.split == "val"], pm)
            if j.y.sum() < 10 or (1 - j.y).sum() < 10:
                pl.append({"concept": name, "n_pos": int(j.y.sum()), "n_neg": int((1 - j.y).sum())}); continue
            j["db"] = pd.cut(j.gt_dist, [0, 15, 30, 100])
            diffs = {}
            for m in ("attn_pct", "mask_dl2_avg"):
                ws, ds = [], []
                for _, g in j.groupby("db"):
                    if g.y.sum() and (1 - g.y).sum():
                        ds.append(g[g.y == 1][m].mean() - g[g.y == 0][m].mean()); ws.append(len(g))
                diffs[m] = float(np.average(ds, weights=ws)) if ws else np.nan
                diffs[m + "_p"] = float(mannwhitneyu(j[j.y == 1][m], j[j.y == 0][m]).pvalue)
            pl.append({"concept": name, "link": "position", "n_pos": int(j.y.sum()), "n_neg": int((1 - j.y).sum()),
                       "attn_pct_pos": float(j[j.y == 1].attn_pct.mean()), "attn_pct_neg": float(j[j.y == 0].attn_pct.mean()),
                       "d_attn_pct_dist_strat": diffs["attn_pct"], "p_attn": diffs["attn_pct_p"],
                       "mask_cm_pos": float(100 * j[j.y == 1].mask_dl2_avg.mean()),
                       "mask_cm_neg": float(100 * j[j.y == 0].mask_dl2_avg.mean()),
                       "d_mask_cm_dist_strat": 100 * diffs["mask_dl2_avg"], "p_mask": diffs["mask_dl2_avg_p"]})
    pl = pd.DataFrame(pl)
    pl.to_csv(os.path.join(out, "planner.csv"), index=False)

    # verdict per concept
    def verdict(r):
        if pd.isna(r.get("bacc_feat")):
            return "not testable (too few examples)"
        enc = r["bacc_feat"] - max(r["bacc_shuf"], 0.5) >= 0.05
        beyond = r["bacc_resid_ci"][0] is not None and r["bacc_resid_ci"][0] > 0.5 and r["bacc_resid"] >= 0.55
        if not enc:
            return "not encoded"
        return "encoded beyond confounds" if beyond else "encoded, but explained by confounds"
    res["verdict"] = [verdict(r) for r in res.to_dict("records")]
    off = er[er.erase != er.probe] if len(er) else er
    res["min_retention_of_others_when_erased"] = res.concept.map(off.groupby("erase").retention.min() if len(off) else {})
    res["entangled_with"] = res.concept.map(
        {A: ",".join(sorted(set(g[g.retention < 0.5].probe) | set(off[(off.probe == A) & (off.retention < 0.5)].erase)))
         for A, g in off.groupby("erase")} if len(off) else {})
    res.to_json(os.path.join(out, "probes.json"), orient="records", indent=1)
    res.drop(columns=["bacc_resid_ci", "bacc_feat_ci"], errors="ignore").to_csv(os.path.join(out, "probes.csv"), index=False)

    # 6 galleries
    if a.gallery:
        make_galleries(d, concepts, out, a.version, a.gallery)
    summary = {"n_ped_tp": len(d), "concepts": res[["concept", "tier", "n_pos", "n_neg", "protocol", "bacc_feat",
                                                    "bacc_conf", "bacc_resid", "bacc_shuf", "verdict",
                                                    "entangled_with"]].to_dict("records"),
               "tcav": tc.to_dict("records"), "planner": pl.to_dict("records")}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1, default=str)
    print(res[["concept", "tier", "n_pos", "n_neg", "protocol", "bacc_feat", "bacc_conf", "bacc_resid", "bacc_shuf",
               "verdict"]].round(3).to_string(index=False))
    if len(tc):
        print(tc.round(4).to_string(index=False))
    if len(pl):
        print(pl.round(3).to_string(index=False))
    print(er.pivot(index="erase", columns="probe", values="retention").round(2).to_string())


def make_galleries(d, concepts, out, version, n):
    import cv2
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.geometry_utils import BoxVisibility, view_points
    nusc = NuScenes("v1.0-mini" if version == "mini" else "v1.0-trainval", DATAROOT, verbose=False)
    cams = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

    def crop(tok_s, tok_a, text, size=160):
        best = None
        s = nusc.get("sample", tok_s)
        for c in cams:
            p, b, K = nusc.get_sample_data(s["data"][c], box_vis_level=BoxVisibility.ANY, selected_anntokens=[tok_a])
            if b:
                uv = view_points(b[0].corners(), np.array(K), normalize=True)[:2]
                lo, hi = np.clip(uv.min(1), 0, [1600, 900]), np.clip(uv.max(1), 0, [1600, 900])
                ar = np.prod(hi - lo)
                if best is None or ar > best[0]:
                    best = (ar, p, uv)
        if best is None:
            return None
        _, p, uv = best
        img = cv2.imread(p)
        for i, j in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
            cv2.line(img, tuple(uv[:, i].astype(int)), tuple(uv[:, j].astype(int)), (60, 200, 250), 2, cv2.LINE_AA)
        cx, cy = uv.mean(1); h = max(np.ptp(uv[1]), np.ptp(uv[0]) * 1.6) * 1.7; w = h / 1.6
        big = cv2.copyMakeBorder(img, 900, 900, 900, 900, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        cr = big[int(cy - h / 2) + 900:int(cy + h / 2) + 900, int(cx - w / 2) + 900:int(cx + w / 2) + 900]
        cr = cv2.resize(cr, (size, int(size * 1.6)))
        cv2.rectangle(cr, (0, cr.shape[0] - 24), (size, cr.shape[0]), (20, 20, 20), -1)
        cv2.putText(cr, text, (5, cr.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
        return cr

    rng = np.random.RandomState(1)
    for name, (fp, fn, _, _) in concepts.items():
        for side, f in (("pos", fp), ("neg", fn)):
            sel = d[f(d).fillna(False).astype(bool)]
            if len(sel) == 0:
                continue
            # readable examples first: val split, visible (v3-4), closer than 25 m; random within each tier
            sel = sel.assign(prio=2 * (sel.split == "val") + (sel.visibility >= 3) + (sel.gt_dist < 25),
                             r=rng.rand(len(sel))).sort_values(["prio", "r"], ascending=[False, True])
            pick = sel.drop_duplicates("gt_instance_token").head(n * 2)
            tiles = []
            for _, r in pick.iterrows():
                t = crop(r.sample_token, r.gt_ann_token, f"{r.split} {r.gt_dist:.0f}m v{int(r.visibility)} p={r.pred_ped_prob:.2f}")
                if t is not None:
                    tiles.append(t)
                if len(tiles) == n:
                    break
            if tiles:
                row = np.hstack([np.hstack([t, np.full((t.shape[0], 5, 3), 255, np.uint8)]) for t in tiles])
                cv2.imwrite(os.path.join(out, "gallery", f"{name}_{side}.jpg"), row, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
