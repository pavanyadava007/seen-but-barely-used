"""Phase 4 (3-4): SAE latent catalogue, downstream loss recovered, steering / ablation.

Models: sdxai TopK SAEs (sae.py) and every trained Overcomplete dictionary (overcomplete_suite.py),
through one interface (codes / recon_from_codes / dictionary).

Catalogue (train split, slots with score > 0.1, GT labels from instances.parquet; unmatched slots
are labelled "unmatched"): per latent the top-32 activating instances, class / distance-bin /
visibility histograms, monosemanticity = max class purity among the top-32, activation frequency.
Pedestrian latents: purity(ped) >= 0.75 among the top-32, ranked by selectivity
(mean activation on pedestrians - mean on others) / pooled std; top-10 used for steering.

Downstream (val replay, live model): at every frame the det head's layer-6 input features of all
900 slots are replaced by the model's reconstruction, refine-6 is re-run (classification, boxes,
quality) and the planner is re-run on the spliced inputs (sdxai.planner.splice_P).
  AP_ped   : nuScenes-style centre-distance AP (sdxai.metrics.ap_center), scores = max-class prob x
             centerness (upstream decoder), top-300 per frame, pedestrian-class predictions
  plan     : dl2 vs the unmodified plan at 1/2/3 s, L2 vs GT, collision flag (upstream definition)
Steering (primary SAE):
  ablate   : zero the top-10 pedestrian latents, error-preserving: x' = x - (dec(z) - dec(z_abl))
             -> dAP_ped, fraction of GT-pedestrian slots (TP, score > 0.3) whose argmax leaves ped
  add      : x' = x + alpha * s * u_ped on GT-non-pedestrian TP slots, u_ped = unit mean decoder
             direction of the pedestrian latents, s = mean L2 norm of their summed contribution when
             active -> false-positive rate = fraction flipping to ped with score > 0.3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sdxai import metrics as MT
from sdxai import planner as PL
from sdxai.concepts.data import all_instances, label_instances
from sdxai.concepts.sae import TopKSAE
from sdxai.instances import GT_RANGE
from sdxai.replay import Replay
from sdxai.runner import CLASS_NAMES, PED

TOPN = 32


class OwnSAE:
    """sdxai TopK SAE behind the same interface as overcomplete_suite.DictModel."""

    def __init__(self, path):
        ck = torch.load(path, map_location="cuda")
        self.m = TopKSAE(ck["d"], ck["n_latents"], ck["k"]).cuda()
        self.m.load_state_dict(ck["state_dict"])
        self.m.eval()
        self.name = os.path.basename(path)[:-3]
        self.family = "sdxai_TopK"
        self.neural = True

    @torch.no_grad()
    def codes(self, x):
        return self.m.encode(x.cuda().float())

    @torch.no_grad()
    def recon_from_codes(self, z):
        return self.m.decode(z)

    @torch.no_grad()
    def recon(self, x):
        return self.recon_from_codes(self.codes(x))

    def dictionary(self):
        return (self.m.W_dec.t() * self.m.scale).detach()  # (n, d), in input units


def load_models(names):
    # Overcomplete DictModels were pickled while overcomplete_suite ran as __main__ (python -m),
    # so their class is recorded as __main__.DictModel: expose it there before unpickling.
    import sys
    from sdxai.concepts.overcomplete_suite import DictModel
    sys.modules["__main__"].DictModel = DictModel
    out = {}
    for n in names:
        own = f"work_dirs/xai/concepts/sae/{n}.pt"
        oc = f"work_dirs/xai/concepts/overcomplete/{n}.pt"
        if os.path.exists(own):
            out[n] = OwnSAE(own)
        elif os.path.exists(oc):
            out[n] = torch.load(oc)
    return out


# ------------------------------------------------------------------------------------------
def catalogue(model, inst: pd.DataFrame, out_dir: str, max_n: int = 500_000):
    X, meta = all_instances("train", 6, 0.1)
    if len(X) > max_n:
        keep = np.sort(np.random.RandomState(0).choice(len(X), max_n, replace=False))
        X, meta = X[keep], meta.iloc[keep].reset_index(drop=True)
    lab = label_instances(meta, inst)
    Xt = torch.from_numpy(X).cuda()
    # codes go to host memory chunk by chunk: 500k x 8192 float32 is ~16 GB, more than the GPU has free
    Z = torch.cat([model.codes(Xt[s:s + 8192]).float().cpu() for s in range(0, len(Xt), 8192)]).numpy()
    labels = lab.label.values
    is_ped = labels == "pedestrian"
    freq = (Z > 0).mean(0)
    cat = []
    for j in range(Z.shape[1]):
        col = Z[:, j]
        if freq[j] == 0:
            continue
        top = np.argsort(-col)[:TOPN]
        top = top[col[top] > 0]
        if len(top) == 0:
            continue
        cls_counts = pd.Series(labels[top]).value_counts()
        mu_p, mu_o = col[is_ped].mean(), col[~is_ped].mean()
        sd = np.sqrt(0.5 * (col[is_ped].var() + col[~is_ped].var())) + 1e-8
        cat.append({
            "latent": j, "freq": float(freq[j]), "top_n": int(len(top)),
            "top_class": cls_counts.index[0], "purity": float(cls_counts.iloc[0] / len(top)),
            "class_hist": cls_counts.to_dict(),
            "dist_hist": lab.dist_bin.iloc[top].fillna("na").value_counts().to_dict(),
            "vis_hist": lab.vis_bin.iloc[top].fillna("na").value_counts().to_dict(),
            "ped_purity": float((labels[top] == "pedestrian").mean()),
            "ped_selectivity": float((mu_p - mu_o) / sd),
            "mean_act_when_active": float(col[col > 0].mean()),
            "top_examples": [{"frame": int(lab.frame.iloc[t]), "pred_idx": int(lab.pred_idx.iloc[t]),
                              "act": float(col[t]), "label": labels[t]} for t in top[:8]],
        })
    cat_df = pd.DataFrame(cat)
    ped_lat = cat_df[(cat_df.ped_purity >= 0.75)].sort_values("ped_selectivity", ascending=False)
    summary = {
        "n_alive": int(len(cat_df)), "n_latents": int(Z.shape[1]),
        "mean_purity": float(cat_df.purity.mean()),
        "frac_purity>=0.9": float((cat_df.purity >= 0.9).mean()),
        "n_ped_latents(purity>=0.75)": int(len(ped_lat)),
        "top10_ped_latents": ped_lat.latent.head(10).tolist(),
        "top_class_counts": cat_df[cat_df.purity >= 0.9].top_class.value_counts().to_dict(),
    }
    json.dump({"summary": summary, "latents": cat}, open(os.path.join(out_dir, f"catalogue_{model.name}.json"), "w"),
              indent=1, default=float)
    # steering direction
    top10 = ped_lat.latent.head(10).tolist()
    D = model.dictionary().float()
    act = Z[:, top10]
    contrib = torch.from_numpy(act).cuda() @ D[top10].cuda()
    active = (act > 0).any(1)
    scale = float(contrib[torch.from_numpy(active).cuda()].norm(dim=1).mean()) if active.any() else 1.0
    u = D[top10].mean(0)
    u = (u / (u.norm() + 1e-8)).cuda()
    return summary, top10, u, scale


# ------------------------------------------------------------------------------------------
def gt_peds(info):
    names, boxes = info["gt_names"], info["gt_boxes"]
    pts = info["num_lidar_pts"] + info["num_radar_pts"]
    rng = np.linalg.norm(boxes[:, :2], axis=-1)
    m = (names == "pedestrian") & (pts > 0) & (rng <= GT_RANGE)
    return boxes[m, :2]


def det_preds(cls, anchor, qt, frame):
    prob = cls[0].float().sigmoid()
    score, c = prob.max(-1)
    score = score * qt[0, :, 0].float().sigmoid()
    top = score.topk(300).indices
    keep = top[c[top] == PED]
    return {"xy": anchor[0, keep, :2].float().cpu().numpy(), "score": score[keep].cpu().numpy(),
            "frame": np.full(len(keep), frame)}


def slot_gt_class(fr, cls):
    """GT class per slot for this live frame (Hungarian at 2 m as in Phase 2), score > 0.3."""
    from scipy.optimize import linear_sum_assignment
    info = fr.info
    prob = cls[0].float().sigmoid().cpu().numpy()
    anc = fr.det["prediction"][-1][0].float().cpu().numpy()
    pidx = np.where(prob.max(-1) > 0.3)[0]
    names, boxes = info["gt_names"], info["gt_boxes"]
    pts = info["num_lidar_pts"] + info["num_radar_pts"]
    rng = np.linalg.norm(boxes[:, :2], axis=-1)
    gidx = np.where(np.isin(names, CLASS_NAMES) & (pts > 0) & (rng <= GT_RANGE))[0]
    out = {}
    if len(pidx) and len(gidx):
        Dm = np.linalg.norm(anc[pidx, None, :2] - boxes[None, gidx, :2], axis=-1)
        r, c = linear_sum_assignment(Dm)
        for a, b in zip(r, c):
            if Dm[a, b] <= 2.0:
                out[int(pidx[a])] = names[gidx[b]]
    return out


def plan_eval(fr, dec):
    final = dec["final"].float().cpu().numpy()
    base = fr.base["final"].float().cpu().numpy()
    res = MT.plan_l2_at(final, base)
    if fr.gt_mask.all():
        res["l2_gt_avg"] = MT.plan_l2_at(final, fr.gt_traj)["dl2_avg"]
        res.update({k: v for k, v in MT.plan_collision(final, fr.gt_traj, fr.fut_boxes).items() if k == "col_any"})
    res["mode_changed"] = int(dec["mode_idx"] != fr.base["mode_idx"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=os.environ.get("SDXAI_VERSION", "mini"))
    ap.add_argument("--instances", default="work_dirs/xai/instances/instances.parquet")
    ap.add_argument("--out", default="work_dirs/xai/concepts/sae_downstream")
    ap.add_argument("--primary", default="sae_n2048_k64")
    ap.add_argument("--models", default="")
    ap.add_argument("--alphas", default="0.5,1,2,4")
    ap.add_argument("--frame-stride", type=int, default=1, help="evaluate every k-th val frame (all frames still stream through the model, so temporal state is exact)")
    ap.add_argument("--catalogue-max", type=int, default=500_000, help="max train instances for the latent catalogue")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    inst = pd.read_parquet(a.instances)
    names = [x for x in a.models.split(",") if x] or (
        [os.path.basename(p)[:-3] for p in sorted(glob.glob("work_dirs/xai/concepts/sae/sae_n*.pt"))]
        + [os.path.basename(p)[:-3] for p in sorted(glob.glob("work_dirs/xai/concepts/overcomplete/oc_*.pt"))])
    models = load_models(names)
    prim = models[a.primary]
    cat_summary, top10, u_ped, scale = catalogue(prim, inst, a.out, a.catalogue_max)
    # catalogues for the overcomplete TopK / RA-TopK too (monosemanticity comparison)
    cat_all = {a.primary: cat_summary}
    for n in ["oc_TopK_n2048_k64", "oc_RATopK_n2048_k64", "sae_n8192_k64"]:
        if n in models and n != a.primary:
            cat_all[n] = catalogue(models[n], inst, a.out, a.catalogue_max)[0]
    json.dump(cat_all, open(os.path.join(a.out, "catalogue_summaries.json"), "w"), indent=2, default=float)
    alphas = [float(x) for x in a.alphas.split(",")]

    rp = Replay(version=a.version)
    h = rp.head
    preds = defaultdict(lambda: {"xy": [], "score": [], "frame": []})
    plan_rows, steer_rows = [], []
    gts, gtf = [], []
    check = []

    def add_preds(key, p):
        for k in ("xy", "score", "frame"):
            preds[key][k].append(p[k])

    idx = list(range(0, len(rp.r.dataset), a.frame_stride))
    for fr in tqdm(rp.frames(indices=set(idx)), total=len(idx), desc="sae downstream"):
        feat6 = fr.cap["layer_feat"][5][None].float()
        anc5 = fr.cap["layer_anchor"][4][None].float()
        g = gt_peds(fr.info)
        gts.append(g)
        gtf.append(np.full(len(g), fr.i))
        with torch.no_grad():
            a6, c6, q6, e6 = PL.refine6_from_features(h.det_head, feat6, anc5, fr.time_interval)
            check.append(float((c6 - fr.det["classification"][-1]).abs().max()))
            add_preds("live", det_preds(fr.det["classification"][-1], fr.det["prediction"][-1], fr.det["quality"][-1], fr.i))
            add_preds("refine6_reference", det_preds(c6, a6, q6, fr.i))
            slot_cls = slot_gt_class(fr, c6)

            def splice_eval(tag, feat_new):
                an, cn, qn, en = PL.refine6_from_features(h.det_head, feat_new, anc5, fr.time_interval)
                add_preds(tag, det_preds(cn, an, qn, fr.i))
                Pn = PL.splice_P(h.motion_plan_head, fr.P, feat_new, an, cn, en)
                dec = fr.run(P=Pn, det=PL.det_with(fr.det, an, cn))
                return cn, {"frame": fr.i, "tag": tag, **plan_eval(fr, dec)}

            # reference splice (identity features): must reproduce the base plan
            _, row = splice_eval("identity", feat6)
            plan_rows.append(row)
            for n, m in models.items():
                rec = m.recon(feat6[0]).float()[None]
                _, row = splice_eval(n, rec)
                plan_rows.append(row)

            # steering on the primary SAE
            z = prim.codes(feat6[0]).float()
            z_abl = z.clone()
            z_abl[:, top10] = 0
            delta = prim.recon_from_codes(z) - prim.recon_from_codes(z_abl)
            c_abl, row = splice_eval("ablate_top10_ped", feat6 - delta[None])
            plan_rows.append(row)
            ped_slots = [s for s, c in slot_cls.items() if c == "pedestrian"]
            non_slots = [s for s, c in slot_cls.items() if c != "pedestrian"]
            if ped_slots:
                flips = (c_abl[0, ped_slots].argmax(-1) != PED).float().mean().item()
                steer_rows.append({"frame": fr.i, "kind": "ablate", "alpha": 0.0, "n": len(ped_slots), "flip_rate": flips})
            for al in alphas:
                f2 = feat6.clone()
                if non_slots:
                    f2[0, non_slots] += al * scale * u_ped
                c_add, row = splice_eval(f"add_ped_alpha{al}", f2)
                plan_rows.append(row)
                if non_slots:
                    pr = c_add[0, non_slots].float().sigmoid()
                    fp = ((pr.argmax(-1) == PED) & (pr.max(-1).values > 0.3)).float().mean().item()
                    steer_rows.append({"frame": fr.i, "kind": "add", "alpha": al, "n": len(non_slots), "flip_rate": fp})
    rp.close()

    G = np.concatenate(gts)
    GF = np.concatenate(gtf)
    ap_rows = []
    for key, p in preds.items():
        r = MT.ap_center(np.concatenate(p["xy"]), np.concatenate(p["score"]), np.concatenate(p["frame"]), G, GF)
        ap_rows.append({"model": key, **r})
    apdf = pd.DataFrame(ap_rows)
    ref_ap = float(apdf[apdf.model == "refine6_reference"].ap.iloc[0])
    apdf["dAP_ped_pts"] = 100 * (apdf.ap - ref_ap)
    pl = pd.DataFrame(plan_rows)
    pl_sum = pl.groupby("tag").agg(dl2_avg=("dl2_avg", "mean"), dl2_3s=("dl2_3s", "mean"),
                                   mode_changed=("mode_changed", "mean"), l2_gt_avg=("l2_gt_avg", "mean"),
                                   col_rate=("col_any", "mean"), n=("frame", "size")).reset_index()
    st = pd.DataFrame(steer_rows)
    st_sum = st.groupby(["kind", "alpha"]).apply(lambda d: np.average(d.flip_rate, weights=d.n)).rename("rate").reset_index()
    apdf.to_csv(os.path.join(a.out, "ap_ped.csv"), index=False)
    pl.to_parquet(os.path.join(a.out, "plan_rows.parquet"), index=False)
    pl_sum.to_csv(os.path.join(a.out, "plan_summary.csv"), index=False)
    st_sum.to_csv(os.path.join(a.out, "steering.csv"), index=False)
    ev = {}
    for p in ["work_dirs/xai/concepts/sae/sae_metrics.json"]:
        for r in json.load(open(p)):
            ev[r["name"]] = r
    ocp = "work_dirs/xai/concepts/overcomplete/overcomplete_metrics.json"
    if os.path.exists(ocp):
        for k, r in json.load(open(ocp)).items():
            if "explained_variance" in r:
                ev[k] = r
    acc_rows = []
    for n in models:
        e = ev.get(n, {})
        d_ap = float(apdf[apdf.model == n].dAP_ped_pts.iloc[0])
        acc_rows.append({"model": n, "EV": e.get("explained_variance"), "L0": e.get("L0"), "dAP_ped_pts": d_ap,
                         "pass_EV>0.9": bool((e.get("explained_variance") or 0) > 0.9),
                         "pass_L0<=64": bool((e.get("L0") or 1e9) <= 64 + 1e-6),
                         "pass_|dAP|<=1": bool(abs(d_ap) <= 1.0)})
    acc = pd.DataFrame(acc_rows)
    acc["pass_all"] = acc["pass_EV>0.9"] & acc["pass_L0<=64"] & acc["pass_|dAP|<=1"]
    acc.to_csv(os.path.join(a.out, "sae_acceptance.csv"), index=False)
    json.dump({"refine6_reproduction_max_abs_logit_diff": max(check), "catalogue_primary": cat_summary,
               "steering_scale": scale, "top10_ped_latents": top10}, open(os.path.join(a.out, "downstream_meta.json"), "w"),
              indent=2, default=float)
    print(apdf.round(4).to_string(index=False))
    print(pl_sum.round(4).to_string(index=False))
    print(st_sum.round(4).to_string(index=False))
    print(acc.round(4).to_string(index=False))
    print("refine6 reproduction max |dlogit|:", max(check))


if __name__ == "__main__":
    main()
