"""Phase 4c: controls for the primary-SAE ablation and steering (reviewer requests).

Same live splice as sae_downstream.py (layer-6 input features of all 900 slots replaced, refine-6 and planner
re-run), on the same frame subset, for the primary SAE only:
  ablate_ped10      zero the top-10 pedestrian latents (as in Phase 4)
  ablate_car10      zero the 10 car-pure latents (purity >= 0.75) with the highest car selectivity, computed exactly
                    like ped_selectivity (standardised mean difference car vs. rest on the catalogue queries)
  ablate_rand10_s   zero 10 random alive latents (3 seeds)
All ablations are error-preserving: x' = x - (dec(z) - dec(z_abl)). Per condition: AP for pedestrians and
cars (centre-distance AP, mean over 0.5/1/2/4 m), and the class flip rate of matched GT pedestrians / cars.
Steering on GT-non-pedestrian TP slots: alpha = 0 (no change: the baseline rate), pedestrian direction and a
norm-matched random direction at alpha 1 and 4; flip rate to pedestrian (argmax, score > 0.3) and mean change
of the pedestrian logit.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sdxai import metrics as MT
from sdxai import planner as PL
from sdxai.concepts.sae_downstream import GT_RANGE, load_models, plan_eval, slot_gt_class
from sdxai.replay import Replay
from sdxai.runner import CLASS_NAMES

PED, CAR = CLASS_NAMES.index("pedestrian"), CLASS_NAMES.index("car")


def gt_xy(info, cls):
    names, boxes = info["gt_names"], info["gt_boxes"]
    pts = info["num_lidar_pts"] + info["num_radar_pts"]
    m = (names == cls) & (pts > 0) & (np.linalg.norm(boxes[:, :2], axis=-1) <= GT_RANGE)
    return boxes[m, :2]


def preds(cls, anchor, qt, frame, c_id):
    prob = cls[0].float().sigmoid()
    score, c = prob.max(-1)
    score = score * qt[0, :, 0].float().sigmoid()
    top = score.topk(300).indices
    keep = top[c[top] == c_id]
    return anchor[0, keep, :2].float().cpu().numpy(), score[keep].cpu().numpy(), np.full(len(keep), frame)


def car_selectivity(model, units, max_n=500_000):
    """(mean car - mean rest) / pooled sd per unit, on the same catalogue queries as ped_selectivity."""
    from sdxai.concepts.data import all_instances, label_instances
    X, meta = all_instances("train", 6, 0.1)
    if len(X) > max_n:
        keep = np.sort(np.random.RandomState(0).choice(len(X), max_n, replace=False))
        X, meta = X[keep], meta.iloc[keep].reset_index(drop=True)
    lab = label_instances(meta, pd.read_parquet("work_dirs/xai/instances/instances.parquet")).label.values
    Xt = torch.from_numpy(X).cuda()
    Z = torch.cat([model.codes(Xt[s:s + 8192]).float()[:, units].cpu() for s in range(0, len(Xt), 8192)]).numpy()
    is_car = lab == "car"
    out = {}
    for j, u in enumerate(units):
        col = Z[:, j]
        sd = np.sqrt(0.5 * (col[is_car].var() + col[~is_car].var())) + 1e-8
        out[u] = float((col[is_car].mean() - col[~is_car].mean()) / sd)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=os.environ.get("SDXAI_VERSION", "mini"))
    ap.add_argument("--work", default="work_dirs/xai")
    ap.add_argument("--primary", default="sae_n2048_k64")
    ap.add_argument("--frame-stride", type=int, default=4)
    a = ap.parse_args()
    out = os.path.join(a.work, "concepts", "sae_controls")
    os.makedirs(out, exist_ok=True)
    cat = json.load(open(os.path.join(a.work, "concepts", "sae_downstream", f"catalogue_{a.primary}.json")))
    meta = json.load(open(os.path.join(a.work, "concepts", "sae_downstream", "downstream_meta.json")))
    top10, scale = meta["top10_ped_latents"], meta["steering_scale"]
    lat = pd.DataFrame(cat["latents"])
    cars = lat[(lat.top_class == "car") & (lat.purity >= 0.75)].latent.tolist()
    m = load_models([a.primary])[a.primary]
    car_sel = car_selectivity(m, cars)
    car10 = sorted(cars, key=lambda u: -car_sel[u])[:10]
    alive = lat.latent.tolist()
    rng = np.random.RandomState(0)
    pool = [x for x in alive if x not in top10 and x not in car10]
    rand = {f"ablate_rand10_s{s}": rng.choice(pool, 10, replace=False).tolist() for s in range(3)}
    sets = {"ablate_ped10": top10, "ablate_car10": car10, **rand}
    D = m.dictionary().float()
    u_ped = D[top10].mean(0)
    u_ped = (u_ped / u_ped.norm()).cuda()
    g = torch.Generator(device="cpu").manual_seed(1)
    u_rand = torch.randn(D.shape[1], generator=g)
    u_rand = (u_rand / u_rand.norm()).cuda()

    rp = Replay(version=a.version)
    h = rp.head
    P = defaultdict(lambda: defaultdict(list))
    gts = {PED: ([], []), CAR: ([], [])}
    flips, steer, plans, norms = [], [], [], []
    idx = list(range(0, len(rp.r.dataset), a.frame_stride))
    for fr in tqdm(rp.frames(indices=set(idx)), total=len(idx), desc="sae controls"):
        feat6 = fr.cap["layer_feat"][5][None].float()
        anc5 = fr.cap["layer_anchor"][4][None].float()
        for c_id, name in ((PED, "pedestrian"), (CAR, "car")):
            xy = gt_xy(fr.info, name)
            gts[c_id][0].append(xy)
            gts[c_id][1].append(np.full(len(xy), fr.i))
        norms.append(float(feat6[0].norm(dim=-1).mean()))
        with torch.no_grad():
            a6, c6, q6, _ = PL.refine6_from_features(h.det_head, feat6, anc5, fr.time_interval)
            slot_cls = slot_gt_class(fr, c6)

            def ev(tag, f_new):
                an, cn, qn, en = PL.refine6_from_features(h.det_head, f_new, anc5, fr.time_interval)
                for c_id in (PED, CAR):
                    for k, v in zip(("xy", "score", "frame"), preds(cn, an, qn, fr.i, c_id)):
                        P[(tag, c_id)][k].append(v)
                dec = fr.run(P=PL.splice_P(h.motion_plan_head, fr.P, f_new, an, cn, en), det=PL.det_with(fr.det, an, cn))
                plans.append({"frame": fr.i, "tag": tag, **plan_eval(fr, dec)})
                return cn

            ev("reference", feat6)
            z = m.codes(feat6[0]).float()
            base_rec = m.recon_from_codes(z)
            for tag, units in sets.items():
                z2 = z.clone()
                z2[:, units] = 0
                cn = ev(tag, feat6 - (base_rec - m.recon_from_codes(z2))[None])
                for gname, cid in (("pedestrian", PED), ("car", CAR)):
                    sl = [s for s, c in slot_cls.items() if c == gname]
                    if sl:
                        flips.append({"frame": fr.i, "tag": tag, "gt": gname, "n": len(sl),
                                      "flip": float((cn[0, sl].argmax(-1) != cid).float().mean())})
            non = [s for s, c in slot_cls.items() if c != "pedestrian"]
            if non:
                base_logit = c6[0, non, PED].float()
                for dname, u in (("ped", u_ped), ("random", u_rand)):
                    for al in ((0.0, 1.0, 4.0) if dname == "ped" else (1.0, 4.0)):
                        f2 = feat6.clone()
                        f2[0, non] += al * scale * u
                        _, cn, _, _ = PL.refine6_from_features(h.det_head, f2, anc5, fr.time_interval)
                        pr = cn[0, non].float().sigmoid()
                        steer.append({"frame": fr.i, "dir": dname, "alpha": al, "n": len(non),
                                      "flip": float(((pr.argmax(-1) == PED) & (pr.max(-1).values > 0.3)).float().mean()),
                                      "d_ped_logit": float((cn[0, non, PED].float() - base_logit).mean())})
    rp.close()
    rows = []
    for (tag, c_id), p in P.items():
        G, GF = gts[c_id]
        r = MT.ap_center(np.concatenate(p["xy"]), np.concatenate(p["score"]), np.concatenate(p["frame"]),
                         np.concatenate(G), np.concatenate(GF))
        rows.append({"tag": tag, "class": CLASS_NAMES[c_id], "ap": r["ap"]})
    apdf = pd.DataFrame(rows)
    ref = apdf[apdf.tag == "reference"].set_index("class").ap
    apdf["dAP_pts"] = 100 * (apdf.ap - apdf["class"].map(ref))
    fl = pd.DataFrame(flips).groupby(["tag", "gt"]).apply(lambda d: np.average(d.flip, weights=d.n)).rename("flip_rate").reset_index()
    st = pd.DataFrame(steer).groupby(["dir", "alpha"]).apply(
        lambda d: pd.Series({"flip_rate": np.average(d.flip, weights=d.n), "d_ped_logit": np.average(d.d_ped_logit, weights=d.n)})).reset_index()
    pl = pd.DataFrame(plans).groupby("tag")[["dl2_avg", "mode_changed"]].mean().reset_index()
    apdf.to_csv(os.path.join(out, "ap_by_class.csv"), index=False)
    fl.to_csv(os.path.join(out, "flip_rates.csv"), index=False)
    st.to_csv(os.path.join(out, "steering_controls.csv"), index=False)
    pl.to_csv(os.path.join(out, "plan.csv"), index=False)
    json.dump({"car10": car10, "car_selectivity": {int(u): car_sel[u] for u in car10}, "rand": rand, "top10": top10, "steering_scale": scale,
               "mean_query_norm": float(np.mean(norms)), "steer_norm_ratio_alpha1": scale / float(np.mean(norms))},
              open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(apdf.round(4).to_string(index=False)); print(fl.round(4).to_string(index=False))
    print(st.round(4).to_string(index=False)); print(pl.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
