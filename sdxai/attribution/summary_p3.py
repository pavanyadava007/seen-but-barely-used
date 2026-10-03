"""Phase 3 summary: tables with cluster-bootstrap CIs, method comparisons, MPRT, acceptance, figures."""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sdxai import stats as ST

METRICS = ["morf_auc", "lerf_auc", "pointing_game", "energy_pg", "iou@0.2", "iou@0.5"]
STRATA = {"all": lambda d: d.index == d.index,
          "critical": lambda d: d.critical.astype(bool) if "critical" in d else d.index != d.index,
          "critical_w8": lambda d: d.critical_w8.astype(bool),
          "near(<15)": lambda d: d.dist_bin == "<15",
          "far(>=15)": lambda d: d.dist_bin != "<15",
          "occluded(1-2)": lambda d: d.vis_bin == "occluded(1-2)",
          "visible(3-4)": lambda d: d.vis_bin == "visible(3-4)"}


def table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sname, f in STRATA.items():
        d = df[f(df)]
        if len(d) == 0:
            continue
        clusters, unit = ST.cluster_unit(d)
        d = d.assign(_cl=clusters.values)
        for m, dm in d.groupby("method"):
            row = {"stratum": sname, "method": m, "n": len(dm), "ci_unit": unit}
            for k in METRICS:
                ci = ST.cluster_bootstrap(dm[k].values, dm._cl.values)
                row[k] = ci["est"]
                row[k + "_lo"], row[k + "_hi"] = ci["lo"], ci["hi"]
            rows.append(row)
    return pd.DataFrame(rows)


def comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """Paired Wilcoxon on identical targets for every method pair, per stratum and metric;
    Holm-Bonferroni over all (pair x metric x stratum) tests; Cliff's delta as effect size."""
    rows = []
    key = ["frame", "gt_idx"]
    for sname, f in STRATA.items():
        d = df[f(df)]
        if d.frame.nunique() == 0:
            continue
        for k in ["morf_auc", "lerf_auc", "energy_pg"]:
            p = d.pivot_table(index=key, columns="method", values=k)
            ms = list(p.columns)
            for i, m1 in enumerate(ms):
                for m2 in ms[i + 1:]:
                    w = ST.paired_wilcoxon(p[m1], p[m2])
                    rows.append({"stratum": sname, "metric": k, "a": m1, "b": m2,
                                 "mean_a": float(p[m1].mean()), "mean_b": float(p[m2].mean()), **w})
    out = pd.DataFrame(rows)
    out["p_holm"] = ST.holm(out["p"].values)
    return out


def map_rank_correlation(d: str, methods=("dfa", "grad", "smoothgrad", "grad_feat", "drise")) -> pd.DataFrame:
    """Mean Spearman correlation between methods' |maps| over all targets (4x-downsampled maps)."""
    import glob
    from scipy.stats import spearmanr
    acc = {(a, b): [] for a in methods for b in methods}
    for path in sorted(glob.glob(os.path.join(d, "maps", "frame_*.npz"))):
        z = np.load(path)
        gts = sorted({k.split("/")[0] for k in z.files})
        for g in gts:
            mp = {m: np.abs(z[f"{g}/{m}"].astype(np.float32)).ravel() for m in methods}
            for a in methods:
                for b in methods:
                    if a <= b:
                        r = 1.0 if a == b else spearmanr(mp[a], mp[b]).correlation
                        acc[(a, b)].append(r)
                        acc[(b, a)].append(r)
    return pd.DataFrame([[np.nanmean(acc[(a, b)]) for b in methods] for a in methods], index=methods, columns=methods)


def mprt_summary(m: pd.DataFrame) -> tuple:
    if len(m) == 0:
        return pd.DataFrame(), {}
    agg = m.groupby(["method", "stage", "stage_name"]).agg(
        spearman=("spearman", "mean"), ssim=("ssim", "mean"),
        degenerate=("degenerate", "mean"), n=("spearman", "size")).reset_index()
    verdict = {}
    for meth, d in agg.groupby("method"):
        d = d.sort_values("stage")
        valid = d[d.degenerate < 0.5]
        sp = valid.spearman.values
        # monotone decay: no increase > 0.05 between consecutive stages, and net drop >= 0.3
        mono = bool(len(sp) >= 3 and np.all(np.diff(sp) <= 0.05) and sp[0] - sp[-1] >= 0.3)
        last = valid.iloc[-1] if len(valid) else None
        fail = bool(last is not None and (last.spearman > 0.7 or last.ssim > 0.7))
        verdict[meth] = {"monotone_decay": mono, "last_valid_stage": None if last is None else last.stage_name,
                         "last_spearman": None if last is None else float(last.spearman),
                         "last_ssim": None if last is None else float(last.ssim), "FAIL(>0.7)": fail,
                         "degenerate_stages": d[d.degenerate >= 0.5].stage_name.tolist()}
    return agg, verdict


def qualitative(df: pd.DataFrame, d: str, out_png: str, n_cases: int = 20):
    from sdxai.runner import build_dataset, load_cfg
    cfg = load_cfg()
    ds = build_dataset(cfg, "test")
    mean = np.array(cfg.img_norm_cfg.mean)[:, None, None]
    std = np.array(cfg.img_norm_cfg.std)[:, None, None]
    base = df[df.method == "dfa"].copy()
    picks = []
    for name, f in [("near", base.dist_bin == "<15"), ("occluded", base.vis_bin == "occluded(1-2)"),
                    ("far", base.dist_bin == ">30"), ("critical_w8", base.critical_w8.astype(bool))]:
        cand = base[f & ~base.set_index(["frame", "gt_idx"]).index.isin([(p[1], p[2]) for p in picks])]
        for _, r in cand.head(5).iterrows():
            picks.append((name, int(r.frame), int(r.gt_idx)))
    picks = picks[:n_cases]
    methods = ["dfa", "smoothgrad", "grad_feat", "drise"]
    fig, axes = plt.subplots(len(picks), 1 + len(methods), figsize=(2.2 * (1 + len(methods)), 1.9 * len(picks)))
    axes = np.atleast_2d(axes)
    cache = {}
    for row, (name, fr, g) in enumerate(picks):
        z = np.load(os.path.join(d, "maps", f"frame_{fr}.npz"))
        gm = z[f"{g}/gt_mask"]
        cam = int(gm.reshape(6, -1).sum(1).argmax())
        ys, xs = np.where(gm[cam])
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        half = max(y1 - y0, x1 - x0, 12) * 1.2
        Y0, Y1 = int(max(cy - half, 0)), int(min(cy + half, gm.shape[1]))
        X0, X1 = int(max(cx - half, 0)), int(min(cx + half, gm.shape[2]))
        if fr not in cache:
            img = ds[fr]["img"].data.numpy()  # 6,3,256,704 normalised (BGR->RGB handled by pipeline)
            cache[fr] = np.clip((img * std[None] + mean[None]) / 255.0, 0, 1)
        im = cache[fr][cam].transpose(1, 2, 0)[::4, ::4]
        axes[row, 0].imshow(im[Y0:Y1, X0:X1])
        axes[row, 0].add_patch(plt.Rectangle((x0 - X0, y0 - Y0), x1 - x0, y1 - y0, fill=False, ec="lime", lw=1))
        r0 = df[(df.frame == fr) & (df.gt_idx == g)].iloc[0]
        axes[row, 0].set_ylabel(f"{name}\n{r0.gt_dist:.0f} m v{int(r0.visibility)}", fontsize=7)
        for j, m in enumerate(methods):
            a = np.abs(z[f"{g}/{m}"][cam].astype(np.float32))
            a = a / (a.max() + 1e-12)
            axes[row, 1 + j].imshow(im[Y0:Y1, X0:X1])
            axes[row, 1 + j].imshow(a[Y0:Y1, X0:X1], cmap="inferno", alpha=0.6)
            axes[row, 1 + j].add_patch(plt.Rectangle((x0 - X0, y0 - Y0), x1 - x0, y1 - y0, fill=False, ec="lime", lw=1))
            if row == 0:
                axes[row, 1 + j].set_title(m, fontsize=8)
        if row == 0:
            axes[row, 0].set_title("input (GT box)", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    plt.close()
    return picks


def road_plot(curves: pd.DataFrame, out_png: str):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.linspace(0, 1, 21)
    for m, d in curves.groupby("method"):
        for k, a in [("morf", ax[0]), ("lerf", ax[1])]:
            c = np.stack(d[k].values).mean(0)
            a.plot(x, c, label=m)
    for a, t in zip(ax, ["MoRF (lower = more faithful)", "LeRF (higher = more faithful)"]):
        a.set_title(t, fontsize=9)
        a.set_xlabel("fraction of pixel budget removed (budget = 3x GT box area)")
        a.set_ylabel("detection similarity / s0")
    ax[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="work_dirs/xai/attribution")
    a = ap.parse_args()
    d = a.dir
    df = pd.read_parquet(os.path.join(d, "attr_results.parquet"))
    df["critical_w8"] = df.critical_w8.astype(bool)
    tab = table(df)
    tab.to_csv(os.path.join(d, "attr_table.csv"), index=False)
    comp = comparisons(df)
    comp.to_csv(os.path.join(d, "attr_comparisons.csv"), index=False)
    rc = map_rank_correlation(d)
    rc.to_csv(os.path.join(d, "method_rank_correlation.csv"))
    m = pd.read_parquet(os.path.join(d, "mprt.parquet"))
    magg, verdict = mprt_summary(m)
    magg.to_csv(os.path.join(d, "mprt_table.csv"), index=False)
    tr = pd.read_parquet(os.path.join(d, "target_rand.parquet"))
    tr_agg = tr.groupby("method")[["spearman", "ssim"]].mean()
    road_plot(pd.read_parquet(os.path.join(d, "road_curves.parquet")), os.path.join(d, "road_curves.png"))
    picks = qualitative(df, d, os.path.join(d, "qualitative_20.png"))

    # acceptance
    from sdxai.instances import analysis_stratum
    cname = analysis_stratum()["flag"]
    crit = tab[tab.stratum == cname].set_index("method")
    allm = tab[tab.stratum == "all"].set_index("method")
    use = crit if len(crit) and crit.n.min() >= 5 else allm
    real = use.drop(index="random")
    best = real.morf_auc.idxmin()
    rel = 1 - use.loc[best, "morf_auc"] / use.loc["random", "morf_auc"]
    n_mono = sum(v["monotone_decay"] for v in verdict.values())
    acc = {
        "n_targets": int(df[df.method == "dfa"].shape[0]),
        "n_frames": int(df.frame.nunique()), "n_scenes": int(df.scene_token.nunique()),
        "ci_unit": str(tab.ci_unit.iloc[0]),
        "road_stratum_used": cname if use is crit else "all",
        "best_method_morf": best, "best_morf_auc": float(use.loc[best, "morf_auc"]),
        "random_morf_auc": float(use.loc["random", "morf_auc"]),
        "relative_improvement_vs_random": float(rel),
        "pass_road_20pct": bool(rel >= 0.20),
        "mprt_verdict": verdict, "n_methods_monotone_decay": int(n_mono),
        "pass_mprt_2_methods": bool(n_mono >= 2),
        "target_randomisation_similarity": tr_agg.round(3).to_dict(),
        "method_map_spearman": rc.round(3).to_dict(),
        "n_significant_pairwise_holm": int((comp.p_holm < 0.05).sum()), "n_pairwise_tests": int(len(comp)),
        "qualitative_cases": [f"{p[0]}:frame{p[1]}:gt{p[2]}" for p in picks],
        "note": "critical stratum is the widened critical_w8 (see Phase 2); strict critical is empty on mini-val",
    }
    json.dump(acc, open(os.path.join(d, "p3_acceptance.json"), "w"), indent=2, default=str)
    print(tab[tab.stratum.isin(["all", "critical_w8"])][["stratum", "method", "n", "morf_auc", "morf_auc_lo",
                                                           "morf_auc_hi", "lerf_auc", "energy_pg", "pointing_game"]]
          .round(3).to_string(index=False))
    print(magg.pivot_table(index="stage_name", columns="method", values="spearman", sort=False).round(2))
    print(json.dumps({k: v for k, v in acc.items() if k != "qualitative_cases"}, indent=2, default=str))


if __name__ == "__main__":
    main()
