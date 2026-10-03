"""Vector figures for the paper, read from the same result files as build_numbers.py (trainval preferred,
mini otherwise; the caption macros carry the dagger). Camera-image figures are copied from the explainer
pipeline (reports/ped_explainer/make_figures.py)."""
import os
import shutil

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from build_numbers import T, R  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
INK, ACC, GREY, BAD, GOOD = "#1b2120", "#d9822b", "#8a9491", "#c0392b", "#2e8b57"

# --- memory: recall of occluded pedestrians vs history length
tr, _ = T("temporal/temporal_recall_rows.parquet")
occ = tr[tr.visibility <= 2]
rec = occ.groupby("tag").detected.mean()
tags = ["history0", "history1", "history2", "history3", "full", "det_temp_gnn_zero_values"]
labels = ["0", "1", "2", "3", "full", "full,\nno det.\ntemp. attn."]
fig, ax = plt.subplots(figsize=(3.4, 1.75))
cols = [BAD, ACC, ACC, ACC, GOOD, GREY]
ax.bar(range(len(tags)), [rec[t] for t in tags], color=cols, width=0.68)
for i, t in enumerate(tags):
    ax.text(i, rec[t] + 0.012, f"{rec[t]:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
ax.set_xticks(range(len(tags)), labels, fontsize=7)
ax.set_xlabel("frames of history available to the detector", fontsize=7.5)
ax.set_ylabel("recall, visibility 0-60 %", fontsize=7.5)
ax.set_ylim(0, max(rec[t] for t in tags) * 1.22)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "memory_recall.pdf"))

# --- planner: rank-AUC per stratum and ranker
au, _ = T("interventions/auc_table.csv")
strata = [("all_agents", "all agents"), ("ped_all", "pedestrians"), ("ped_near(<15)", "ped. < 15 m"),
          ("ped_occluded(1-2)", "ped. occluded"), ("ped_critical_w8", "ped. in corridor")]
rankers = [("grad", "plan gradient", GOOD), ("attn", "planner attention", ACC), ("random", "random", GREY),
           ("physics", "1/TTC (physics)", BAD)]
fig, ax = plt.subplots(figsize=(3.4, 2.15))
for j, (rk, lab, c) in enumerate(rankers):
    for i, (s, _) in enumerate(strata):
        d = au[(au.stratum == s) & (au.ranker == rk)]
        if len(d) == 0:
            continue
        d = d.iloc[0]
        y = i + (j - 1.5) * 0.17
        ax.plot([d.lo, d.hi], [y, y], color=c, lw=1.1)
        ax.plot(d.auc, y, "o", color=c, ms=3.2, label=lab if i == 0 else None)
ax.axvline(0.5, color=GREY, lw=0.6, ls=":")
ax.set_yticks(range(len(strata)), [s[1] for s in strata], fontsize=7)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("rank-AUC (positives: top-20 % masking effect)", fontsize=7.5)
fig.legend(fontsize=6.5, frameon=False, loc="upper center", ncol=2, handletextpad=0.3, columnspacing=1.2,
           bbox_to_anchor=(0.55, 1.0))
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.88))
fig.savefig(os.path.join(OUT, "auc.pdf"), bbox_inches="tight", pad_inches=0.02)

# --- camera-image figures from the explainer pipeline
IMG = f"{R}/reports/ped_explainer/img_tv"  # full-data camera figures
for f in ["f1_frame.jpg", "f2_attribution.jpg", "f3_occlusion.jpg", "f5_memory.jpg", "f6_planner.jpg"] + \
        [f"f4_latent_{k}.jpg" for k in range(5)]:
    shutil.copy(os.path.join(IMG, f), os.path.join(OUT, f))
G = f"{R}/SparseDrive_tv/work_dirs/xai/concepts/ped_concepts/gallery"
for c in ["crossing_vs_alongside", "crossing_zone_vs_walkway", "group_vs_alone", "night_vs_day"]:
    p = os.path.join(G, f"{c}_pos.jpg")
    if os.path.exists(p):
        shutil.copy(p, os.path.join(OUT, f"concept_{c}.jpg"))
print(sorted(os.listdir(OUT)))
