"""Seed stability of the primary SAE (sae_n2048_k64, seed 0) against seeds 1 and 2 trained on the same data:
Hungarian matching of decoder atoms by cosine similarity; reported for all alive atoms, for the 20 pedestrian atoms,
and for the four units shown in the paper's unit figure. Writes sae_seed_stability.json."""
import json
import os
SDXAI_ROOT = os.environ.get("SDXAI_ROOT", os.getcwd())  # workspace with SparseDrive/, nuscenes/, navsim_exp/, b2d/

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

X = os.path.join(SDXAI_ROOT, "SparseDrive_tv/work_dirs/xai/concepts")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sae_seed_stability.json")


def atoms(path):
    sd = torch.load(path, map_location="cpu")["state_dict"]
    D = sd["W_dec"].float().T  # (n_latents, d)
    return (D / D.norm(dim=1, keepdim=True)).numpy()


cat = json.load(open(f"{X}/sae_downstream/catalogue_sae_n2048_k64.json"))
alive = [c["latent"] for c in cat["latents"]]
ped = [c["latent"] for c in cat["latents"] if c["ped_purity"] >= 0.75]
facts = json.load(open(os.path.join(SDXAI_ROOT, "reports/ped_explainer/figure_facts_tv.json")))
fig_units = [facts["sae_latents"][i]["latent"] for i in (0, 2, 3)]  # the three pedestrian units shown in Fig. 4 (rows 1-3)
A0 = atoms(f"{X}/sae/sae_n2048_k64.pt")
res = {"n_alive": len(alive), "n_ped": len(ped)}
best = {}
for s in (1, 2):
    A1 = atoms(f"{X}/sae_seed{s}/sae_n2048_k64.pt")
    C = A0 @ A1.T
    r, c = linear_sum_assignment(-C)
    m = dict(zip(r, C[r, c]))
    best[s] = m
    res[f"seed{s}"] = {"mean_cos_all": float(np.mean([m[u] for u in alive])),
                       "mean_cos_ped": float(np.mean([m[u] for u in ped])),
                       "ped_ge_09": int(sum(m[u] >= 0.9 for u in ped)),
                       "fig_units": {int(u): float(m[u]) for u in fig_units}}
res["mean_cos_all"] = float(np.mean([res[f"seed{s}"]["mean_cos_all"] for s in (1, 2)]))
res["mean_cos_ped"] = float(np.mean([res[f"seed{s}"]["mean_cos_ped"] for s in (1, 2)]))
res["ped_matched_09"] = int(sum(min(best[1][u], best[2][u]) >= 0.9 for u in ped))
res["fig_units_matched"] = int(sum(min(best[1][u], best[2][u]) >= 0.9 for u in fig_units))
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
