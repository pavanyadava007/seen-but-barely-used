"""Concept-level seed stability of the primary SAE (complements sae_seed_stability.py, which matches single units).

For seeds 1 and 2 the latent catalogue is rebuilt exactly as for seed 0 (sae_downstream.catalogue: same 500k training
queries, same purity rule), then compared with seed 0 on three levels:
  n_ped           number of pedestrian units (purity >= 0.75)
  dir_cos         cosine between the pedestrian directions (mean of the top-10 pedestrian atoms, as used for steering)
  subspace_cos    mean cosine of the principal angles between the spans of the pedestrian atoms
  subspace_rand   the same for 20 random alive atoms of each seed (chance level)
Run from SparseDrive_tv with PYTHONPATH=. ; writes paper/sae_seed_concepts.json."""
import json
import os
SDXAI_ROOT = os.environ.get("SDXAI_ROOT", os.getcwd())  # workspace with SparseDrive/, nuscenes/, navsim_exp/, b2d/

import numpy as np
import pandas as pd
import torch

from sdxai.concepts.sae_downstream import OwnSAE, catalogue

C = "work_dirs/xai/concepts"
OUT = os.path.join(SDXAI_ROOT, "paper/sae_seed_concepts.json")


def ped_atoms(model, cat_path):
    lat = pd.DataFrame(json.load(open(cat_path))["latents"])
    ped = lat[lat.ped_purity >= 0.75].sort_values("ped_selectivity", ascending=False).latent.tolist()
    D = model.dictionary().float().cpu()
    D = D / D.norm(dim=1, keepdim=True)
    return D, ped, lat.latent.tolist()


def principal_cos(A, B):
    qa, _ = torch.linalg.qr(A.T)
    qb, _ = torch.linalg.qr(B.T)
    return float(torch.linalg.svdvals(qa.T @ qb).mean())


inst = pd.read_parquet("work_dirs/xai/instances/instances.parquet")
seeds = {0: (f"{C}/sae/sae_n2048_k64.pt", f"{C}/sae_downstream/catalogue_sae_n2048_k64.json")}
for s in (1, 2):
    d = f"{C}/sae_seed{s}"
    cp = f"{d}/catalogue_sae_n2048_k64.json"
    if not os.path.exists(cp):
        catalogue(OwnSAE(f"{d}/sae_n2048_k64.pt"), inst, d)
    seeds[s] = (f"{d}/sae_n2048_k64.pt", cp)

info = {s: ped_atoms(OwnSAE(p), c) for s, (p, c) in seeds.items()}
rng = np.random.RandomState(0)
res = {"n_ped": {s: len(info[s][1]) for s in info}}
for s in (1, 2):
    D0, p0, a0 = info[0]
    D1, p1, a1 = info[s]
    u0, u1 = D0[p0[:10]].mean(0), D1[p1[:10]].mean(0)
    n = min(len(p0), len(p1))
    rnd = [principal_cos(D0[rng.choice(a0, n, replace=False)], D1[rng.choice(a1, n, replace=False)]) for _ in range(20)]
    res[f"seed{s}"] = {"dir_cos": float(torch.nn.functional.cosine_similarity(u0, u1, dim=0)),
                       "subspace_cos": principal_cos(D0[p0[:n]], D1[p1[:n]]),
                       "subspace_rand": float(np.mean(rnd)), "n_compared": n}
for k in ("dir_cos", "subspace_cos", "subspace_rand"):
    res[k] = float(np.mean([res[f"seed{s}"][k] for s in (1, 2)]))
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
