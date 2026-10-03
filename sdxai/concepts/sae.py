"""TopK sparse autoencoders on decoder-layer-6 instance features (the planner's input features).

Architecture (Gao et al. 2024 TopK SAE): z = TopK(ReLU(W_enc (x - b_dec) + b_enc)), x_hat = W_dec z + b_dec,
unit-norm decoder columns, inputs standardised by a global scale (mean norm -> sqrt(d)).
Training data: train-split slots with final score > `--train-score-thr`. The plan's protocol is
score > 0.1 (~5 M vectors on full train); on v1.0-mini that yields only 43 k vectors and the SAE is
data-limited (val EV rises monotonically with epochs, see sensitivity.log), so the mini default uses
all 900 slots per frame (290 k vectors, threshold 0.0). Evaluation is always on val slots with
score > 0.1.

Metrics: explained variance, mean cosine(x, x_hat), L0, dead fraction (never active on val),
dense fraction (active on > 10 % of val inputs). Downstream effects are evaluated in
`sae_downstream.py` (det head re-run for AP_ped, planner re-run for plan L2 / collision).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from sdxai.concepts.data import all_instances
from sdxai.runner import seed_all


class TopKSAE(nn.Module):
    def __init__(self, d: int, n_latents: int, k: int):
        super().__init__()
        self.d, self.n, self.k = d, n_latents, k
        self.W_enc = nn.Parameter(torch.empty(n_latents, d))
        self.b_enc = nn.Parameter(torch.zeros(n_latents))
        self.W_dec = nn.Parameter(torch.empty(d, n_latents))
        self.b_dec = nn.Parameter(torch.zeros(d))
        self.register_buffer("scale", torch.ones(()))
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.data /= self.W_dec.data.norm(dim=0, keepdim=True)
            self.W_enc.data = self.W_dec.data.t().clone()

    def encode_pre(self, x):
        return (x / self.scale - self.b_dec) @ self.W_enc.t() + self.b_enc

    def encode(self, x):
        pre = torch.relu(self.encode_pre(x))
        v, i = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, i, v)
        return z

    def decode(self, z):
        return (z @ self.W_dec.t() + self.b_dec) * self.scale

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    @torch.no_grad()
    def normalise_decoder(self):
        n = self.W_dec.data.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.W_dec.data /= n


def train_sae(X: torch.Tensor, n_latents: int, k: int, epochs: int = 60, bs: int = 1024, lr: float = 3e-4,
              seed: int = 0, aux_k: int = 256, aux_coef: float = 1 / 32) -> TopKSAE:
    seed_all(seed)
    d = X.shape[1]
    sae = TopKSAE(d, n_latents, k).cuda()
    sae.scale.fill_(float(X.norm(dim=1).mean() / np.sqrt(d)))
    with torch.no_grad():
        sae.b_dec.data = (X / float(sae.scale)).median(0).values.cuda()
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    last_active = torch.zeros(n_latents, device="cuda")
    step = 0
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for s in range(0, len(X), bs):
            x = X[perm[s:s + bs]].cuda()
            x_hat, z = sae(x)
            err = (x - x_hat) / sae.scale
            loss = (err ** 2).sum(-1).mean()
            # AuxK loss: dead latents (inactive for > 1 epoch-worth of steps) model the residual
            last_active += 1
            last_active[(z > 0).any(0)] = 0
            dead = last_active > (len(X) // bs)
            if dead.any():
                pre = torch.relu(sae.encode_pre(x))
                pre = pre.masked_fill(~dead[None], 0)
                kk = min(aux_k, int(dead.sum()))
                v, i = pre.topk(kk, dim=-1)
                za = torch.zeros_like(pre).scatter_(-1, i, v)
                aux = (za @ sae.W_dec.t())
                loss = loss + aux_coef * ((err.detach() - aux) ** 2).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.normalise_decoder()
            step += 1
    return sae


@torch.no_grad()
def evaluate(sae: TopKSAE, X: torch.Tensor, bs: int = 4096) -> dict:
    fires = torch.zeros(sae.n, device="cuda")
    sse = sst = cos = l0 = 0.0
    mu = X.mean(0).cuda()
    for s in range(0, len(X), bs):
        x = X[s:s + bs].cuda()
        x_hat, z = sae(x)
        sse += ((x - x_hat) ** 2).sum().item()
        sst += ((x - mu) ** 2).sum().item()
        cos += torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).sum().item()
        l0 += (z > 0).sum().item()
        fires += (z > 0).float().sum(0)
    n = len(X)
    freq = fires / n
    return {"explained_variance": 1 - sse / sst, "cosine": cos / n, "L0": l0 / n,
            "dead_frac": float((freq == 0).float().mean()), "dense_frac": float((freq > 0.1).float().mean()),
            "n_eval": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work_dirs/xai/concepts/sae")
    ap.add_argument("--ks", default="16,32,64")
    ap.add_argument("--sizes", default="2048,8192")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--train-score-thr", type=float, default=None,
                    help="default: 0.0 on mini, 0.1 on trainval (runner.sae_train_score_thr)")
    ap.add_argument("--max-train", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.train_score_thr is None:
        from sdxai.runner import sae_train_score_thr
        a.train_score_thr = sae_train_score_thr(os.environ.get("SDXAI_VERSION", "mini"))
    Xtr, mtr = all_instances("train", 6, a.train_score_thr)
    Xva, mva = all_instances("val", 6, 0.1)
    if len(Xtr) > a.max_train:
        Xtr = Xtr[np.random.RandomState(a.seed).choice(len(Xtr), a.max_train, replace=False)]
    Xtr_t, Xva_t = torch.from_numpy(Xtr), torch.from_numpy(Xva)
    results = []
    for n_lat in [int(x) for x in a.sizes.split(",")]:
        for k in [int(x) for x in a.ks.split(",")]:
            t0 = time.time()
            sae = train_sae(Xtr_t, n_lat, k, epochs=a.epochs, seed=a.seed)
            ev = evaluate(sae, Xva_t)
            ev_tr = evaluate(sae, Xtr_t)
            name = f"sae_n{n_lat}_k{k}"
            torch.save({"state_dict": sae.state_dict(), "n_latents": n_lat, "k": k, "d": 256},
                       os.path.join(a.out, name + ".pt"))
            row = {"name": name, "n_latents": n_lat, "k": k, "n_train": len(Xtr),
                   "train_score_thr": a.train_score_thr, "epochs": a.epochs, **ev,
                   "train_explained_variance": ev_tr["explained_variance"], "seconds": time.time() - t0}
            results.append(row)
            print(json.dumps(row), flush=True)
    json.dump(results, open(os.path.join(a.out, "sae_metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
