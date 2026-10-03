"""Dictionary learning on decoder-layer-6 instance features with the Overcomplete toolbox
(KempnerInstitute/overcomplete v0.3.0, Fel et al.), side by side with sdxai's own TopK SAE.

Data: train = all train-split slots (score >= 0, see sae.py for why on v1.0-mini); eval = val slots
with score > 0.1. Inputs are standardised per dimension with train statistics (the library's
JumpSAE docstring recommends this); every metric is computed after mapping reconstructions back to
the original feature space, so numbers are comparable with sae.py.

Families (all from overcomplete):
  neural SAEs, trained with overcomplete.sae.train_sae (non-AMP; train_sae_amp needs torch>=2.3):
    SAE (ReLU, L1), TopKSAE (encoders: linear | mlp_ln_1 | tied), BatchTopKSAE, JumpSAE, MpSAE,
    OMPSAE, QSAE, RATopKSAE, RAJumpSAE (archetypal points = k-means centroids of the train set)
  optimisation-based: SkPCA, SkICA, SkSVD, SkSparsePCA, SkKMeans, SkDictionaryLearning,
    SemiNMF, ConvexNMF, ArchetypalAnalysis, NMF and SkNMF (the latter two need non-negative data:
    fitted on the split embedding [relu(x), relu(-x)], reconstruction x = x+ - x-)
Metrics (overcomplete.metrics): r2_score, relative_avg_l2_loss, l0, hoyer, kappa_4, dead_codes,
  dictionary_collinearity, codes_correlation_matrix (max off-diagonal), energy_of_codes
  (share of the top-10 concepts), wasserstein_1d and frechet_distance between x and x_hat;
  plus explained variance and cosine as in sae.py.
Stability: for TopKSAE and RATopKSAE, two seeds x two disjoint halves of the train data;
  overcomplete.metrics.cosine_hungarian_loss between the dictionaries (lower = more stable).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import overcomplete.metrics as OM
from overcomplete.optimization import (ConvexNMF, NMF, SemiNMF, SkDictionaryLearning, SkICA, SkKMeans, SkNMF,
                                       SkPCA, SkSparsePCA, SkSVD)
from overcomplete.optimization.archetypal_analysis import ArchetypalAnalysis
from overcomplete.sae import (SAE, BatchTopKSAE, JumpSAE, MpSAE, OMPSAE, QSAE, RAJumpSAE, RATopKSAE, TopKSAE,
                              CosineScheduler, train_sae)
from overcomplete.sae.losses import mse_l1, top_k_auxiliary_loss

from sdxai.concepts.data import all_instances
from sdxai.runner import seed_all

DEV = "cuda"
D = 256


def _proj_simplex(Z: torch.Tensor) -> torch.Tensor:
    """Exact Euclidean projection of each row onto the probability simplex (Duchi et al. 2008)."""
    u, _ = torch.sort(Z, dim=1, descending=True)
    css = u.cumsum(1) - 1
    idx = torch.arange(1, Z.shape[1] + 1, device=Z.device, dtype=Z.dtype)
    cond = u - css / idx > 0
    rho = cond.float().cumsum(1).argmax(1, keepdim=True)
    theta = css.gather(1, rho) / (rho + 1).to(Z.dtype)
    return torch.clamp(Z - theta, min=0)


def pgd_codes(Dct: torch.Tensor, X: torch.Tensor, simplex: bool, iters: int = 300) -> torch.Tensor:
    """min_Z ||Z D - X||^2, Z >= 0 (ConvexNMF) or rows of Z on the simplex (archetypal analysis).
    Used for out-of-sample coding: the library's encode() for these two is transductive (its W is
    tied to the fit set)."""
    G = Dct @ Dct.T
    L = torch.linalg.matrix_norm(G, ord=2)
    B = X @ Dct.T
    Z = torch.full((X.shape[0], Dct.shape[0]), 1.0 / Dct.shape[0], device=X.device)
    for _ in range(iters):
        Z = Z - (Z @ G - B) / L
        Z = _proj_simplex(Z) if simplex else torch.clamp(Z, min=0)
    return Z


class DictModel:
    """Uniform wrapper: standardise -> (split) -> encode/decode -> unstandardise."""

    def __init__(self, name, family, model, mean, std, split=False, neural=True):
        self.name, self.family, self.model = name, family, model
        self.mean, self.std, self.split, self.neural = mean, std, split, neural

    def _in(self, x):
        xs = (x - self.mean) / self.std
        if self.split:
            xs = torch.cat([torch.relu(xs), torch.relu(-xs)], -1)
        return xs

    def _out(self, xh):
        if self.split:
            xh = xh[:, :D] - xh[:, D:]
        return xh * self.std + self.mean

    @torch.no_grad()
    def codes(self, x: torch.Tensor) -> torch.Tensor:
        xs = self._in(x.to(DEV))
        if self.neural:
            return self.model.encode(xs)[1]
        if self.family == "SkKMeans":  # library encode returns distances; code = nearest centroid
            dist = self.model.encode(xs.cpu()).to(DEV)
            return torch.nn.functional.one_hot(dist.argmin(1), self.model.nb_concepts).float()
        if self.family in ("ConvexNMF", "ArchetypalAnalysis"):
            return pgd_codes(self.dictionary().to(DEV), xs, simplex=self.family == "ArchetypalAnalysis")
        dev = getattr(self.model, "device", "cpu")
        return self.model.encode(xs.to(dev) if dev != "cpu" else xs.cpu()).to(DEV)

    @torch.no_grad()
    def recon_from_codes(self, z: torch.Tensor) -> torch.Tensor:
        if self.family in ("ConvexNMF", "ArchetypalAnalysis"):
            xh = z.to(DEV) @ self.dictionary().to(DEV)
        else:
            dev = getattr(self.model, "device", "cpu")
            xh = self.model.decode(z.to(dev) if dev != "cpu" else z.cpu())
        return self._out(xh.to(DEV).float())

    @torch.no_grad()
    def recon(self, x: torch.Tensor, bs: int = 8192) -> torch.Tensor:
        outs = []
        for s in range(0, len(x), bs):
            outs.append(self.recon_from_codes(self.codes(x[s:s + bs])))
        return torch.cat(outs)

    def dictionary(self) -> torch.Tensor:
        return self.model.get_dictionary().detach().float()


# ------------------------------------------------------------------------------------------
def metrics(dm: DictModel, X: torch.Tensor, bs: int = 8192) -> dict:
    zs, xhs = [], []
    for s in range(0, len(X), bs):
        z = dm.codes(X[s:s + bs]).float()
        zs.append(z.cpu())
        xhs.append(dm.recon_from_codes(z.to(DEV) if dm.neural else z).cpu())
    Z, Xh = torch.cat(zs), torch.cat(xhs)
    Xc = X.cpu()
    ev = 1 - ((Xc - Xh) ** 2).sum() / ((Xc - Xc.mean(0)) ** 2).sum()
    Dct = dm.dictionary().cpu()
    freq = (Z.abs() > 1e-8).float().mean(0)
    out = {
        "explained_variance": float(ev),
        "r2_score": float(OM.r2_score(Xc, Xh)),
        "relative_l2": float(OM.relative_avg_l2_loss(Xc, Xh)),
        "cosine": float(torch.nn.functional.cosine_similarity(Xc, Xh, dim=-1).mean()),
        "L0": float((Z.abs() > 1e-8).float().sum(1).mean()),  # count (OM.l0_eps returns a fraction)
        "hoyer": float(OM.hoyer(Z).mean()),
        "kappa_4": float(OM.kappa_4(Z).mean()),
        "dead_frac": float(OM.dead_codes(Z).float().mean()),
        "dense_frac": float((freq > 0.1).float().mean()),
        "dict_max_collinearity": float(OM.dictionary_collinearity(Dct)[0]),
        "codes_max_corr": float(OM.codes_correlation_matrix(Z[:20000])[0]),
        "energy_top10_share": _energy_top10(Z, Dct),
        "wasserstein_1d": float(OM.wasserstein_1d(Xc, Xh)),
        "frechet_distance": float(OM.frechet_distance(Xc, Xh)),
        "n_concepts": int(Z.shape[1]),
    }
    return out


def _energy_top10(Z, Dct):
    if Dct.shape[0] != Z.shape[1]:
        return float("nan")
    e = OM.energy_of_codes(Z, Dct)
    return float(torch.sort(e, descending=True)[0][:10].sum() / (e.sum() + 1e-12))


def loader(Xs: torch.Tensor, bs: int = 1024, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(Xs), batch_size=bs, shuffle=True, drop_last=True, generator=g)


def train_neural(model, Xs, criterion, epochs, lr=5e-4, bs=1024, seed=0):
    dl = loader(Xs, bs, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    total = epochs * len(dl)
    sch = CosineScheduler(opt, base_value=lr, final_value=lr * 0.05, total_iters=total, warmup_iters=min(500, total // 10))
    logs = train_sae(model, dl, criterion, opt, scheduler=sch, nb_epochs=epochs, clip_grad=1.0, monitoring=1, device=DEV)
    model.eval()
    return {k: v for k, v in logs.items() if k in ("avg_loss", "r2", "z_sparsity", "dead_features")}


def kmeans_points(Xs: torch.Tensor, n: int, seed: int = 0) -> torch.Tensor:
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=n, random_state=seed, batch_size=4096, n_init=3).fit(Xs.cpu().numpy())
    return torch.tensor(km.cluster_centers_, dtype=torch.float32, device=DEV)


def build_neural(spec: dict, n: int, k: int, points=None):
    kind = spec["kind"]
    if kind == "SAE":
        return SAE(D, n, device=DEV)
    if kind == "TopKSAE":
        m = TopKSAE(D, n, top_k=k, encoder_module=spec.get("encoder"), device=DEV)
        return m.tied() if spec.get("tied") else m
    if kind == "BatchTopKSAE":
        return BatchTopKSAE(D, n, top_k=k * spec["bs"], device=DEV)
    if kind == "JumpSAE":
        return JumpSAE(D, n, bandwidth=spec.get("bandwidth", 1e-3), device=DEV)
    if kind == "MpSAE":
        return MpSAE(D, n, k=k, device=DEV)
    if kind == "OMPSAE":
        return OMPSAE(D, n, k=k, max_iter=10, device=DEV)
    if kind == "QSAE":
        return QSAE(D, n, q=4, device=DEV)
    if kind == "RATopKSAE":
        return RATopKSAE(D, n, points=points, top_k=k, delta=1.0, device=DEV)
    if kind == "RAJumpSAE":
        return RAJumpSAE(D, n, points=points, delta=1.0, device=DEV)
    raise ValueError(kind)


NEURAL = [
    # name, spec, n, k, loss, epochs
    ("oc_SAE_relu_l1_n2048", {"kind": "SAE"}, 2048, None, "l1:0.05", 20),
    ("oc_TopK_n2048_k16", {"kind": "TopKSAE"}, 2048, 16, "auxk", 20),
    ("oc_TopK_n2048_k32", {"kind": "TopKSAE"}, 2048, 32, "auxk", 20),
    ("oc_TopK_n2048_k64", {"kind": "TopKSAE"}, 2048, 64, "auxk", 20),
    ("oc_TopK_n8192_k64", {"kind": "TopKSAE"}, 8192, 64, "auxk", 20),
    ("oc_TopK_mlp_ln_1_n2048_k64", {"kind": "TopKSAE", "encoder": "mlp_ln_1"}, 2048, 64, "auxk", 20),
    ("oc_TopK_tied_n2048_k64", {"kind": "TopKSAE", "tied": True}, 2048, 64, "auxk", 20),
    ("oc_BatchTopK_n2048_k64", {"kind": "BatchTopKSAE", "bs": 1024}, 2048, 64, "auxk", 20),
    ("oc_Jump_n2048", {"kind": "JumpSAE"}, 2048, None, "l1:0.02", 20),
    ("oc_MP_n2048_k32", {"kind": "MpSAE"}, 2048, 32, "l1:0.0", 10),
    ("oc_OMP_n2048_k32", {"kind": "OMPSAE"}, 2048, 32, "l1:0.0", 5),
    ("oc_QSAE_n2048", {"kind": "QSAE"}, 2048, None, "l1:0.05", 20),
    ("oc_RATopK_n2048_k64", {"kind": "RATopKSAE"}, 2048, 64, "auxk", 20),
    ("oc_RAJump_n2048", {"kind": "RAJumpSAE"}, 2048, None, "l1:0.02", 20),
]

OPTIM = [
    # name, factory, fit subsample, split
    ("oc_PCA_64", lambda: SkPCA(64), 50000, False),
    ("oc_ICA_64", lambda: SkICA(64, max_iter=500), 50000, False),
    ("oc_SVD_64", lambda: SkSVD(64), 50000, False),
    ("oc_SparsePCA_64", lambda: SkSparsePCA(64, sparsity=1.0), 10000, False),
    ("oc_KMeans_512", lambda: SkKMeans(512, n_init=3, random_state=0), 50000, False),
    ("oc_DictLearning_512", lambda: SkDictionaryLearning(512, sparsity=0.1, max_iter=30), 10000, False),
    ("oc_SemiNMF_256", lambda: SemiNMF(256, solver="mu", device=DEV), 50000, False),
    ("oc_ConvexNMF_256", lambda: ConvexNMF(256, solver="pgd", device=DEV), 20000, False),
    ("oc_ArchetypalAnalysis_256", lambda: ArchetypalAnalysis(256, device=DEV), 20000, False),
    ("oc_NMF_split_256", lambda: NMF(256, solver="hals", device=DEV), 50000, True),
    ("oc_SkNMF_split_256", lambda: SkNMF(256, max_iter=300, init="nndsvda"), 20000, True),
]


def criterion_for(loss: str):
    if loss == "auxk":
        return lambda *a: top_k_auxiliary_loss(*a, penalty=0.1)
    pen = float(loss.split(":")[1])
    return lambda *a: mse_l1(*a, penalty=pen)


def fit_optim(model, Xs_fit):
    if isinstance(model, ArchetypalAnalysis):
        # The library's aa_pgd_solver re-applies project_simplex (a softmax, not a Euclidean
        # projection) after every Adam step; at the class default lr=1e-2 this flattens rows of
        # W towards uniform and the archetypes collapse to the data mean (EV ~ 0 on our data).
        # lr=1.0 (not exposed by ArchetypalAnalysis.fit) partly overcomes it; same solver otherwise.
        from overcomplete.optimization.archetypal_analysis import aa_pgd_solver
        A = Xs_fit.to(DEV)
        Z, W = aa_pgd_solver(A, model.init_random_z(A), model.init_random_w(A), lr=1.0, max_iter=1000, tol=0.0)
        model.register_buffer("W", W)
        model.register_buffer("D", W @ A)
        model._set_fitted()
        return model
    if isinstance(model, (SemiNMF, ConvexNMF, NMF)):
        model.fit(Xs_fit.to(DEV), max_iter=300)
    else:
        model.fit(Xs_fit.cpu())
    return model


L1_FAMILY = [("oc_SAE_relu_l1_n2048", "SAE"), ("oc_Jump_n2048", "JumpSAE"),
             ("oc_QSAE_n2048", "QSAE"), ("oc_RAJump_n2048", "RAJumpSAE")]
# overcomplete's mse_l1 = mean-squared error (per element) + penalty * SUM of |codes| over the whole
# batch (metrics.l1 with dims=None), so the effective per-code weight is penalty * batch * n_concepts
# (~2e6 x penalty here); grids in the usual 1e-4..1e-2 range collapse every code to zero.
L1_GRID = [1e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5]


def l1_sweep(a, Xs, Xva, Xtr, mean, std, results, res_path):
    points = None
    sweep = {}
    for base, kind in L1_FAMILY:
        trials = []
        for pen in L1_GRID:
            seed_all(a.seed)
            if kind.startswith("RA") and points is None:
                points = kmeans_points(Xs, 4096, a.seed)
            m = build_neural({"kind": kind}, 2048, None, points)
            train_neural(m, Xs, criterion_for(f"l1:{pen}"), 5, seed=a.seed)
            met = metrics(DictModel(base, kind, m, mean, std), Xva)
            trials.append({"penalty": pen, "EV": met["explained_variance"], "L0": met["L0"], "dead": met["dead_frac"]})
            print("sweep", base, trials[-1], flush=True)
        ok = [t for t in trials if t["L0"] <= 64]
        best = max(ok, key=lambda t: t["EV"]) if ok else min(trials, key=lambda t: t["L0"])
        sweep[base] = {"trials": trials, "chosen": best}
        name = base + "_tuned"
        t0 = time.time()
        seed_all(a.seed)
        m = build_neural({"kind": kind}, 2048, None, points)
        logs = train_neural(m, Xs, criterion_for(f"l1:{best['penalty']}"), 20, seed=a.seed)
        dm = DictModel(name, kind, m, mean, std)
        met = metrics(dm, Xva)
        results[name] = {"family": "neural", "kind": kind, "n": 2048, "k": None, "loss": f"l1:{best['penalty']}",
                         "epochs": 20, **met, "train_explained_variance": metrics(dm, Xtr[:50000])["explained_variance"],
                         "final_train_log": {kk: v[-1] for kk, v in logs.items() if v}, "seconds": time.time() - t0}
        torch.save(dm, os.path.join(a.out, name + ".pt"))
        print(name, {k: round(results[name][k], 4) for k in ("explained_variance", "L0", "dead_frac")}, flush=True)
        json.dump(results, open(res_path, "w"), indent=2)
    json.dump(sweep, open(os.path.join(a.out, "l1_sweep.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work_dirs/xai/concepts/overcomplete")
    ap.add_argument("--only", default="", help="comma list of names to run")
    ap.add_argument("--max-train", type=int, default=5_000_000)
    ap.add_argument("--skip-stability", action="store_true")
    ap.add_argument("--l1-sweep", action="store_true",
                    help="tune the L1 penalty of the L1-family SAEs (SAE, JumpSAE, QSAE, RAJumpSAE): 5-epoch "
                         "sweep, pick max val EV subject to L0 <= 64 (else min L0), retrain 20 epochs as *_tuned")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seed_all(a.seed)
    from sdxai.runner import sae_train_score_thr
    Xtr, _ = all_instances("train", 6, sae_train_score_thr(os.environ.get("SDXAI_VERSION", "mini")))
    if len(Xtr) > a.max_train:
        Xtr = Xtr[np.random.RandomState(a.seed).choice(len(Xtr), a.max_train, replace=False)]
    Xva, _ = all_instances("val", 6, 0.1)
    Xtr, Xva = torch.from_numpy(Xtr).to(DEV), torch.from_numpy(Xva).to(DEV)
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xs = (Xtr - mean) / std
    torch.save({"mean": mean.cpu(), "std": std.cpu()}, os.path.join(a.out, "standardisation.pt"))
    only = set(a.only.split(",")) if a.only else None
    res_path = os.path.join(a.out, "overcomplete_metrics.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    points = None

    if a.l1_sweep:
        l1_sweep(a, Xs, Xva, Xtr, mean, std, results, res_path)
        return

    for name, spec, n, k, loss, ep in NEURAL:
        if only and name not in only:
            continue
        t0 = time.time()
        try:
            seed_all(a.seed)
            if spec["kind"].startswith("RA") and points is None:
                points = kmeans_points(Xs, 4096, a.seed)
            model = build_neural(spec, n, k, points)
            logs = train_neural(model, Xs, criterion_for(loss), ep, seed=a.seed)
            dm = DictModel(name, spec["kind"], model, mean, std)
            m = metrics(dm, Xva)
            m_tr = metrics(dm, Xtr[:50000])
            results[name] = {"family": "neural", "kind": spec["kind"], "n": n, "k": k, "loss": loss, "epochs": ep,
                             **m, "train_explained_variance": m_tr["explained_variance"],
                             "final_train_log": {kk: v[-1] for kk, v in logs.items() if v},
                             "seconds": time.time() - t0}
            torch.save(dm, os.path.join(a.out, name + ".pt"))
        except Exception as e:  # record and continue: one method failing must not stop the suite
            results[name] = {"family": "neural", "error": f"{type(e).__name__}: {e}",
                             "trace": traceback.format_exc()[-1500:]}
        print(name, json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in results[name].items()
                                if kk in ("explained_variance", "L0", "dead_frac", "hoyer", "error")}), flush=True)
        json.dump(results, open(res_path, "w"), indent=2)

    for name, factory, nfit, split in OPTIM:
        if only and name not in only:
            continue
        t0 = time.time()
        try:
            seed_all(a.seed)
            idx = torch.randperm(len(Xs), device=DEV)[:nfit]
            Xf = Xs[idx]
            if split:
                Xf = torch.cat([torch.relu(Xf), torch.relu(-Xf)], -1)
            model = fit_optim(factory(), Xf)
            dm = DictModel(name, type(model).__name__, model, mean, std, split=split, neural=False)
            m = metrics(dm, Xva)
            results[name] = {"family": "optimisation", "kind": type(model).__name__, "n_fit": nfit,
                             "split_nonneg": split, **m, "seconds": time.time() - t0}
            torch.save(dm, os.path.join(a.out, name + ".pt"))
        except Exception as e:
            results[name] = {"family": "optimisation", "error": f"{type(e).__name__}: {e}",
                             "trace": traceback.format_exc()[-1500:]}
        print(name, json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in results[name].items()
                                if kk in ("explained_variance", "L0", "dead_frac", "error")}), flush=True)
        json.dump(results, open(res_path, "w"), indent=2)

    if not a.skip_stability:
        stab = {}
        half = len(Xs) // 2
        perm = torch.randperm(len(Xs), generator=torch.Generator().manual_seed(123)).to(DEV)
        halves = [Xs[perm[:half]], Xs[perm[half:]]]
        for kind in ["TopKSAE", "RATopKSAE"]:
            dicts = {}
            for hi, Xh in enumerate(halves):
                for sd in [0, 1]:
                    seed_all(sd)
                    pts = kmeans_points(Xh, 4096, sd) if kind == "RATopKSAE" else None
                    mdl = build_neural({"kind": kind}, 2048, 64, pts)
                    train_neural(mdl, Xh, criterion_for("auxk"), 10, seed=sd)
                    dicts[(hi, sd)] = mdl.get_dictionary().detach().float()
            stab[kind] = {
                "same_data_diff_seed": float(OM.cosine_hungarian_loss(dicts[(0, 0)], dicts[(0, 1)])),
                "diff_data_same_seed": float(OM.cosine_hungarian_loss(dicts[(0, 0)], dicts[(1, 0)])),
                "diff_data_diff_seed": float(OM.cosine_hungarian_loss(dicts[(0, 0)], dicts[(1, 1)])),
            }
            print("stability", kind, stab[kind], flush=True)
        json.dump(stab, open(os.path.join(a.out, "stability.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
