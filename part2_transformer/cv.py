"""
Subject-rotated evaluation for every Part 2 model (extends the fixed 9-10 split).

  python -u cv.py --models eegnet geom --seeds 0 1 2            # appends tagged rows to results.jsonl
  python -u cv.py --models geom eegnet csp --drop rim           # B2: rim electrodes removed
  python -u cv.py --models geom eegnet csp --tmin 1 --tmax 4    # B2: cue/saccade second removed

Folds: GroupKFold(5) over subjects 1-10, identical to part1_diagnose/train.py --split cv (asserted):
fold k tests 2 subjects, validates on fold (k+1)%5's subjects, trains on the other 6.
Headline = test accuracy at the epoch with the lowest VALIDATION loss. Final-epoch accuracy is logged; peak test
accuracy is logged as a test-selected diagnostic only. Same optimiser settings as train.train_model.
Nothing here changes main.py's behaviour: `python -u main.py --epochs 20 --seeds 0 1 2` still reproduces results.jsonl.
"""
import os
import torch

torch.set_num_threads(int(os.environ.get("NTHREADS", 4)))

import argparse
import json
import time

import numpy as np
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import TensorDataset, DataLoader

import data
from data import get_channel_positions
from models import EEGNet, VanillaEEGTransformer, count_params
from train import set_seed
from main import YourModel, PatchTransformer, channel_names

HERE = os.path.dirname(os.path.abspath(__file__))
SUBJECTS = list(range(1, 11))
RIM = ["Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F8", "FT7", "FT8", "T7", "T8",
       "T9", "T10", "TP7", "TP8", "P7", "P8", "PO7", "PO8", "O1", "Oz", "O2", "Iz"]


def load_all(tmin, tmax):
    """X (N, 64, T), y, groups (subject id per trial); per-subject epochs cached in scratch/."""
    path = os.path.join(HERE, "scratch", f"cache_t{tmin:g}_{tmax:g}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["X"], d["y"], d["groups"]
    data.TMIN, data.TMAX = tmin, tmax
    Xs, ys, gs = [], [], []
    for s in SUBJECTS:
        X, y = data._load_subjects([s])
        Xs.append(X); ys.append(y); gs.append(np.full(len(y), s))
    X, y, groups = np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, X=X, y=y, groups=groups)
    return X, y, groups


def make_folds(groups):
    folds = [sorted(np.unique(groups[te]).tolist())
             for _, te in GroupKFold(n_splits=5).split(groups, groups, groups)]
    # Part 1's CV uses 45 trials per subject in subject order; GroupKFold depends only on the groups vector.
    p1 = np.repeat(SUBJECTS, 45)
    p1_folds = [sorted(np.unique(p1[te]).tolist())
                for _, te in GroupKFold(n_splits=5).split(p1, p1, p1)]
    assert np.bincount(groups)[1:].tolist() == [45] * 10, np.bincount(groups)
    assert folds == p1_folds, (folds, p1_folds)
    return folds


def euclidean_align(X, groups):
    """TRANSDUCTIVE: whiten each subject by its own mean trial covariance (uses unlabelled test-subject data)."""
    X = X.copy()
    for s in np.unique(groups):
        m = groups == s
        R = np.mean([x @ x.T / x.shape[1] for x in X[m]], axis=0)
        w, V = np.linalg.eigh(R)
        X[m] = np.einsum("ij,njt->nit", (V / np.sqrt(np.maximum(w, 1e-10))) @ V.T, X[m])
    return X.astype(np.float32)


def build(name, C, T, pos, seed):
    """Returns (model, extra-log-fields). 'shuffled' = geom PE with a fresh electrode permutation per seed."""
    if name == "eegnet":
        return EEGNet(C, T), {}
    if name == "eegnet_p1":
        return EEGNet(C, T, kern=80, p=0.5), {}
    if name == "vanilla":
        return VanillaEEGTransformer(C, T), {}
    if name == "patch":
        return PatchTransformer(C, T), {}
    if name in ("geom", "learned", "none"):
        return YourModel(C, T, pos=pos, pe=name), {}
    if name == "shuffled":
        perm = np.random.default_rng(1000 + seed).permutation(C)
        return YourModel(C, T, pos=pos[perm], pe="geom"), {"perm_seed": 1000 + seed}
    if name in ("distbias", "distbias_shuf"):
        from variants import YourModelDistBias
        perm = np.random.default_rng(2000 + seed).permutation(C) if name == "distbias_shuf" else None
        return YourModelDistBias(C, T, pos=pos, dist_perm=perm), (
            {"perm_seed": 2000 + seed} if perm is not None else {})
    if name == "tokens_noattn":
        from variants import TokensNoAttn
        return TokensNoAttn(C, T, pos=pos), {}
    raise ValueError(name)


@torch.no_grad()
def predict(model, X, bs=256):
    model.eval()
    return torch.cat([model(torch.from_numpy(X[i:i + bs])) for i in range(0, len(X), bs)])


def fit_nn(model, Xtr, ytr, Xva, yva, Xte, yte, epochs):
    """train.train_model's settings (AdamW 1e-3/1e-3, batch 64), checkpoint chosen on validation loss."""
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=64, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best = (float("inf"), -1, None)
    accs, t_train = [], 0.0
    for ep in range(epochs):
        model.train(); t0 = time.time()
        for xb, yb in loader:
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
        t_train += time.time() - t0
        va_loss = loss_fn(predict(model, Xva), torch.from_numpy(yva)).item()
        accs.append((predict(model, Xte).argmax(1).numpy() == yte).mean())
        if va_loss < best[0]:
            best = (va_loss, ep, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    t0 = time.time(); pred = predict(model, Xte).argmax(1).numpy()
    infer_ms = (time.time() - t0) * 1000 / len(Xte)
    return pred, dict(best_epoch=best[1], acc_final=float(accs[-1]), acc_peak_test_selected=float(max(accs)),
                      sec_per_epoch=t_train / epochs, infer_ms_per_trial=infer_ms, params=count_params(model))


def fit_csp(Xtr, ytr, Xte):
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    clf = make_pipeline(CSP(n_components=6, reg="ledoit_wolf", log=True), LinearDiscriminantAnalysis())
    t0 = time.time(); clf.fit(Xtr.astype(np.float64), ytr); t_fit = time.time() - t0
    t0 = time.time(); pred = clf.predict(Xte.astype(np.float64))
    return pred, dict(best_epoch=None, acc_final=None, acc_peak_test_selected=None, sec_per_epoch=t_fit,
                      infer_ms_per_trial=(time.time() - t0) * 1000 / len(Xte), params=None)


def fit_ts(Xtr, ytr, Xte, gtr, gte, rpa):
    """Covariances(oas) -> TangentSpace(riemann) -> LogisticRegression, as part1_diagnose/classical.py.
    rpa=True re-centres each subject's covariances on their own Riemannian mean (TRANSDUCTIVE for test subjects)."""
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from pyriemann.utils.mean import mean_riemann
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    def recenter(Cs, g):
        out = np.empty_like(Cs)
        for s in np.unique(g):
            w, V = np.linalg.eigh(mean_riemann(Cs[g == s]))
            W = (V / np.sqrt(w)) @ V.T
            out[g == s] = W @ Cs[g == s] @ W
        return out

    t0 = time.time()
    cov = Covariances(estimator="oas")
    Ctr, Cte = cov.fit_transform(Xtr.astype(np.float64)), cov.transform(Xte.astype(np.float64))
    if rpa:
        Ctr, Cte = recenter(Ctr, gtr), recenter(Cte, gte)
    clf = make_pipeline(TangentSpace(metric="riemann"), LogisticRegression(C=1.0, max_iter=1000))
    clf.fit(Ctr, ytr); t_fit = time.time() - t0
    t0 = time.time(); pred = clf.predict(Cte)
    return pred, dict(best_epoch=None, acc_final=None, acc_peak_test_selected=None, sec_per_epoch=t_fit,
                      infer_ms_per_trial=(time.time() - t0) * 1000 / len(Xte), params=None)


MOTOR = ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
         "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"]
CLASSICAL = ("csp", "ts", "ts_rpa")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="eegnet eegnet_p1 vanilla patch geom learned none shuffled distbias distbias_shuf tokens_noattn"
                         " csp ts ts_rpa")
    ap.add_argument("--channels", choices=["all", "motor"], default="all")
    ap.add_argument("--classical-train", choices=["train", "trainval"], default="train",
                    help="classical models (no checkpoint selection) fit on the 6 train or all 8 non-test subjects")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=3.0)
    ap.add_argument("--drop", choices=["none", "rim"], default="none")
    ap.add_argument("--align", choices=["none", "ea"], default="none")
    ap.add_argument("--tag", default="B1")
    ap.add_argument("--out", default=os.path.join(HERE, "results.jsonl"))
    ap.add_argument("--save-models", nargs="*", default=[],
                    help="model names whose val-selected weights are saved to scratch/models/ (for analysis.py)")
    args = ap.parse_args()

    X, y, groups = load_all(args.tmin, args.tmax)
    assert X.shape[2] == 481, X.shape
    pos, names = get_channel_positions(), channel_names()
    keep = np.arange(X.shape[1])
    if args.drop == "rim":
        low = [n.lower() for n in names]
        rim_idx = [low.index(r.lower()) for r in RIM]       # raises if any rim name is missing
        keep = np.array([i for i in keep if i not in rim_idx])
    if args.channels == "motor":
        low = [n.lower() for n in names]
        keep = np.array([i for i in keep if i in [low.index(m.lower()) for m in MOTOR]])
        assert len(keep) == len(MOTOR) or args.drop != "none"
    X, pos = X[:, keep], pos[keep]
    if args.align == "ea":
        X = euclidean_align(X, groups)
    folds = make_folds(groups)
    C, T = X.shape[1], X.shape[2]
    print(f"threads {torch.get_num_threads()}  X {X.shape}  folds {folds}  drop={args.drop}  "
          f"t={args.tmin:g}-{args.tmax:g}s  align={args.align}"
          + ("  [TRANSDUCTIVE]" if args.align != "none" else ""), flush=True)

    for name in args.models:
        for seed in args.seeds:
            for k in args.folds:
                te_s, va_s = folds[k], folds[(k + 1) % 5]
                tr_s = [s for s in SUBJECTS if s not in te_s + va_s]
                tr, va, te = (np.isin(groups, g) for g in (tr_s, va_s, te_s))
                mu = X[tr].mean(axis=(0, 2), keepdims=True)
                sd = X[tr].std(axis=(0, 2), keepdims=True) + 1e-6
                Z = ((X - mu) / sd).astype(np.float32)      # training-subject statistics only
                set_seed(seed)
                t0 = time.time()
                if name in CLASSICAL:
                    fit = tr | va if args.classical_train == "trainval" else tr
                    if name == "csp":
                        pred, info = fit_csp(Z[fit], y[fit], Z[te])
                    else:
                        pred, info = fit_ts(Z[fit], y[fit], Z[te], groups[fit], groups[te], rpa=name == "ts_rpa")
                    extra = {"classical_train": args.classical_train}
                else:
                    model, extra = build(name, C, T, pos, seed)
                    pred, info = fit_nn(model, Z[tr], y[tr], Z[va], y[va], Z[te], y[te], args.epochs)
                    if name in args.save_models:
                        d = os.path.join(HERE, "scratch", "models"); os.makedirs(d, exist_ok=True)
                        torch.save(model.state_dict(), os.path.join(
                            d, f"{args.tag}_{name}_s{seed}_f{k}.pt"))
                yte, gte = y[te], groups[te]
                row = dict(exp=args.tag, model=name, seed=seed, fold=k, test_subjects=te_s, val_subjects=va_s,
                           acc=float((pred == yte).mean()),
                           per_subject={int(s): float((pred[gte == s] == yte[gte == s]).mean()) for s in te_s},
                           n_test=int(te.sum()), epochs=args.epochs if name not in CLASSICAL else None,
                           tmin=args.tmin, tmax=args.tmax, drop=args.drop, n_chan=C, align=args.align,
                           transductive=args.align != "none" or name == "ts_rpa", **info, **extra)
                if args.channels != "all":
                    row["channels"] = args.channels
                with open(args.out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"  [{args.tag} {name}] seed {seed} fold {k} test {te_s}: acc(val-sel) {row['acc']:.3f}  "
                      f"final {info['acc_final'] if info['acc_final'] is not None else float('nan'):.3f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
