"""Task 3 follow-up: classical baselines on the same folds as train.py.

Pipelines: csp = CSP(6, ledoit_wolf, log) + LDA;  ts = Covariances(oas) -> TangentSpace -> LogisticRegression.
Alignment (TRANSDUCTIVE, uses unlabelled test-subject trials): ea = Euclidean alignment per subject;
rpa = per-subject re-centering of covariances to identity (ts only). --align-ntrials K estimates the
test subjects' reference from their first K trials only (cross-subject only).

Usage:
  python classical.py --grid                                   # full grid -> results.jsonl
  python classical.py --model csp --channels motor --fband 8 30 --crop 1 4 --permute 1000
"""
import os
import argparse
import itertools
import json
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("NTHREADS", 4)))

from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline

from align import euclidean_align, inv_sqrt
from data import build_dataset, SUBJECTS
from train import CHANNEL_GROUPS

# C3: rim channels (frontal pole, temporal, mastoid-adjacent) most exposed to eye and temporal-muscle EMG.
RIM = ["Fp1", "Fpz", "Fp2", "AF7", "AF8", "F7", "F8", "FT7", "FT8", "T7", "T8", "T9", "T10", "TP7", "TP8"]
GROUPS = dict(CHANNEL_GROUPS, norim="all minus RIM")


class Recenter:
    """RPA-style re-centering: C <- M_s^{-1/2} C M_s^{-1/2}, M_s = Riemannian mean of subject s's covariances."""
    def __init__(self, groups):
        self.groups = groups

    def __call__(self, C, idx):
        out = np.empty_like(C)
        for s in np.unique(self.groups[idx]):
            m = self.groups[idx] == s
            W = inv_sqrt(mean_riemann(C[m]))
            out[m] = W @ C[m] @ W
        return out


def make_model(name):
    if name == "csp":
        return make_pipeline(CSP(n_components=6, reg="ledoit_wolf", log=True),
                             LinearDiscriminantAnalysis())
    return make_pipeline(TangentSpace(metric="riemann"), LogisticRegression(C=1.0, max_iter=1000))


def cv_folds(X, y, groups):
    return [np.unique(groups[te]) for _, te in GroupKFold(n_splits=5).split(X, y, groups)]


def run(X, y, groups, model, split, align, seed, ntrials=None):
    """Returns dict subject -> accuracy (and pooled). Same folds as train.py --split cv."""
    acc_subj, correct, n = {}, 0, 0
    if split == "cv":
        for te_subj in cv_folds(X, y, groups):
            te = np.flatnonzero(np.isin(groups, te_subj)); tr = np.flatnonzero(~np.isin(groups, te_subj))
            Xf = X
            if align == "ea":
                Xf = euclidean_align(X, groups, limit={s: ntrials for s in te_subj} if ntrials else None)
            pred = predict(model, Xf, y, tr, te, groups, align)
            for s in te_subj:
                m = groups[te] == s
                acc_subj[int(s)] = float((pred[m] == y[te][m]).mean())
            correct += int((pred == y[te]).sum()); n += len(te)
    else:
        Xf = euclidean_align(X, groups) if align == "ea" else X
        for s in np.unique(groups):
            idx = np.flatnonzero(groups == s); c = 0
            for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(idx, y[idx]):
                pred = predict(model, Xf, y, idx[a], idx[b], groups, align)
                c += int((pred == y[idx[b]]).sum())
            acc_subj[int(s)] = c / len(idx); correct += c; n += len(idx)
    return acc_subj, correct / n, n


def predict(model, X, y, tr, te, groups, align):
    if model == "csp":
        m = make_model("csp").fit(X[tr].astype(np.float64), y[tr])
        return m.predict(X[te].astype(np.float64))
    C = Covariances("oas").fit_transform(X[np.concatenate([tr, te])].astype(np.float64))
    Ctr, Cte = C[:len(tr)], C[len(tr):]
    if align == "rpa":
        Ctr, Cte = Recenter(groups)(Ctr, tr), Recenter(groups)(Cte, te)
    return make_model("ts").fit(Ctr, y[tr]).predict(Cte)


def log(row, out):
    print("ROW " + json.dumps(row), flush=True)   # also in the log, to recover rows lost to concurrent appends
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true", help="run the full A1 grid")
    ap.add_argument("--model", choices=["csp", "ts"], default="csp")
    ap.add_argument("--channels", choices=list(GROUPS), default="all")
    ap.add_argument("--crop", type=float, nargs=2, default=[0.0, 4.0])
    ap.add_argument("--fband", type=float, nargs=2, default=[1.0, 40.0])
    ap.add_argument("--align", choices=["none", "ea", "rpa"], default="none")
    ap.add_argument("--align-ntrials", type=int, default=None)
    ap.add_argument("--split", choices=["cv", "within"], default="cv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--permute", type=int, default=0, help="label permutations (within subject)")
    ap.add_argument("--tag", default="a1")
    ap.add_argument("--out", default="results.jsonl")
    args = ap.parse_args()

    if args.grid:
        # EOG-free settings (8-30 Hz, 1-4 s) first: they answer the main question.
        configs = list(itertools.product([(8.0, 30.0), (1.0, 40.0)], [(1.0, 4.0), (0.0, 4.0)],
                                         ["all", "motor", "frontal"], ["none", "ea"], ["cv", "within"],
                                         ["csp", "ts"]))
    else:
        configs = [(tuple(args.fband), tuple(args.crop), args.channels, args.align, args.split, args.model)]

    done = set()
    if os.path.exists(args.out):        # resume: skip configs already logged under this tag
        for l in open(args.out):
            r = json.loads(l)
            if r.get("tag") == args.tag and "model" in r and not args.permute:
                done.add((tuple(r["fband"]), tuple(r["crop"]), r["channels"], r["align"], r["split"], r["model"]))
    cache = {}
    for fband, crop, ch, align, split, model in configs:
        if (fband, crop, ch, align, split, model) in done:
            continue
        if (fband, crop) not in cache:
            cache[(fband, crop)] = build_dataset(SUBJECTS, crop, fband)
        X, y, groups, ch_names = cache[(fband, crop)]
        picks = [c for c in ch_names if c not in RIM] if ch == "norim" else CHANNEL_GROUPS[ch]
        Xc = X if picks is None else X[:, [ch_names.index(c) for c in picks]]
        acc, pooled, n = run(Xc, y, groups, model, split, align, args.seed, args.align_ntrials)
        row = dict(tag=args.tag, model=model, split=split, align=align,
                   transductive=align != "none", align_ntrials=args.align_ntrials,
                   channels=ch, crop=list(crop), fband=list(fband), seed=args.seed,
                   test_acc_pooled=pooled, n_test=n, test_acc_per_subject=acc,
                   test_acc_subject_mean=float(np.mean(list(acc.values()))))
        if args.permute:
            rng = np.random.default_rng(args.seed); null = []
            for _ in range(args.permute):
                yp = y.copy()
                for s in np.unique(groups):
                    m = np.flatnonzero(groups == s); yp[m] = rng.permutation(y[m])
                null.append(run(Xc, yp, groups, model, split, align, args.seed, args.align_ntrials)[1])
            null = np.array(null)
            row.update(permutations=args.permute, perm_null_mean=float(null.mean()),
                       perm_null_95=float(np.quantile(null, 0.95)),
                       perm_p=float(((null >= pooled).sum() + 1) / (len(null) + 1)))
        log(row, args.out)
        print(f"{model:3s} {split:6s} align={align:4s} {ch:7s} crop={crop} band={fband}: "
              f"pooled {pooled:.3f}  subject-mean {row['test_acc_subject_mean']:.3f}"
              + (f"  perm p={row['perm_p']:.4f}" if args.permute else ""), flush=True)


if __name__ == "__main__":
    main()
