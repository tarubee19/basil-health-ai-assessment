"""Train and evaluate a hands-vs-feet decoder on unseen subjects.

Usage:
  python train.py --channels all --seed 0
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

from data import build_dataset, SUBJECTS, CROP, L_FREQ, H_FREQ
from model import EEGNet, count_params

# Electrode groups for the channel-subset run (Task 3).
# frontal: the 8 electrodes nearest the eyes (Fp/AF rows), the best EOG proxy we have (no EOG
#          channel is recorded). If these alone decode the task, eye movements are a viable shortcut.
# motor:   control group over sensorimotor cortex (FC/C/CP rows) where hand/foot ERD should live.
CHANNEL_GROUPS = {
    "all": None,
    "frontal": ["Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8"],
    "motor": ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
              "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
              "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
}


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)


def evaluate(model, X, y, bs=256):
    model.eval(); correct = 0
    with torch.no_grad():
        for i in range(0, len(y), bs):
            xb = torch.from_numpy(X[i:i + bs]); yb = torch.from_numpy(y[i:i + bs])
            correct += (model(xb).argmax(1) == yb).sum().item()
    return correct / len(y)


def mean_loss(model, X, y, loss_fn, bs=256):
    model.eval(); total = 0.0
    with torch.no_grad():
        for i in range(0, len(y), bs):
            xb = torch.from_numpy(X[i:i + bs]); yb = torch.from_numpy(y[i:i + bs])
            total += loss_fn(model(xb), yb).item() * len(yb)
    return total / len(y)


def run_split(X, y, groups, tr_subj, va_subj, te_subj, args):
    """Train on tr_subj, select on va_subj, evaluate once on te_subj. Returns a result row."""
    set_seed(args.seed)
    # FIX (A): split by SUBJECT, not trial. Random trial splits put every person in both
    # train and test, so the model could score by recognising people it had seen -> inflated accuracy.
    assert not (tr_subj & te_subj) and not (tr_subj & va_subj) and not (va_subj & te_subj)
    assert tr_subj and va_subj and te_subj
    tr = np.flatnonzero(np.isin(groups, list(tr_subj)))
    va = np.flatnonzero(np.isin(groups, list(va_subj)))
    te = np.flatnonzero(np.isin(groups, list(te_subj)))
    print(f"train subjects {sorted(tr_subj)}  val subjects {sorted(va_subj)}  "
          f"test subjects {sorted(te_subj)}")

    # FIX (C): z-score with TRAINING-subject statistics only (starter used all trials, test
    # included, before the split -> test scale leaked into inputs, small inflation).
    # Per-subject z-scoring is a legitimate alternative, but it uses the test subject's
    # unlabelled statistics, so it must be declared as such.
    mean = X[tr].mean(axis=(0, 2), keepdims=True)
    std = X[tr].std(axis=(0, 2), keepdims=True) + 1e-6
    X = ((X - mean) / std).astype(np.float32)

    n_ch, n_t = X.shape[1], X.shape[2]
    model = EEGNet(n_ch, n_t)
    print(f"EEGNet: {count_params(model)} params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(args.seed)

    # FIX (B): select the checkpoint on VALIDATION-subject loss. The starter kept argmax of
    # test accuracy over 30 epochs (max of a noisy curve) -> optimistic, inflated number.
    best_val_loss, best_state, best_ep = float("inf"), None, -1
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), args.bs):
            b = order[i:i + args.bs]
            xb, yb = torch.from_numpy(X[b]), torch.from_numpy(y[b])
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        va_acc = evaluate(model, X[va], y[va])
        va_loss = mean_loss(model, X[va], y[va], loss_fn)
        te_acc = evaluate(model, X[te], y[te])   # printed only; never used for selection
        if va_loss < best_val_loss:   # keep the checkpoint with the lowest validation loss
            best_val_loss, best_ep = va_loss, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep:3d}  loss {loss.item():.3f}  val_loss {va_loss:.3f}  "
              f"val {va_acc:.3f}  test {te_acc:.3f}")

    test_acc_last = te_acc
    model.load_state_dict(best_state)
    per_subject = {int(s): evaluate(model, X[te][groups[te] == s], y[te][groups[te] == s])
                   for s in np.unique(groups[te])}
    return dict(tag=args.tag, channels=args.channels, crop=args.crop, fband=args.fband,
                seed=args.seed, best_epoch=best_ep,
                test_acc=evaluate(model, X[te], y[te]), n_test=len(te),
                test_acc_last=test_acc_last,
                test_subjects=sorted(int(s) for s in np.unique(groups[te])),
                test_acc_per_subject=per_subject)


def log(result):
    print(json.dumps(result))
    with open("results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", choices=list(CHANNEL_GROUPS), default="all")
    ap.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    # Task 3 probes: --crop 1 4 drops the cue/saccade second; --fband 8 30 removes slow EOG.
    ap.add_argument("--crop", type=float, nargs=2, default=list(CROP), metavar=("T0", "T1"))
    ap.add_argument("--fband", type=float, nargs=2, default=[L_FREQ, H_FREQ], metavar=("LO", "HI"))
    ap.add_argument("--split", choices=["fixed", "cv"], default="fixed",
                    help="fixed: --test/--val-subjects; cv: 5-fold GroupKFold over subjects")
    ap.add_argument("--test-subjects", type=int, nargs="+", default=[9, 10])
    ap.add_argument("--val-subjects", type=int, nargs="+", default=[7, 8])
    ap.add_argument("--tag", default="", help="label stored in results.jsonl")
    args = ap.parse_args()
    set_seed(args.seed)

    X, y, groups, ch_names = build_dataset(args.subjects, tuple(args.crop), tuple(args.fband))
    picks = CHANNEL_GROUPS[args.channels]
    if picks is not None:
        if not picks:
            raise ValueError(f"Fill in CHANNEL_GROUPS['{args.channels}'] first.")
        X = X[:, [ch_names.index(c) for c in picks]]
    print(f"data: {X.shape}, subjects={len(np.unique(groups))}, "
          f"class balance={np.bincount(y)}")

    all_subj = set(np.unique(groups).tolist())
    if args.split == "fixed":
        te_subj, va_subj = set(args.test_subjects), set(args.val_subjects)
        log(run_split(X, y, groups, all_subj - te_subj - va_subj, va_subj, te_subj, args))
        return

    # Subject-level CV: each fold tests 2 subjects; the next fold's test subjects
    # are its validation set, so every subject is test once and validation once.
    folds = [set(np.unique(groups[te]).tolist())
             for _, te in GroupKFold(n_splits=5).split(X, y, groups)]
    rows = []
    for k, te_subj in enumerate(folds):
        va_subj = folds[(k + 1) % len(folds)]
        row = run_split(X, y, groups, all_subj - te_subj - va_subj, va_subj, te_subj, args)
        row["fold"] = k
        log(row); rows.append(row)
    accs = np.array([r["test_acc"] for r in rows])
    n = np.array([r["n_test"] for r in rows])
    log(dict(tag=args.tag, channels=args.channels, crop=args.crop, fband=args.fband,
             seed=args.seed, split="cv",
             fold_test_acc=accs.tolist(), test_acc_mean=accs.mean(), test_acc_sd=accs.std(ddof=1),
             test_acc_pooled=float((accs * n).sum() / n.sum()), n_test=int(n.sum()),
             test_acc_last_mean=float(np.mean([r["test_acc_last"] for r in rows]))))


if __name__ == "__main__":
    main()
