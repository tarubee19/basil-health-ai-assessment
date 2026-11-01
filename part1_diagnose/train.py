"""Train and evaluate a hands-vs-feet decoder on unseen subjects.

Usage:
  python train.py --channels all --seed 0
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn

from data import build_dataset, SUBJECTS
from model import EEGNet, count_params

# Electrode groups for the channel-subset run (Task 3).
# TODO: fill in "frontal" with the electrode names you choose.
CHANNEL_GROUPS = {
    "all": None,
    "frontal": [],
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", choices=list(CHANNEL_GROUPS), default="all")
    ap.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-subjects", type=int, nargs="+", default=[9, 10])
    ap.add_argument("--val-subjects", type=int, nargs="+", default=[7, 8])
    ap.add_argument("--tag", default="", help="label stored in results.jsonl")
    args = ap.parse_args()
    set_seed(args.seed)

    X, y, groups, ch_names = build_dataset(args.subjects)
    picks = CHANNEL_GROUPS[args.channels]
    if picks is not None:
        if not picks:
            raise ValueError(f"Fill in CHANNEL_GROUPS['{args.channels}'] first.")
        X = X[:, [ch_names.index(c) for c in picks]]
    print(f"data: {X.shape}, subjects={len(np.unique(groups))}, "
          f"class balance={np.bincount(y)}")

    # Per-channel z-scoring.
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True) + 1e-6
    X = ((X - mean) / std).astype(np.float32)

    # Subject-independent evaluation: whole subjects are held out for test and
    # validation, so no person appears in more than one set.
    te_subj, va_subj = set(args.test_subjects), set(args.val_subjects)
    tr_subj = set(np.unique(groups).tolist()) - te_subj - va_subj
    assert not (tr_subj & te_subj) and not (tr_subj & va_subj) and not (va_subj & te_subj)
    assert tr_subj and va_subj and te_subj
    tr = np.flatnonzero(np.isin(groups, list(tr_subj)))
    va = np.flatnonzero(np.isin(groups, list(va_subj)))
    te = np.flatnonzero(np.isin(groups, list(te_subj)))
    print(f"train subjects {sorted(tr_subj)}  val subjects {sorted(va_subj)}  "
          f"test subjects {sorted(te_subj)}")

    n_ch, n_t = X.shape[1], X.shape[2]
    model = EEGNet(n_ch, n_t)
    print(f"EEGNet: {count_params(model)} params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(args.seed)

    best_acc, best_state, best_ep = -1.0, None, -1
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), args.bs):
            b = order[i:i + args.bs]
            xb, yb = torch.from_numpy(X[b]), torch.from_numpy(y[b])
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        va_acc = evaluate(model, X[va], y[va])
        te_acc = evaluate(model, X[te], y[te])
        if te_acc > best_acc:   # keep the best checkpoint
            best_acc, best_ep = te_acc, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep:3d}  loss {loss.item():.3f}  val {va_acc:.3f}  test {te_acc:.3f}")

    test_acc_last = te_acc
    model.load_state_dict(best_state)
    per_subject = {int(s): evaluate(model, X[te][groups[te] == s], y[te][groups[te] == s])
                   for s in np.unique(groups[te])}
    result = dict(tag=args.tag, channels=args.channels, seed=args.seed, best_epoch=best_ep,
                  test_acc=evaluate(model, X[te], y[te]), n_test=len(te),
                  test_acc_last=test_acc_last,
                  test_subjects=sorted(int(s) for s in np.unique(groups[te])),
                  test_acc_per_subject=per_subject)
    print(json.dumps(result))
    with open("results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
