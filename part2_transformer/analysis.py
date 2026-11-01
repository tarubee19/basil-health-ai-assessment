"""
Analysis of the cv.py rows in results.jsonl (rows with an "exp" tag) and the weights cv.py saved.

  python -u analysis.py attn --tag B1              # B2: attention vs occlusion maps (geom) + occlusion (eegnet)
  python -u analysis.py stats                      # B4: tables and tests -> stdout (markdown)
  python -u analysis.py figure --tag B1            # B4: figure_cv.png

Attention is not attribution: attention says where the CLS token reads from; occlusion (zeroing a channel = setting it
to the training mean after z-scoring) says which channels the decision depends on. Both are reported, and compared.
"""
import os
import torch

torch.set_num_threads(int(os.environ.get("NTHREADS", 4)))

import argparse
import glob
import itertools
import json
import re

import numpy as np
from scipy import stats

from data import get_channel_positions
from main import channel_names
import cv

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "results.jsonl")
ORDER = ["geom", "shuffled", "learned", "none", "tokens_noattn", "distbias", "distbias_shuf",
         "eegnet", "eegnet_p1", "patch", "vanilla", "csp", "ts", "ts_rpa"]
ATTN_SIZE = {"vanilla": 482 ** 2, "patch": (481 // 32 + 1) ** 2, "geom": 65 ** 2, "shuffled": 65 ** 2,
             "learned": 65 ** 2, "none": 65 ** 2, "distbias": 65 ** 2, "distbias_shuf": 65 ** 2}


def rows(tag=None):
    with open(LOG) as f:
        r = [json.loads(l) for l in f if l.strip()]
    r = [x for x in r if "exp" in x]                       # cv.py rows only; exam-day main.py rows have no tag
    return [x for x in r if tag is None or x["exp"] == tag]


def is_central(name):
    n = name.lower()
    return bool(re.match(r"^(fc|cp|c)(\d|z)", n))


def prepare(tag):
    """Rebuild the z-scored data exactly as cv.py did for this tag."""
    r0 = rows(tag)[0]
    X, y, groups = cv.load_all(r0["tmin"], r0["tmax"])
    pos, names = get_channel_positions(), channel_names()
    keep = np.arange(X.shape[1])
    if r0["drop"] == "rim":
        low = [n.lower() for n in names]
        rim_idx = [low.index(r.lower()) for r in cv.RIM]
        keep = np.array([i for i in keep if i not in rim_idx])
    X, pos, names = X[:, keep], pos[keep], [names[i] for i in keep]
    if r0["align"] == "ea":
        X = cv.euclidean_align(X, groups)
    return X, y, groups, pos, names, cv.make_folds(groups)


REGIONS = {
    "frontal-polar": "Fp1 Fpz Fp2 AF7 AF3 AFz AF4 AF8",
    "frontal": "F7 F5 F3 F1 Fz F2 F4 F6 F8",
    "L temporal": "FT7 T7 T9 TP7",
    "R temporal": "FT8 T8 T10 TP8",
    "L central": "FC5 FC3 FC1 C5 C3 C1 CP5 CP3 CP1",
    "R central": "FC2 FC4 FC6 C2 C4 C6 CP2 CP4 CP6",
    "midline central": "FCz Cz CPz",
    "parietal": "P7 P5 P3 P1 Pz P2 P4 P6 P8",
    "occipital": "PO7 PO3 POz PO4 PO8 O1 Oz O2 Iz",
}


def region_index(names):
    """{region: [channel indices]} for the channels present; asserts the 9 regions partition the full montage."""
    low = [n.lower() for n in names]
    full = [c.lower() for v in REGIONS.values() for c in v.split()]
    if len(names) == 64:
        assert sorted(full) == sorted(low), set(full) ^ set(low)
    reg = {r: [low.index(c.lower()) for c in v.split() if c.lower() in low] for r, v in REGIONS.items()}
    return {r: i for r, i in reg.items() if i}


@torch.no_grad()
def maps_for(model, Z, y, want_attn, regions, max_trials=100):
    model.eval()
    if len(y) > max_trials:
        keep = np.random.default_rng(0).choice(len(y), max_trials, replace=False)
        Z, y = Z[keep], y[keep]
    xt = torch.from_numpy(Z)
    if want_attn:
        logits, a = model(xt, return_attn=True)
        w = a[:, :, 0, 1:].mean(1).numpy()                         # (N, C) CLS->electrode, mean over heads
        attn = np.stack([w[y == c].mean(0) for c in (0, 1)])
    else:
        logits, attn = model(xt), None
    p0 = logits.softmax(1).numpy()[np.arange(len(y)), y]
    occ = np.zeros((2, len(regions)))
    for j, idx in enumerate(regions.values()):
        Zo = Z.copy(); Zo[:, idx] = 0.0                             # 0 = training mean after z-scoring
        p = model(torch.from_numpy(Zo)).softmax(1).numpy()[np.arange(len(y)), y]
        occ[:, j] = [(p0 - p)[y == c].mean() for c in (0, 1)]
    return attn, occ


def cmd_attn(args):
    X, y, groups, pos, names, folds = prepare(args.tag)
    C, T = X.shape[1], X.shape[2]
    regions = region_index(names)
    out = {"regions": list(regions), "region_channels": {r: [names[i] for i in v] for r, v in regions.items()}}
    for name in ("geom", "eegnet"):
        A, O, keys = [], [], []
        for k in range(5):
            te_s, va_s = folds[k], folds[(k + 1) % 5]
            tr = np.isin(groups, [s for s in cv.SUBJECTS if s not in te_s + va_s]); te = np.isin(groups, te_s)
            mu, sd = X[tr].mean(axis=(0, 2), keepdims=True), X[tr].std(axis=(0, 2), keepdims=True) + 1e-6
            Z = ((X[te] - mu) / sd).astype(np.float32)
            for seed in args.seeds:
                f = os.path.join(HERE, "scratch", "models", f"{args.tag}_{name}_s{seed}_f{k}.pt")
                if not os.path.exists(f):
                    continue
                model, _ = cv.build(name, C, T, pos, seed)
                model.load_state_dict(torch.load(f))
                a, o = maps_for(model, Z, y[te], want_attn=name != "eegnet", regions=regions)
                A.append(a); O.append(o); keys.append((k, seed))
        if not keys:
            continue
        O = np.array(O)
        res = {"n_models": len(keys), "runs": keys, "occ_region_mean": O.mean(0).tolist()}
        occ_cls = O.mean(1)                                          # (runs, regions), class-averaged
        res["occ_by_region"] = dict(zip(regions, np.round(occ_cls.mean(0), 4).tolist()))
        if name != "eegnet":
            A = np.array(A); att_cls = A.mean(1)
            res["attn_mean"] = A.mean(0).tolist()
            att_reg = np.stack([att_cls[:, idx].sum(1) for idx in regions.values()], 1)   # (runs, regions)
            share = np.array([len(i) / C for i in regions.values()])
            res["attn_by_region_vs_uniform"] = dict(zip(regions, np.round(att_reg.mean(0) / share, 3).tolist()))
            res["spearman_attn_occ_regions_avg"] = float(stats.spearmanr(att_reg.mean(0), occ_cls.mean(0))[0])
            res["spearman_attn_occ_regions_per_run_mean"] = float(np.nanmean(
                [stats.spearmanr(a, o)[0] for a, o in zip(att_reg, occ_cls)]))
            pair = [stats.spearmanr(att_cls[i], att_cls[j])[0]
                    for i, j in itertools.combinations(range(len(keys)), 2) if keys[i][0] == keys[j][0]]
            res["attn_seed_stability_mean_pairwise_spearman"] = float(np.mean(pair))
            cen = np.array([is_central(n) for n in names])
            res["attn_central_enrichment"] = float(att_cls.mean(0)[cen].sum() / (cen.sum() / C))
            low = [r.lower() for r in cv.RIM]
            rim = np.array([n.lower() in low for n in names])
            if rim.any():
                res["attn_rim_enrichment"] = float(att_cls.mean(0)[rim].sum() / (rim.sum() / C))
            top = np.argsort(att_cls.mean(0))[::-1][:10]
            res["attn_top10"] = [names[i] for i in top]
        pair_o = [stats.spearmanr(occ_cls[i], occ_cls[j])[0]
                  for i, j in itertools.combinations(range(len(keys)), 2) if keys[i][0] == keys[j][0]]
        res["occ_seed_stability_mean_pairwise_spearman"] = float(np.mean(pair_o))
        out[name] = res
        print(name, {k: v for k, v in res.items() if k not in ("attn_mean", "occ_region_mean", "runs")},
              flush=True)
    out["names"] = names
    out["pos"] = pos.tolist()
    with open(os.path.join(HERE, f"analysis_{args.tag}.json"), "w") as f:
        json.dump(out, f)
    print(f"saved analysis_{args.tag}.json")


def per_subject(rs):
    """{subject: mean accuracy over seeds}"""
    d = {}
    for r in rs:
        for s, a in r["per_subject"].items():
            d.setdefault(int(s), []).append(a)
    return {s: float(np.mean(v)) for s, v in sorted(d.items())}


def boot_ci(v, n=10000):
    rng = np.random.default_rng(0)
    v = np.asarray(v)
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    nz = d[d != 0]
    w = stats.wilcoxon(d).pvalue if len(nz) else 1.0
    sign = stats.binomtest(int((nz > 0).sum()), len(nz)).pvalue if len(nz) else 1.0
    return d.mean(), boot_ci(d), w, sign


def label(r):
    """Row label: model, channel subset if not all, [T] if transductive."""
    return r["model"] + (f"_{r['channels']}" if r.get("channels") else "") + (" [T]" if r["transductive"] else "")


def sort_labels(labels):
    base = lambda l: l.split(" ")[0].replace("_motor", "")
    return sorted(labels, key=lambda l: (ORDER.index(base(l)) if base(l) in ORDER else 99, l))


def summarise_label(rm):
    seeds = sorted({r["seed"] for r in rm})
    complete = [s for s in seeds if len({r["fold"] for r in rm if r["seed"] == s}) == 5]
    return seeds, complete


def cmd_stats(args):
    R = rows()
    thr = next(k for k in range(451) if stats.binom.sf(k - 1, 450, 0.5) < 0.05) / 450
    print(f"binomial p<.05 threshold on 450 pooled trials: {thr:.3f}\n")
    g_rows = [r for r in R if r["exp"] == "B1" and label(r) == "geom"]
    g_subj = per_subject([r for r in g_rows if r["seed"] in summarise_label(g_rows)[1]]) if g_rows else None
    for tag in dict.fromkeys(r["exp"] for r in R):
        rt = [r for r in R if r["exp"] == tag]
        r0 = rt[0]
        print(f"### {tag}  (t={r0['tmin']:g}-{r0['tmax']:g} s, drop={r0['drop']}, align={r0['align']})\n")
        print("| model | n_chan | runs (fold×seed) | seeds | pooled acc per seed | mean | best run | worst run | "
              "std across seeds (mean within fold) | std across folds | final-epoch mean | peak (test-selected) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        subj = {}
        for m in sort_labels(dict.fromkeys(label(r) for r in rt)):
            rm = [r for r in rt if label(r) == m]
            seeds, complete = summarise_label(rm)
            pooled = [np.mean([r["acc"] for r in rm if r["seed"] == s]) for s in complete]
            acc = np.array([r["acc"] for r in rm])
            within = np.mean([np.std([r["acc"] for r in rm if r["fold"] == k])
                              for k in range(5) if sum(r["fold"] == k for r in rm) > 1] or [np.nan])
            across = np.std([np.mean([r["acc"] for r in rm if r["fold"] == k]) for k in range(5)
                             if any(r["fold"] == k for r in rm)])
            fin = [r["acc_final"] for r in rm if r["acc_final"] is not None]
            pk = [r["acc_peak_test_selected"] for r in rm if r["acc_peak_test_selected"] is not None]
            nan = float("nan")
            print(f"| {m} | {rm[0]['n_chan']} | {len(rm)} | {','.join(map(str, seeds))} | "
                  f"{' / '.join(f'{p:.3f}' for p in pooled) or '-'} | {acc.mean():.3f} | {acc.max():.3f} | "
                  f"{acc.min():.3f} | {within:.3f} | {across:.3f} | "
                  f"{np.mean(fin) if fin else nan:.3f} | {np.mean(pk) if pk else nan:.3f} |")
            if complete:
                subj[m] = per_subject([r for r in rm if r["seed"] in complete])
        print("\nPer-subject accuracy (mean over complete seeds), bootstrap 95% CI over subjects, Wilcoxon vs 0.5:\n")
        print("| model | " + " | ".join(f"S{s}" for s in cv.SUBJECTS) + " | mean | 95% CI | Wilcoxon p vs .5 |")
        print("|---|" + "---|" * (len(cv.SUBJECTS) + 3))
        for m, d in subj.items():
            v = np.array([d[s] for s in cv.SUBJECTS])
            lo, hi = boot_ci(v)
            p = stats.wilcoxon(v - 0.5).pvalue if np.any(v != 0.5) else 1.0
            print(f"| {m} | " + " | ".join(f"{x:.2f}" for x in v) + f" | {v.mean():.3f} | {lo:.3f}–{hi:.3f} | {p:.3f} |")
        if g_subj:
            print("\nPaired over 10 subjects: B1 geom − X (mean diff, bootstrap 95% CI, Wilcoxon p, sign-test p):\n")
            print("| X | mean diff | 95% CI | Wilcoxon p | sign p |")
            print("|---|---|---|---|---|")
            g = [g_subj[s] for s in cv.SUBJECTS]
            for m in subj:
                if tag == "B1" and m == "geom":
                    continue
                d, (lo, hi), w, sg = paired(g, [subj[m][s] for s in cv.SUBJECTS])
                print(f"| {m} | {d:+.3f} | {lo:+.3f}–{hi:+.3f} | {w:.3f} | {sg:.3f} |")
        print()
    rb = [r for r in R if r["exp"] in ("B1", "B6")]
    if rb:
        print("### Efficiency (4 threads, train subjects = 6, 270 trials)\n")
        print("**Timings are not comparable between models:** Part 1's follow-up jobs ran on the same CPU at the same")
        print("time, with varying load, so s/epoch and ms/trial mix model cost with contention. Params and attention")
        print("sizes are exact.\n")
        print("| model | params | train s/epoch | inference ms/trial | attention matrix entries per head |")
        print("|---|---|---|---|---|")
        for m in sort_labels(dict.fromkeys(label(r) for r in rb)):
            rm = [r for r in rb if label(r) == m]
            p = rm[0]["params"]
            sec = f"{np.mean([r['sec_per_epoch'] for r in rm]):.2f}" + (" (total fit)" if p is None else "")
            ms = np.mean([r["infer_ms_per_trial"] for r in rm])
            att = f"{ATTN_SIZE[m]:,}" if m in ATTN_SIZE else "—"
            print(f"| {m} | {f'{p:,}' if p is not None else '—'} | {sec} | {ms:.3f} | {att} |")


def cmd_figure(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne
    rt = [r for r in rows() if r["exp"] in (args.tag, "B6", "B7")]
    name = lambda r: "csp 6-subj" if r["exp"] == args.tag and r["model"] == "csp" else label(r)
    models = [n for e in (args.tag, "B6", "B7") for n in dict.fromkeys(name(r) for r in rt if r["exp"] == e)]
    thr = next(k for k in range(451) if stats.binom.sf(k - 1, 450, 0.5) < 0.05) / 450
    ink, muted, accent = "#1f2328", "#8b949e", "#c2410c"
    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1], hspace=0.45)
    ax = fig.add_subplot(gs[0, :])
    for i, m in enumerate(models):
        rm = [r for r in rt if name(r) == m]
        d = per_subject(rm)
        v = np.array(list(d.values()))
        col = accent if m == "geom" else ink
        ax.scatter(np.full(len(v), i) + np.linspace(-0.2, 0.2, len(v)), v, s=22, color=col, alpha=0.8, zorder=3)
        pooled = np.mean([r["acc"] for r in rm])
        ax.hlines(pooled, i - 0.3, i + 0.3, color=col, lw=2.2, zorder=4)
    ax.axhline(0.5, color=muted, lw=1)
    ax.axhline(thr, color=muted, lw=1, ls="--")
    ax.text(len(models) - 0.45, 0.5, "chance", va="center", fontsize=8, color=muted)
    ax.text(len(models) - 0.45, thr, f"p<.05, n=450\n({thr:.3f})", va="center", fontsize=8, color=muted)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace("_", "\n").replace(" ", "\n") for m in models], fontsize=8)
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_ylabel("test accuracy per subject")
    ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("(a) per-subject accuracy (dots, mean over seeds 0-2), pooled mean (bar); 5-fold subject CV. "
                 "Neural models: validation-selected checkpoint. [T] = transductive. Vanilla not run.",
                 loc="left", fontsize=10)
    with open(os.path.join(HERE, f"analysis_{args.tag}.json")) as f:
        an = json.load(f)
    pos = np.array(an["pos"])
    A, Oreg = np.array(an["geom"]["attn_mean"]), np.array(an["geom"]["occ_region_mean"])   # (2, C), (2, R)
    O = np.zeros_like(A)                                   # paint each region's occlusion value on its channels
    for j, r in enumerate(an["regions"]):
        O[:, [an["names"].index(n) for n in an["region_channels"][r]]] = Oreg[:, [j]]
    panels = [(A[0], "attention, fists"), (A[1], "attention, feet"),
              (O[0], "region occlusion Δp(correct), fists"), (O[1], "region occlusion Δp(correct), feet")]
    for j, (w, lab) in enumerate(panels):
        axj = fig.add_subplot(gs[1, j])
        vl = (0, 2 / A.shape[1]) if j < 2 else (-np.abs(O).max(), np.abs(O).max())   # attention: 0 to 2x uniform
        im, _ = mne.viz.plot_topomap(w, pos[:, :2], axes=axj, show=False, contours=0, sensors=True,
                                     cmap="Oranges" if j < 2 else "RdBu_r", vlim=vl)
        axj.set_title(f"({'bcde'[j]}) geom {lab}", fontsize=9, loc="left")
        if j in (1, 3):
            fig.colorbar(im, ax=axj, shrink=0.75)
    fig.text(0.5, 0.02, "(a) csp, ts and ts rpa rows fit on all 8 non-test subjects; csp 6-subj on 6. "
             "(b, c) last-layer CLS attention, mean over heads; scale 0 to 2x uniform (uniform = 1/64). "
             "(d, e) occlusion of each of 9 regions, painted on its channels; positive = correct-class probability "
             "drops.\nYourModel geom, averaged over test trials, 5 folds x 3 seeds. Attention is not attribution.",
             ha="center", fontsize=8, color=muted)
    path = os.path.join(HERE, "figure_cv.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"saved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["attn", "stats", "figure"])
    ap.add_argument("--tag", default="B1")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="attn: which seeds' models to analyse")
    a = ap.parse_args()
    {"attn": cmd_attn, "stats": cmd_stats, "figure": cmd_figure}[a.cmd](a)
