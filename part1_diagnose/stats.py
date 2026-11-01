"""Task 3 follow-up: subject-level statistics over results.jsonl (+ results_improvements.jsonl if present).

Unit of analysis = subject (n=10). Per config: per-subject accuracy (mean over seeds), mean, bootstrap 95% CI
over subjects, Wilcoxon signed-rank and sign test vs 0.5 (one-sided, greater). The pooled-trial binomial is kept
for reference but is OPTIMISTIC (treats correlated trials as independent).
Paired comparisons (EEGNet vs CSP vs TS, same channels/band/crop/split/align): two-sided sign test over subjects.

Usage: python stats.py [--plot]   -> stats_table.md (+ accuracy_grid.png)
"""
import os
import argparse
import json
from collections import defaultdict
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("NTHREADS", 4)))

from scipy.stats import binomtest, wilcoxon

SUBJ = list(range(1, 11))
EOG_FREE = ((8.0, 30.0), (1.0, 4.0))     # (band, crop)
# results.jsonl rows used here: Task 3 CV probes (t3_) and the follow-up runs; Task 1 fix rows are skipped.
FOLLOWUP_TAGS = ("t3_", "a1_", "a2_", "a3_", "c3_", "imp")
rng = np.random.default_rng(0)


def load(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def key_of(r):
    band, crop = tuple(r.get("fband", [1.0, 40.0])), tuple(r.get("crop", [0.0, 4.0]))
    split = r.get("split") if r.get("split") in ("cv", "within") else ("within" if r.get("split") == "within_fold" else "cv")
    ch = r.get("channels", "all") + ("+regressF" if r.get("regress_frontal") else "")
    align = r.get("align", "none") + (f"_k{r['align_ntrials']}" if r.get("align_ntrials") else "")   # EA from K test trials
    return (r.get("model", "eegnet"), split, align, ch, band, crop)


def collect():
    """config -> {seed: {subject: acc}} plus the pooled numbers and their n."""
    per = defaultdict(lambda: defaultdict(dict)); pooled = defaultdict(list); perm = {}
    for src in ("results.jsonl", "results_improvements.jsonl"):
        for r in load(src):
            tag = r.get("tag") or ""
            if src == "results.jsonl" and not tag.startswith(FOLLOWUP_TAGS):
                continue                       # fixed-split / bug-fix rows are not subject-level CV
            if tag in ("smoke",) or "repro" in tag:
                continue
            k = key_of(r)
            if "model" in r:                   # classical summary row: per-subject for all 10
                per[k][r["seed"]].update({int(s): v for s, v in r["test_acc_per_subject"].items()})
                pooled[k].append((r["test_acc_pooled"], r["n_test"]))
                if "perm_p" in r:
                    perm[k] = r
            elif "fold" in r and r.get("split") != "within_fold":   # EEGNet cv fold rows
                per[k][r["seed"]].update({int(s): v for s, v in r["test_acc_per_subject"].items()})
            elif r.get("split") == "within":                       # EEGNet within summary
                per[k][r["seed"]].update({int(s): v for s, v in r["test_acc_per_subject"].items()})
                pooled[k].append((r["test_acc_pooled"], r["n_test"]))
            elif r.get("split") == "cv":                           # EEGNet cv summary
                pooled[k].append((r["test_acc_pooled"], r["n_test"]))
    return per, pooled, perm


def subject_vector(seeds):
    complete = [d for d in seeds.values() if len(d) == 10]
    if not complete:
        return None, 0
    return np.array([np.mean([d[s] for d in complete]) for s in SUBJ]), len(complete)


def describe(v):
    boot = v[rng.integers(0, 10, size=(5000, 10))].mean(1)
    d = v - 0.5
    w = wilcoxon(d, alternative="greater").pvalue if np.any(d != 0) else 1.0
    k, n = int((d > 0).sum()), int((d != 0).sum())
    sp = binomtest(k, n, 0.5, alternative="greater").pvalue if n else 1.0
    return dict(mean=float(v.mean()), lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)),
                wilcoxon_p=float(w), above=k, n_nonzero=n, sign_p=float(sp))


def fmt_key(k):
    model, split, align, ch, band, crop = k
    return f"{model} | {split} | {align}{' (transductive)' if align != 'none' else ''} | {ch} | {band[0]:g}-{band[1]:g} Hz | {crop[0]:g}-{crop[1]:g} s"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--plot", action="store_true"); args = ap.parse_args()
    per, pooled, perm = collect()
    vecs, desc = {}, {}
    for k in sorted(per, key=lambda k: (k[1], k[4], k[5], k[3], k[0], k[2])):
        v, nseeds = subject_vector(per[k])
        if v is not None:
            vecs[k] = v; desc[k] = dict(describe(v), nseeds=nseeds)
    # Holm correction (on the Wilcoxon p) across the family of EOG-free cross-subject configs in this table.
    fam = sorted([k for k in desc if k[1] == "cv" and k[4] == EOG_FREE[0] and k[5] == EOG_FREE[1]],
                 key=lambda k: desc[k]["wilcoxon_p"])
    run_max = 0.0
    for i, k in enumerate(fam):
        run_max = max(run_max, min(1.0, (len(fam) - i) * desc[k]["wilcoxon_p"]))
        desc[k]["holm_p"] = run_max
    rows = []
    for k, d in desc.items():
        v = vecs[k]
        pl = pooled.get(k, [])
        pool_txt = ""
        if pl:
            acc = np.mean([a for a, _ in pl]); n = pl[0][1]
            pool_txt = f"{acc:.3f}, p={binomtest(round(acc * n), n, 0.5, alternative='greater').pvalue:.3g}"
        pm = perm.get(k)
        rows.append(f"| {fmt_key(k)} | {d['nseeds']} | {d['mean']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}] | {d['wilcoxon_p']:.3g} | "
                    f"{'' if 'holm_p' not in d else f'{d['holm_p']:.3g}'} | "
                    f"{d['above']}/10{'' if d['n_nonzero'] == 10 else f' ({10 - d['n_nonzero']} tied)'} (p={d['sign_p']:.3g}) | {pool_txt} | "
                    f"{'' if pm is None else f'p={pm['perm_p']:.4f} (N={pm['permutations']}, null 95% {pm['perm_null_95']:.3f})'} | "
                    f"{' '.join(f'{x:.2f}' for x in v)} |")
    head = ("Holm: Wilcoxon p corrected over the " + str(len(fam)) + " EOG-free (8-30 Hz, 1-4 s) cross-subject configs "
            "in this table; blank = not in that family.\n\n"
            "| model \\| split \\| align \\| channels \\| band \\| crop | seeds | subject mean [bootstrap 95% CI] | "
            "Wilcoxon p (one-sided, vs 0.5) | Holm p | subjects > 0.5 /10 (sign p) | "
            "pooled-trial binomial: optimistic (treats trials as independent) | permutation | per-subject S1..S10 |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n")
    comp = ["\n## Paired comparisons (two-sided sign test over subjects; ties dropped)\n",
            "| config | A | B | A > B | B > A | p |", "|---|---|---|---|---|---|"]
    by_cfg = defaultdict(dict)
    for k, v in vecs.items():
        by_cfg[k[1:]][k[0]] = v
    for cfg, models in sorted(by_cfg.items(), key=lambda x: str(x[0])):
        names = sorted(models)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = models[names[i]], models[names[j]]
                wa, wb = int((a > b).sum()), int((b > a).sum())
                p = binomtest(wa, wa + wb, 0.5).pvalue if wa + wb else 1.0
                comp.append(f"| {' | '.join(map(str, cfg))} | {names[i]} | {names[j]} | {wa} | {wb} | {p:.3g} |")
    text = "# Subject-level statistics (Task 3 follow-up)\n\n" + head + "\n".join(rows) + "\n" + "\n".join(comp) + "\n"
    open("stats_table.md", "w", encoding="utf-8").write(text)
    print(text)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sel = [k for k in vecs if k[2] in ("none", "ea")]
        sel = sorted(sel, key=lambda k: (k[1], k[4] != (1.0, 40.0), k[3], k[0], k[2]))
        fig, ax = plt.subplots(figsize=(max(10, 0.33 * len(sel)), 5.5))
        for i, k in enumerate(sel):
            v = vecs[k]
            ax.scatter(np.full(10, i) + rng.uniform(-0.15, 0.15, 10), v, s=10, alpha=0.5, color="C0" if k[1] == "cv" else "C2")
            ax.plot([i - 0.3, i + 0.3], [v.mean()] * 2, color="k", lw=2)
        ax.axhline(0.5, color="r", lw=1, label="chance")
        perm_thr = [p["perm_null_95"] for p in perm.values()]
        if perm_thr:
            ax.axhline(max(perm_thr), color="r", ls="--", lw=1, label="permutation 95% (pooled, CSP)")
        ax.set_xticks(range(len(sel)))
        ax.set_xticklabels([f"{k[0]} {k[3]} {k[4][0]:g}-{k[4][1]:g}Hz {k[5][0]:g}-{k[5][1]:g}s {'EA' if k[2] == 'ea' else ''}"
                            for k in sel], rotation=90, fontsize=6)
        ax.set_ylabel("accuracy per subject (dots) / mean (bar)")
        ax.set_title("Per-subject accuracy by config (blue = cross-subject CV, green = within-subject)")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("accuracy_grid.png", dpi=120)
        print("saved accuracy_grid.png")


if __name__ == "__main__":
    main()
