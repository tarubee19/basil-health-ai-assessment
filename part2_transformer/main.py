"""
EEG Transformer Sprint — entry point.
Run:  python -u main.py --epochs 20 --seeds 0 1 2   # what the reported numbers come from
      (default --epochs is the starter's 40; 20 was used under the exam time limit)

This script:
  1. loads the cross-subject split (train subjects 1-8, test 9-10),
  2. trains every model on the SAME seeds with the SAME harness/hyperparameters:
     EEGNet (supplied), EEGNet-P1 (kern=80, p=0.5, the Part-1 config), the vanilla
     transformer, the starter patch transformer, and YourModel,
  3. appends one line per run to results.jsonl and prints a summary table,
  4. saves figure.png (accuracy per seed + CLS->electrode attention topomaps)
     and figure_tsne.png (the starter t-SNE of YourModel's CLS embedding).

Headline metric = FINAL-epoch test accuracy. train_model's "best" is the max of
the test curve, i.e. chosen on the test set; it is reported alongside as "peak".
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from data import load_data, get_channel_positions, RUNS
from models import EEGNet, VanillaEEGTransformer, count_params
from train import train_model, set_seed

HERE = os.path.dirname(os.path.abspath(__file__))


# =========================================================================== #
#  Starter example, kept UNCHANGED as a comparison row (was the YourModel stub):
#  learned TEMPORAL PATCH tokens. This is the PDF's own worked example, so it is
#  not our bet — it shows what the "tokenisation" bias alone buys.
# =========================================================================== #
class PatchTransformer(nn.Module):
    def __init__(self, n_chan, n_time, n_classes=2,
                 patch=32, d_model=64, n_heads=4, depth=3, p=0.3):
        super().__init__()
        # Learned temporal patches: a Conv1d with stride == kernel tokenises
        # the signal into n_time//patch tokens, each summarising `patch` samples
        # across all electrodes. This is the ONE bias in the starter example.
        self.tokenizer = nn.Conv1d(n_chan, d_model, kernel_size=patch,
                                   stride=patch)
        n_tokens = n_time // patch
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2,
            dropout=p, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, return_embed=False):   # x: (B, C, T)
        tok = self.tokenizer(x).transpose(1, 2)  # (B, n_tokens, d)
        cls = self.cls.expand(x.size(0), -1, -1)
        z = torch.cat([cls, tok], dim=1) + self.pos
        z = self.encoder(z)
        embed = self.norm(z[:, 0])
        logits = self.head(embed)
        return (logits, embed) if return_embed else logits


# =========================================================================== #
#  YOUR MODEL — electrode tokens + 3-D scalp-geometry positional encoding.
#
#  The one bias: attend across SPACE (64 electrode tokens), not TIME (481 samples).
#  Fists-vs-feet lives in where mu/beta power drops (lateral hand areas vs the
#  medial foot area near Cz: an anterior-posterior/medial contrast), and motor
#  imagery is induced (random phase), so each token carries phase-invariant log
#  band-power and is told where it sits on the scalp via its xyz position.
# =========================================================================== #
class Block(nn.Module):
    """Pre-LN transformer block that also returns its attention weights."""

    def __init__(self, d, h, p, attn_p=0.1):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.adrop, self.drop = nn.Dropout(attn_p), nn.Dropout(p)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(p),
                                nn.Linear(2 * d, d))

    def forward(self, z):
        B, N, D = z.shape
        q, k, v = (self.qkv(self.ln1(z)).view(B, N, 3, self.h, D // self.h)
                   .permute(2, 0, 3, 1, 4))
        # q.k is a sum of d_head terms -> variance ~d_head; /sqrt(d_head) keeps the
        # logits ~unit variance so the softmax does not saturate (vanishing grads).
        a = (q @ k.transpose(-2, -1) / math.sqrt(D // self.h)).softmax(-1)  # (B, h, N, N)
        z = z + self.drop(self.proj((self.adrop(a) @ v).transpose(1, 2).reshape(B, N, D)))
        return z + self.drop(self.ff(self.ln2(z))), a


class YourModel(nn.Module):
    """Electrode tokens + 3-D scalp-geometry position.
    Attention runs over 64 electrodes + CLS (65x65), not 481 timesteps + CLS (482x482)."""

    def __init__(self, n_chan, n_time, n_classes=2, pos=None, pe="geom",
                 n_filt=16, kern=64, win=40, d_model=64, n_heads=4, depth=2, p=0.3):
        super().__init__()
        # Shared per-electrode temporal filter bank (0.4 s kernel ~ 2.5 Hz resolution:
        # separates mu from beta). Shared across electrodes -> tokens are comparable.
        self.temporal = nn.Conv1d(1, n_filt, kern, stride=2, padding=kern // 2, bias=False)
        self.pool = nn.AvgPool1d(win, win)            # 40 samples at stride 2 = 0.5 s windows
        with torch.no_grad():
            f = self.pool(self.temporal(torch.zeros(1, 1, n_time))).numel()
        self.embed = nn.Sequential(nn.LayerNorm(f), nn.Linear(f, d_model))
        self.pe = pe
        if pe in ("geom", "shuffled"):
            P = torch.as_tensor(pos, dtype=torch.float32)
            P = (P - P.mean(0)) / P.std(0)
            if pe == "shuffled":                       # control: same values, wrong electrodes
                P = P[torch.randperm(n_chan, generator=torch.Generator().manual_seed(0))]
            self.register_buffer("xyz", P)
            self.pos_mlp = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, d_model))
        elif pe == "learned":                          # control: identity without geometry
            self.chan_emb = nn.Parameter(torch.zeros(1, n_chan, d_model))
            nn.init.trunc_normal_(self.chan_emb, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, p) for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(d_model), nn.Linear(d_model, n_classes)

    def forward(self, x, return_embed=False, return_attn=False):   # x: (B, C, T)
        B, C, T = x.shape
        h = self.temporal(x.reshape(B * C, 1, T))       # (B*C, F, T/2)
        h = torch.log(self.pool(h ** 2) + 1e-6)         # log band-power per window: phase-invariant
        tok = self.embed(h.flatten(1)).view(B, C, -1)   # (B, C, d)
        if self.pe in ("geom", "shuffled"):
            tok = tok + self.pos_mlp(self.xyz)
        elif self.pe == "learned":
            tok = tok + self.chan_emb
        z = torch.cat([self.cls.expand(B, -1, -1), tok], 1)   # (B, C+1, d)
        for blk in self.blocks:
            z, a = blk(z)
        emb = self.norm(z[:, 0])
        logits = self.head(emb)
        if return_attn:
            return logits, a                            # last-layer attention (B, h, C+1, C+1)
        return (logits, emb) if return_embed else logits


# =========================================================================== #


def channel_names():
    """Channel names in the same order as X / get_channel_positions()."""
    import mne
    from mne.datasets import eegbci
    from mne.io import read_raw_edf
    raw = read_raw_edf(eegbci.load_data(1, RUNS[:1], verbose="ERROR")[0],
                       preload=False, verbose="ERROR")
    raw.rename_channels(lambda c: c.strip("."))
    return [raw.ch_names[p] for p in mne.pick_types(raw.info, eeg=True, exclude="bads")]


def run_across_seeds(name, pe, build_fn, data, seeds, device, epochs, log_path):
    """Train build_fn() once per seed; log every run. Returns (rows, seed-0 model)."""
    Xtr, ytr, Xte, yte = data
    rows, keep = [], None
    for s in seeds:
        set_seed(s)
        model = build_fn()
        t0 = time.time()
        peak, final = train_model(model, Xtr, ytr, Xte, yte, epochs=epochs, device=device)
        row = dict(model=name, pe=pe, seed=s, final_acc=round(final, 4),
                   peak_acc=round(peak, 4), params=count_params(model), epochs=epochs)
        rows.append(row)
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  [{name}] seed {s}: final {final:.3f}  peak {peak:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        if s == seeds[0]:
            keep = model
    return rows, keep


def summarise(rows):
    print(f"\n{'model':<22}{'n':>3}{'best':>8}{'worst':>8}{'mean':>8}{'std':>7}"
          f"{'peak mean':>11}   (final-epoch test acc; std ddof=0)")
    for name in dict.fromkeys(r["model"] for r in rows):
        f = np.array([r["final_acc"] for r in rows if r["model"] == name])
        pk = np.array([r["peak_acc"] for r in rows if r["model"] == name])
        print(f"{name:<22}{len(f):>3}{f.max():>8.3f}{f.min():>8.3f}{f.mean():>8.3f}"
              f"{f.std():>7.3f}{pk.mean():>11.3f}")


def chance_threshold(n, alpha=0.05):
    """Smallest accuracy k/n with one-sided binomial P(X >= k | p=0.5) < alpha."""
    from scipy.stats import binom
    k = next(k for k in range(n + 1) if binom.sf(k - 1, n, 0.5) < alpha)
    return k / n


@torch.no_grad()
def cls_attention(model, Xte, yte, device):
    """Last-layer CLS->electrode attention, averaged over heads and trials, per class."""
    model.eval()
    _, a = model(torch.from_numpy(Xte).to(device), return_attn=True)
    w = a[:, :, 0, 1:].mean(1).cpu().numpy()            # (N, C)
    return {c: w[yte == c].mean(0) for c in (0, 1)}


def save_figure(rows, attn, pos, names, thr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne

    ink, muted, accent = "#1f2328", "#8b949e", "#c2410c"
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6),
                           gridspec_kw=dict(width_ratios=[2, 1, 1]))
    models = list(dict.fromkeys(r["model"] for r in rows))
    for i, m in enumerate(models):
        f = np.array([r["final_acc"] for r in rows if r["model"] == m])
        col = accent if m.startswith("YourModel (geom)") else ink
        ax[0].scatter(np.full(len(f), i) + np.linspace(-0.12, 0.12, len(f)), f,
                      s=36, color=col, zorder=3)
        ax[0].hlines(f.mean(), i - 0.25, i + 0.25, color=col, lw=2, zorder=3)
    ax[0].axhline(0.5, color=muted, lw=1, ls="-")
    ax[0].axhline(thr, color=muted, lw=1, ls="--")
    ax[0].text(-0.45, 0.5, "chance", va="bottom", ha="left", color=muted, fontsize=8)
    ax[0].text(-0.45, thr, f"p<.05 ({thr:.0%}, n=90)", va="bottom", ha="left",
               color=muted, fontsize=8)
    ax[0].set_xlim(-0.5, len(models) - 0.5)
    ax[0].set_xticks(range(len(models)))
    ax[0].set_xticklabels([m.replace(" (", "\n(").replace("Transformer", "\nTransformer")
                           for m in models], fontsize=7.5)
    ax[0].set_ylabel("final-epoch test accuracy (subj 9-10)")
    ax[0].set_ylim(0.3, 0.8)
    ax[0].grid(axis="y", color="#e5e7eb", lw=0.6)
    ax[0].spines[["top", "right"]].set_visible(False)
    ax[0].set_title("(a) accuracy per seed (dots), mean (bar)", fontsize=10, loc="left")

    vmin = min(attn[0].min(), attn[1].min())
    vmax = max(attn[0].max(), attn[1].max())
    for j, (c, lab) in enumerate([(0, "fists"), (1, "feet")]):
        im, _ = mne.viz.plot_topomap(attn[c], pos[:, :2], axes=ax[1 + j], show=False,
                                     cmap="Oranges", vlim=(vmin, vmax), contours=0,
                                     names=names, sensors=False)
        for t in ax[1 + j].texts:
            t.set_fontsize(5)
        ax[1 + j].set_title(f"({'bc'[j]}) CLS→electrode attention: {lab}", fontsize=10, loc="left")
    fig.text(0.70, 0.02, "YourModel (geom), seed 0, last layer, mean over heads and test trials",
             ha="center", fontsize=8, color=muted)
    cb = fig.colorbar(im, ax=ax[1:], shrink=0.7, pad=0.02)
    cb.set_label("attention weight (uniform = 1/64 = 0.0156)", fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def save_tsne(model, Xte, yte, device, path):
    """Starter figure: t-SNE of the CLS embeddings on the test subjects, coloured by class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    model.eval()
    with torch.no_grad():
        emb = model(torch.from_numpy(Xte).to(device), return_embed=True)[1].cpu().numpy()
    n = len(emb)
    emb2 = TSNE(n_components=2, perplexity=min(30, max(5, n // 4)),
                init="pca", random_state=0).fit_transform(emb)
    plt.figure(figsize=(5, 4))
    for c, name in [(0, "fists"), (1, "feet")]:
        m = yte == c
        plt.scatter(emb2[m, 0], emb2[m, 1], s=18, alpha=0.7, label=name)
    plt.legend(); plt.title("t-SNE, test CLS embeddings\nYourModel (geom), seed 0", fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=130)
    plt.close()
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--pe", choices=["geom", "learned", "none", "shuffled"], default="geom",
                    help="positional encoding of YourModel's electrode tokens")
    ap.add_argument("--no-ablation", action="store_true",
                    help="skip YourModel with pe=shuffled and pe=learned")
    ap.add_argument("--only", nargs="+", default=None,
                    help="run only these rows (eegnet eegnet_p1 vanilla patch yours ablation)")
    ap.add_argument("--log", default=os.path.join(HERE, "results.jsonl"))
    ap.add_argument("--figure-only", action="store_true",
                    help="redraw the figures from --log; retrains only YourModel seed 0 (not logged)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}   seeds: {args.seeds}   epochs: {args.epochs}")

    print("\nloading data (first run downloads ~50 MB)...")
    Xtr, ytr, Xte, yte = data = load_data()
    C, T = Xtr.shape[1], Xtr.shape[2]
    pos = get_channel_positions()
    assert pos.shape == (C, 3) and not np.isnan(pos).any()
    print(f"train {Xtr.shape}  test {Xte.shape}  "
          f"(class balance train {ytr.mean():.2f} / test {yte.mean():.2f})")
    thr = chance_threshold(len(yte))
    print(f"chance: 0.5; one-sided binomial p<.05 needs >= {thr:.3f} on n={len(yte)}; "
          f"SE at 0.5 = {math.sqrt(0.25 / len(yte)):.3f}")

    if args.figure_only:
        with open(args.log) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        rows = [r for r in rows if r["epochs"] == args.epochs]
        s = args.seeds[0]
        set_seed(s)
        m = YourModel(C, T, pos=pos, pe=args.pe)
        peak, final = train_model(m, Xtr, ytr, Xte, yte, epochs=args.epochs, device=device)
        print(f"  [YourModel ({args.pe})] seed {s}: final {final:.3f}  peak {peak:.3f}  "
              f"(retrained for the figure; compare with results.jsonl)")
        attn, names = cls_attention(m, Xte, yte, device), channel_names()
        save_figure(rows, attn, pos, names, thr, os.path.join(HERE, "figure.png"))
        save_tsne(m, Xte, yte, device, os.path.join(HERE, "figure_tsne.png"))
        summarise(rows)
        return

    # Ordered cheap-and-ours first, slowest (vanilla, ~482-token attention) last, so
    # results land early; every run is appended to results.jsonl as it finishes.
    rows_spec = [("yours", f"YourModel ({args.pe})", args.pe,
                  lambda: YourModel(C, T, pos=pos, pe=args.pe))]
    if not args.no_ablation:
        for pe in ("shuffled", "learned"):
            if pe != args.pe:
                rows_spec.append(("ablation", f"YourModel ({pe})", pe,
                                  lambda pe=pe: YourModel(C, T, pos=pos, pe=pe)))
    rows_spec += [
        ("eegnet", "EEGNet (supplied)", "-", lambda: EEGNet(C, T)),
        ("patch", "PatchTransformer", "-", lambda: PatchTransformer(C, T)),
        ("vanilla", "VanillaTransformer", "-", lambda: VanillaEEGTransformer(C, T)),
        ("eegnet_p1", "EEGNet-P1 (k80,p.5)", "-", lambda: EEGNet(C, T, kern=80, p=0.5)),
    ]
    if args.only:
        rows_spec = [r for r in rows_spec if r[0] in args.only]

    rows, attn, names = [], None, None
    fig_path = os.path.join(HERE, "figure.png")
    for key, name, pe, build in rows_spec:
        print(f"\n=== {name} ===  params: {count_params(build()):,}", flush=True)
        r, m = run_across_seeds(name, pe, build, data, args.seeds, device, args.epochs, args.log)
        rows += r
        if key == "yours" and args.pe != "none":
            # Figure as soon as YourModel is trained (seed-0 model, no retraining).
            attn, names = cls_attention(m, Xte, yte, device), channel_names()
            for c, lab in [(0, "fists"), (1, "feet")]:
                top = np.argsort(attn[c])[::-1][:10]
                print(f"  top-10 attended ({lab}): "
                      + ", ".join(f"{names[i]} {attn[c][i]:.3f}" for i in top))
            save_tsne(m, Xte, yte, device, os.path.join(HERE, "figure_tsne.png"))
        if attn is not None:                            # panel (a) grows as rows finish
            save_figure(rows, attn, pos, names, thr, fig_path)
    summarise(rows)

    print("\ndone. Headline = final-epoch accuracy; report best AND worst — no cherry-picking.")


if __name__ == "__main__":
    main()
