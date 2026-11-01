"""Task 2 follow-up: ERD analysis with subject-level CIs (erd.py is left unchanged; its functions are reused).

- per-subject ERD curves, mean ± bootstrap 95% CI over SUBJECTS, mu and beta, C3/Cz/C4
- the same with a small Laplacian (centre minus mean of 4 neighbours)
- individual mu band: peak of pre-cue (-2..0 s) PSD in 7-14 Hz, mean of C3/C4, band = peak ± 2 Hz
- topomaps of mean ERD% (1-4 s): hands, feet, hands - feet, mu and beta
- sign test: subjects with negative ERD per channel/condition
Outputs: erd_plus_mu.png, erd_plus_beta.png, erd_topomaps.png, erd_plus_summary.json
"""
import os
import json
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("NTHREADS", 4)))

import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import binomtest

import erd
from data import load_raw, get_events, EVENT_ID, SUBJECTS

LAP = {"C3": ["FC3", "C5", "C1", "CP3"], "Cz": ["FCz", "C1", "C2", "CPz"], "C4": ["FC4", "C2", "C6", "CP4"]}
WIN = (1.0, 4.0)
N_BOOT = 2000
rng = np.random.default_rng(0)


def derived_raw(raw):
    """RawArray with C3/Cz/C4 (referential) and their small-Laplacian versions."""
    names = sorted({c for k, v in LAP.items() for c in [k, *v]})
    d = dict(zip(names, raw.get_data(picks=names)))
    sig = [d[c] for c in LAP] + [d[c] - np.mean([d[n] for n in LAP[c]], axis=0) for c in LAP]
    ra = mne.io.RawArray(np.array(sig), mne.create_info(list(LAP) + [f"{c}_lap" for c in LAP],
                                                        raw.info["sfreq"], "eeg"), verbose="error")
    ra.set_meas_date(raw.info["meas_date"])      # so the annotations (boundaries, T0/T1/T2) align
    ra.set_annotations(raw.annotations)
    return ra


def mu_peak(raw, events):
    ep = mne.Epochs(raw, events, EVENT_ID, -2.0, 0.0, baseline=None, picks=["C3", "C4"],
                    preload=True, verbose="error")
    f, P = welch(ep.get_data(copy=True), fs=raw.info["sfreq"], nperseg=320, axis=-1)
    P = P.mean(axis=(0, 1)); m = (f >= 7) & (f <= 14)
    # The raw-PSD argmax hits the 7 Hz edge for half the subjects (1/f slope dominates), so remove
    # the aperiodic part first: line fit in log-log over 3-30 Hz excluding 7-14, peak of the residual.
    fit = (f >= 3) & (f <= 30) & ~m
    a, b = np.polyfit(np.log10(f[fit]), np.log10(P[fit]), 1)
    resid = np.log10(P) - (a * np.log10(np.maximum(f, 1e-3)) + b)
    return float(f[m][np.argmax(resid[m])])


def boot_ci(curves):
    """curves (subjects, time) -> mean, lo, hi (bootstrap over subjects)."""
    idx = rng.integers(0, len(curves), size=(N_BOOT, len(curves)))
    bm = curves[idx].mean(axis=1)
    return curves.mean(0), np.percentile(bm, 2.5, axis=0), np.percentile(bm, 97.5, axis=0)


def main():
    out = {"mu_peak_hz": {}, "tables": {}}
    curves = {}          # (band, cond) -> (subjects, 6 channels, time)
    topo = {}            # (band, cond) -> (subjects, 64)
    info64 = None
    for s in SUBJECTS:
        raw = load_raw(s); events = get_events(raw)
        pk = mu_peak(raw, events); out["mu_peak_hz"][s] = pk
        ra = derived_raw(raw)
        for band, rng_hz in (("mu", erd.BANDS["mu"]), ("beta", erd.BANDS["beta"]), ("mu_ind", (pk - 2, pk + 2))):
            ep = erd.band_power_epochs(ra, events, rng_hz)
            for c in EVENT_ID:
                e, t = erd.erd_percent(ep, c)
                curves.setdefault((band, c), []).append(e)
        for band in ("mu", "beta"):
            ep = erd.band_power_epochs(raw, events, erd.BANDS[band])
            w = (ep.times >= WIN[0]) & (ep.times < WIN[1])
            for c in EVENT_ID:
                e, _ = erd.erd_percent(ep, c)
                topo.setdefault((band, c), []).append(e[:, w].mean(axis=1))
        info64 = raw.info
        print(f"S{s:02d} done, individual mu peak {pk:.1f} Hz", flush=True)

    curves = {k: np.array(v) for k, v in curves.items()}
    w = (t >= WIN[0]) & (t < WIN[1])
    chans = list(LAP) + [f"{c}_lap" for c in LAP]
    for band in ("mu", "beta", "mu_ind"):
        tab = {}
        for c in EVENT_ID:
            per = curves[(band, c)][:, :, w].mean(axis=2)          # (subjects, 6)
            for i, ch in enumerate(chans):
                k = int((per[:, i] < 0).sum())
                tab[f"{c} {ch}"] = dict(mean_erd=float(per[:, i].mean()), n_negative=k,
                                        sign_test_p=float(binomtest(k, len(per), 0.5, alternative="greater").pvalue),
                                        per_subject=np.round(per[:, i], 2).tolist())
        out["tables"][band] = tab
        print(f"\n{band}: mean ERD% {WIN[0]}-{WIN[1]} s (subjects negative /10, one-sided sign-test p)")
        for key, v in tab.items():
            print(f"  {key:12s} {v['mean_erd']:+6.1f}%  {v['n_negative']}/10  p={v['sign_test_p']:.3f}")

    for band in ("mu", "beta"):
        fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
        for row, suffix in enumerate(["", "_lap"]):
            for j, ch in enumerate(LAP):
                ax = axes[row, j]; i = chans.index(ch + suffix)
                for c in EVENT_ID:
                    m, lo, hi = boot_ci(curves[(band, c)][:, i])
                    ax.plot(t, m, label=c); ax.fill_between(t, lo, hi, alpha=0.2)
                if band == "mu":
                    m, _, _ = boot_ci(curves[("mu_ind", "feet")][:, i])
                    ax.plot(t, m, "--", lw=0.8, color="C1", label="feet, individual mu band")
                    m, _, _ = boot_ci(curves[("mu_ind", "hands")][:, i])
                    ax.plot(t, m, "--", lw=0.8, color="C0", label="hands, individual mu band")
                ax.axvline(0, color="k", lw=0.8); ax.axhline(0, color="k", lw=0.5)
                ax.set_title(f"{ch} {'Laplacian' if suffix else 'referential'} {band}")
                if row == 1: ax.set_xlabel("time from cue (s)")
            axes[row, 0].set_ylabel("ERD/ERS (%)  mean ± 95% CI over subjects")
        axes[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"erd_plus_{band}.png", dpi=110)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    for r, band in enumerate(("mu", "beta")):
        H, F = np.mean(topo[(band, "hands")], 0), np.mean(topo[(band, "feet")], 0)
        for j, (name, v) in enumerate((("hands", H), ("feet", F), ("hands - feet", H - F))):
            lim = np.abs(v).max()
            im, _ = mne.viz.plot_topomap(v, info64, axes=axes[r, j], show=False, cmap="RdBu_r", vlim=(-lim, lim))
            axes[r, j].set_title(f"{band} {name}\n(mean ERD% {WIN[0]:.0f}-{WIN[1]:.0f} s)")
            plt.colorbar(im, ax=axes[r, j], shrink=0.7)
        out["tables"][f"topo_{band}"] = {"hands_min_channel": info64.ch_names[int(np.argmin(H))],
                                         "feet_min_channel": info64.ch_names[int(np.argmin(F))],
                                         "hands_minus_feet_min_channel": info64.ch_names[int(np.argmin(H - F))],
                                         "hands_minus_feet_max_channel": info64.ch_names[int(np.argmax(H - F))]}
    fig.tight_layout(); fig.savefig("erd_topomaps.png", dpi=110)
    print("\nmu peaks (Hz):", out["mu_peak_hz"])
    print("topomap extremes:", {k: v for k, v in out["tables"].items() if k.startswith("topo")})
    json.dump(out, open("erd_plus_summary.json", "w"), indent=1)
    print("saved erd_plus_mu.png erd_plus_beta.png erd_topomaps.png erd_plus_summary.json")


if __name__ == "__main__":
    main()
