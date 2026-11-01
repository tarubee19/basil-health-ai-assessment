"""Task 3 evidence: cue-locked average at Fp1/Fp2 (nearest the eyes) for hands vs feet.

A class-dependent deflection right after the cue would be the signature of vertical eye
movements towards the target (top of screen = fists, bottom = feet).
Usage: python eog_probe.py   -> frontal_eog.png
"""
import numpy as np
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_raw, get_events, EVENT_ID, SUBJECTS

PICKS, TMIN, TMAX, WIN = ["Fp1", "Fp2"], -0.5, 2.0, (0.2, 1.0)

avg = {c: [] for c in EVENT_ID}
for s in SUBJECTS:
    raw = load_raw(s).filter(1.0, 15.0, fir_design="firwin", verbose="error")   # light low-pass
    ep = mne.Epochs(raw, get_events(raw), EVENT_ID, TMIN, TMAX, baseline=(TMIN, 0), picks=PICKS,
                    preload=True, verbose="error")
    for c in EVENT_ID:
        avg[c].append(ep[c].get_data(copy=True).mean(axis=(0, 1)) * 1e6)   # mean of Fp1, Fp2 (uV)
t = ep.times
H, F = np.array(avg["hands"]), np.array(avg["feet"])          # (subjects, time)

w = (t >= WIN[0]) & (t < WIN[1])
d = (H - F)[:, w].mean(axis=1)
print(f"mean Fp1/Fp2 hands-feet, {WIN[0]}-{WIN[1]} s, per subject (uV):", np.round(d, 1).tolist())
print(f"grand mean {d.mean():+.1f} uV; subjects with hands>feet: {(d > 0).sum()}/{len(d)}")
for a, b in ((0.0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 2.0)):
    m = (t >= a) & (t < b)
    print(f"  {a}-{b} s: hands {H[:, m].mean():+.1f} uV  feet {F[:, m].mean():+.1f} uV")

fig, ax = plt.subplots(figsize=(7, 3.5))
for X, name in ((H, "hands (target top)"), (F, "feet (target bottom)")):
    ax.plot(t, X.mean(0), label=name)
    ax.fill_between(t, X.mean(0) - X.std(0) / np.sqrt(len(X)), X.mean(0) + X.std(0) / np.sqrt(len(X)), alpha=0.2)
ax.axvline(0, color="k", lw=0.8); ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("time from cue (s)"); ax.set_ylabel("mean Fp1/Fp2 (uV, 1-15 Hz)")
ax.set_title("Cue-locked frontal signal, subjects 1-10 (mean ± SEM over subjects)"); ax.legend()
fig.tight_layout(); fig.savefig("frontal_eog.png", dpi=120)
print("saved frontal_eog.png")
