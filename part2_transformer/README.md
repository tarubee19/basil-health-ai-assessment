# EEG Transformer Sprint — Starter Kit

You have 90 minutes. This kit removes all the plumbing so you spend the time on
one thing: **giving self-attention a fighting chance on tiny EEG data, and
understanding why your idea does (or doesn't) help.**

## Setup (first ~10 minutes)

```bash
pip install -r requirements.txt
python data.py      # sanity check: downloads ~50 MB, prints array shapes
python main.py      # runs both baselines + the starter YourModel + saves figure.png
```

Everything runs on CPU in well under a minute per model. The first run
downloads the PhysioNet EEGBCI data and caches it.

## The task

**Both fists vs. both feet** (imagined movement), cross-subject: train on
subjects 1–8, test on held-out subjects 9–10.

This is a **bilateral** task. The discriminative signal is largely
anterior–posterior (hand/arm motor area vs. the medial foot representation near
the vertex, ~Cz), **not** the classic left/right lateralised mu/beta ERD. A
spatial prior that only encodes left vs. right will not help you here.

## What's in the box

| File | What it is |
|------|------------|
| `data.py` | Loader: download, 7–30 Hz filter, epoch, cross-subject split, per-channel z-score (train stats only). Also `get_channel_positions()` for 3D electrode coordinates. |
| `models.py` | `EEGNet` (conv-prior baseline) and `VanillaEEGTransformer` (per-timestep tokens, **no** EEG prior — the thing to beat/understand). |
| `train.py` | Shared train/eval loop, seed control, data loaders. |
| `main.py` | Entry point. Runs the baselines, runs **your** model across seeds, reports best + worst, saves `figure.png`. |

## Your job — in `main.py`

Implement **one** inductive bias in `YourModel` (or replace it entirely). The
starter version already shows one legal example: learned **temporal patch**
tokens instead of per-timestep tokens, which shrinks the O(T²) attention
surface. Swap in your own bet.

Do **not** quietly edit the vanilla transformer in `models.py` — the whole point
is the contrast between "attention with no help" and "attention with your one
idea."

## What to hand in

- This repo with your `YourModel` filled in. `python main.py` must reproduce
  your numbers end to end, seed printed.
- A short note (half a page — edit the section below is fine) with:
  1. **Hypothesis** — one line, written *before* you coded it.
  2. **What you changed and why** it targets a real weakness of attention on EEG.
  3. **What happened** — best *and* worst run across seeds, no cherry-picking.
  4. One sentence on what `figure.png` shows.
- A live code walkthrough at the end: share your screen, walk us through the
  file, answer questions.

Accuracy is **not** scored. We're reading your code and your answers for how you
reason about EEG and attention under a clock. A clear negative result beats a
lucky spike you can't explain.

---

## Your write-up (fill this in)

**Hypothesis:** Fists-vs-feet is encoded in *where* mu/beta power drops (lateral hand areas vs the medial vertex, an
anterior–posterior/medial contrast), not in *when* individual samples occur; per-timestep attention compares random-phase
amplitudes it can't generalise from 8 subjects, so attending across 64 electrode tokens that carry log band-power and
know their 3-D scalp position should beat the vanilla transformer with lower seed variance, and its attention should
concentrate on central (FC/C/CP, Cz) rather than frontal electrodes.

*The one bias:* change the attention axis from time to space (electrode tokens), with tokens located by scalp geometry.
The geometry part is tested by an ablation (shuffled / learned position) so we can say whether it matters.

Hypothesis committed in `07411fc`, before any model code (`0c465b7`).

**What I changed and why:**
- **The weakness:** per-timestep attention has no locality prior. It compares single samples whose phase is random from
  trial to trial: motor-imagery ERD is *induced*, not phase-locked, so on 8 subjects there is little to generalise.
- **Tokens:** each of the 64 electrodes becomes one token. A shared temporal filter bank (16 filters, 0.4 s kernels) is
  applied to each electrode, squared, averaged over 0.5 s windows and logged. That gives log band-power, which is
  phase-invariant, like ERD itself.
- **Attention axis:** attention runs across electrodes, because the task signal is spatial: lateral hand areas vs the
  medial foot area near Cz. That is an anterior–posterior/medial contrast, so a left/right-only prior would not help.
- **Positional encoding:** an MLP on standardised 3-D xyz gives each token its scalp location.
- **Cost:** the attention matrix shrinks from 482² = 232,324 to 65² = 4,225 entries per head (≈ 55× smaller).
- **Ablation:** the same model with xyz assigned to the wrong electrodes (`shuffled`) or a learned per-electrode
  embedding (`learned`) tests whether *geometry* matters, or just having electrode tokens.

**Setup / deviations:** same split (train 1–8, test 9–10, n_test = 90), preprocessing, harness, lr and weight decay for
every model; seeds 0, 1, 2 for all. **Time-limit deviation:** all models were trained for **20 epochs**, not the starter
default of 40 (same for every model). `train_model`'s "best" is the max of the test-accuracy curve, i.e. chosen on the
test set (Part 1's checkpoint bug), so the headline is **final-epoch** accuracy, with peak alongside. On 90 trials the
SE at 50 % is 5.3 pts and one-sided p < .05 needs ≥ 60 % (54/90), so model differences under ~7–10 pts are not conclusive.
Reproduce with `python -u main.py --epochs 20 --seeds 0 1 2`; every run is appended to `results.jsonl`.

**Results (best / worst / mean across seeds):** final-epoch test accuracy on subjects 9–10 (n = 90), 20 epochs, seeds
0/1/2, std over seeds (ddof = 0); all rows from `results.jsonl`.

<!-- table -->
| Model | params | seeds done | final per seed (0/1/2) | best | worst | mean ± std | peak mean |
|---|---|---|---|---|---|---|---|
| **YourModel (geom)** | 76,930 | 0,1,2 | 0.478 / 0.644 / 0.600 | 0.644 | 0.478 | 0.574 ± 0.070 | 0.611 |
| YourModel (shuffled) | 76,930 | 0,1,2 | 0.456 / 0.544 / 0.422 | 0.544 | 0.422 | 0.474 ± 0.052 | 0.526 |
| YourModel (learned) | 78,786 | 0,1,2 | 0.433 / 0.556 / 0.533 | 0.556 | 0.433 | 0.507 ± 0.053 | 0.552 |
| EEGNet (supplied) | 2,610 | 0,1,2 | 0.522 / 0.589 / 0.467 | 0.589 | 0.467 | 0.526 ± 0.050 | 0.541 |
| PatchTransformer | 232,898 | 0,1,2 | 0.511 / 0.489 / 0.511 | 0.511 | 0.489 | 0.504 ± 0.010 | 0.593 |
| VanillaTransformer | 135,746 | 0,1,2 | 0.533 / 0.522 / 0.533 | 0.533 | 0.522 | 0.530 ± 0.005 | 0.552 |
| EEGNet-P1 (k80,p.5) | 2,738 | 0,1,2 | 0.533 / 0.411 / 0.467 | 0.533 | 0.411 | 0.470 ± 0.050 | 0.537 |
<!-- /table -->

**Run provenance:** VanillaTransformer and EEGNet-P1 (the Part 1 config, `kern=80, p=0.5`, set via constructor
arguments; `models.py` unchanged) were run after the time limit with `--only vanilla` and `--only eegnet_p1`: same
harness, seeds and 20 epochs. Vanilla took 310–1073 s per seed because the CPU was shared with Part 1.
**Reproducibility:** `--figure-only` retrained YourModel (geom) seed 0 twice; both runs gave final 0.478 and peak 0.567,
identical to `results.jsonl`, because CPU training is deterministic here.

**Reading:**
- **Nothing is conclusively above chance on average.** Only YourModel (geom) has seeds at or above the p < .05 line
  (seed 1 0.644, seed 2 0.600), and it has the highest mean (0.574). But its seed 0 is at chance (0.478), and its mean is
  only 4.4 pts above vanilla (0.530) and 4.8 above EEGNet (0.526), inside one SE (5.3 pts) of a 90-trial test.
- **Vanilla vs ours:**
  - The vanilla transformer sits at chance in all 3 seeds (0.522–0.533), with almost no spread (std 0.005).
  - So "beat the vanilla transformer" is at most a trend, and "with lower seed variance" is **contradicted**: geom's std
    is 0.070, 14× vanilla's.
  - Vanilla's flat ~0.53 looks more like a model that has not learned anything that transfers to new people than like a
    noisy one. I did not log training accuracy, so I can't say whether it overfits or underfits.
- **Both EEGNets are at chance on this pipeline** (supplied 0.526, Part 1 config 0.470). That is consistent with Part 1:
  on subjects 9–10 EEGNet scored 0.522, and in the 8–30 Hz band even motor channels fell to chance. Part 2's loader
  filters to 7–30 Hz, which removes the low-frequency eye signal Part 1 found EEGNet exploiting.
- **Low variance is not the same as learning.** The patch model (std 0.010) and vanilla (0.005) are the most stable rows
  and both sit at chance. The patch model also has the largest peak-vs-final gap (8.9 pts).
- **Peak flatters every model** (by 1.5–8.9 pts on the means), which is why final is the headline.

**Figure:** `figure.png`: (a) final accuracy per seed for every model; (b, c) the geom seed-0 model's last-layer
CLS→electrode attention for fists and feet. Attention is **not** central: it sits on the scalp rim. For fists the top
electrodes are Fp2, F8, AF8, Fpz, AF3 and Fp1 (frontal / periocular); for feet they are AF8, T10, Fp1, T9 and TP8
(periocular and temporal). C3/Cz/C4 sit near the uniform level of 1/64. Those rim sites are where eye movements (EOG) and
temporal-muscle activity (EMG) dominate, not the motor strip. This is the Part 1 Task 3 cue/eye confound surviving the
7–30 Hz filter, though only in a model whose own accuracy is at chance (seed 0: 0.478), so it is weak evidence about
*any* mechanism. The topomap's interpolation also exaggerates rim values. `figure_tsne.png`: the test CLS embeddings do
**not** separate by class; t-SNE is in any case not evidence of a neural mechanism.

**What failed / next:**
- **Hypothesis on attention location: failed.** The model does not attend to FC/C/CP or Cz.
- **Geometry ablation:** scrambling which electrode gets which xyz (`shuffled`, mean final 0.474) or replacing geometry
  with a learned per-electrode embedding (`learned`, 0.507) scored below `geom` (0.574). Gaps of 7–10 pts on 3 seeds and
  one 90-trial test set are suggestive, not shown, and neither ablation is above chance.
- **Seed variance is not lower:** geom final ranges 0.478–0.644, the widest spread of any row.
- **Small test set:** only 2 test subjects (both among the hardest in Part 1's CV) and 90 trials make any single number
  noisy (SE 5.3 pts). 3 seeds each.
- **Next:**
  - rotate the held-out pair (subject-level CV, as in Part 1);
  - Euclidean alignment per subject, for the cross-subject covariance shift;
  - a scalp-distance attention bias (add −λ·distance to the logits), so geometry constrains attention instead of only
    labelling tokens;
  - drop 0–1 s after the cue, or regress out Fp channels, to deny the model the eye shortcut, then re-check where it
    attends.

**Walkthrough prep:**
- **√d scaling:** q·k is a sum of d_head products, so its variance grows like d_head. Unscaled logits saturate the
  softmax (near one-hot), and its gradients vanish. Dividing by √d_head keeps the logits at about unit variance.
- **Sequence lengths:** vanilla 481 timesteps + CLS = 482 tokens; ours 64 electrodes + CLS = 65.
- **Cost:** O(N²·d) per layer. Attention entries per head: 232,324 vs 4,225. Parameters: vanilla 135,746, ours 76,930,
  EEGNet 2,610.
- **With 80 subjects:** priors matter less, since attention could learn spatial structure from data. The phase-invariance
  argument still holds: per-sample amplitudes of an induced rhythm don't line up across trials however many subjects
  you have. A cue that sits at the same screen location for both classes matters more than more data.
- **Where it attends:** see the topomaps above. Rim and periocular electrodes, not the motor strip. That is not
  neurophysiologically sensible for motor imagery; it is consistent with the eye confound.
