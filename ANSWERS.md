# ANSWERS — EEG × Deep Learning live exam

> **Note on commit timestamps:** several commits share the same timestamp because they were
> committed in one batch after a long uninterrupted work block rather than individually in real time.

### Part 1 at a glance

- **A trustworthy evaluation is not the same as a model that uses motor imagery.** The fixed pipeline's 0.566
  cross-subject accuracy comes from an eye-movement shortcut (Task 3), not from brain signal — and in Part 2,
  no attention-based inductive bias could fix that, because the real bottleneck is the shift between people,
  not the attention axis.
- **Six bugs found and fixed, one commit each.** Accuracy went from 0.667 (inflated) to 0.522 on the fixed test split, and
  0.566 with 5-fold subject-level cross-validation (3 seeds).
- **ERD matches motor anatomy for hands** (C3/C4 mu ERD). The foot effect at Cz is weak because the foot area is deep on the
  mesial wall.
- **The remaining 0.566 is an eye-movement shortcut.** 8 frontal channels alone score 0.617. That signal sits in the first
  second after the cue and below 8 Hz, and it disappears with 8–30 Hz / 1–4 s ("EOG-free").
- **Without the shortcut**, motor imagery is decodable within subjects (EEGNet 0.553 on motor channels).
  Across subjects it is only suggestive.

All numbers below come from `part1_diagnose/results.jsonl` (one row per run, labelled by `tag`) or from a quoted command
output. Training is deterministic on CPU: re-running seed 0 reproduced the untouched number exactly (0.667, best epoch 17).

---

## Part 1 — Task 1: the six defects

Line numbers refer to the untouched starter code (`git show 86e1619:part1_diagnose/<file>`). Bugs are listed in the order
they were fixed.

| ID | File:line | What was wrong | Effect on the printed accuracy | Fix commit |
|---|---|---|---|---|
| A | `train.py:62-66` | `train_test_split` on **trials**: all 10 subjects were in train, val *and* test (checked: overlap = [1..10]), although the comment and the brief claim held-out people. | **Up (inflated):** the model is tested on people it trained on, plus neighbouring, correlated trials. | `ea090f2` |
| C | `train.py:57-60` | Per-channel mean/std computed over **all 420 trials before the split**, including the 84 test trials (per-channel std all/train-only: 0.92–1.27, measured at the time on the starter runs). | **Slightly up:** test-subject amplitude scale leaks into the inputs; small effect. | `5f36e54` |
| B | `train.py:84-88` | "Best" checkpoint = **argmax of test accuracy** over 30 epochs; the validation set was computed and printed but never used. | **Up:** the maximum of a noisy test curve is optimistic; test data is used for model selection. | `7a4d202` |
| D | `data.py:15` | `RUNS = [5, 9, 13]` are the **executed** fists/feet runs; the **imagined** ones are 6, 10, 14 (PhysioNet / MNE `eegbci` run table; Part 2's loader uses 6/10/14). | **Up relative to the claimed task:** real movement gives a stronger ERD plus EMG and movement artefacts. | `0ca40ff` |
| E | `erd.py:18` | ERD `BASELINE = (0.0, 0.5)` s lies **after** the cue (t = 0), though the comment says "rest before the imagery"; epochs start at −2.0 s, so pre-cue rest exists. Now (−1.5, −0.5) s. | **No effect on accuracy** (`train.py` doesn't use it). It distorts ERD: the reference holds the cue-evoked response. Measured: post-cue baseline made mu ERD look large and uniform (−12 to −21 % at every channel for both classes); with the pre-cue baseline the condition differences appear (e.g. hands C3 −13.1 % vs feet −1.4 %). | `1ba9c20` |
| F | `data.py:19, 47-49` | Epochs cut at `[TMIN, TMAX] = [-2, 5]` s, then cropped to 0–4 s. Each run is 124.99 s and its last cue is at 120.40 s, so the window overruns and the **last trial of every run is silently dropped** (`drop_log`: `BAD boundary`, `BAD boundary`, `TOO_SHORT`): 42 of 45 trials per subject, 420 of 450. Now epoched directly on `CROP` (45/45). | **~No bias** (drops not class-dependent; the network sees the same samples), just ~7 % less train and test data. Measured move 0.548 → 0.522 is within noise. | `abc7243` |

Comments explaining each fix (`# FIX (X): …`) are in `c491ce0`. Other non-fix commits: logging (`6f890ea`), subject-level
cross-validation option `--split cv` (`df36684`, a feature, not a bug fix).

### Before / after (seed 0, `--channels all`; built from `results.jsonl` by script)

| Step | Commit | Test subjects | Test acc (selected checkpoint) | Test acc (last epoch) | n_test |
|---|---|---|---|---|---|
| before (untouched) | `86e1619` | 1–10 (trial split) | 0.667 | — | 84 |
| before (+ logging) | `6f890ea` | 1–10 (trial split) | 0.667 | 0.667 | 84 |
| + fix A | `ea090f2` | 9, 10 | 0.643 | 0.607 | 84 |
| + fix C | `5f36e54` | 9, 10 | 0.643 | 0.619 | 84 |
| + fix B | `7a4d202` | 9, 10 | 0.583 | 0.619 | 84 |
| + fix D | `0ca40ff` | 9, 10 | 0.548 | 0.512 | 84 |
| + fix F | `abc7243` | 9, 10 | **0.522** | 0.522 | 90 |
| CV seed 0 | `df36684` | all, 2 per fold | 0.584 ± 0.074 over folds (pooled 0.584) | 0.622 | 450 |
| CV seed 1 | `df36684` | all, 2 per fold | 0.540 ± 0.070 (pooled 0.540) | 0.571 | 450 |
| CV seed 2 | `df36684` | all, 2 per fold | 0.573 ± 0.062 (pooled 0.573) | 0.582 | 450 |
| **final, CV 3 seeds** | `df36684` | all | **pooled 0.566 ± 0.023 across seeds** | | 450 |

Most of the inflation came from fixes A + B + D (0.667 → 0.548). The final number can be trusted: every test person is
unseen in training and model selection, the checkpoint is chosen on validation subjects, and normalisation uses
training-subject statistics only. One caveat remains: the classifier may be reading a non-brain signal (the fists target
appears at the top of the screen and the feet target at the bottom, creating an eye-movement shortcut). Task 3 tests this.

---

## Part 1 — Task 2: ERD (`part1_diagnose/erd_mu.png`, plus `erd_beta.png`)

Settings: subjects 1–10, imagined runs, pre-cue baseline −1.5…−0.5 s.

Mean ERD% over 1–4 s, averaged over subjects (reproduced by `erd_plus.py`, saved in `erd_plus_summary.json`).

| band | condition | C3 | Cz | C4 |
|---|---|---|---|---|
| mu 8–13 | hands | **−12.6** | −3.1 | −7.8 |
| mu 8–13 | feet | −0.4 | −4.6 | −4.1 |
| beta 13–30 | hands | −6.6 | −6.0 | −5.0 |
| beta 13–30 | feet | −2.8 | −6.4 | −7.1 |

Hand imagery gives a clear mu ERD over the lateral hand areas (C3 −12.6 %, C4 −7.8 %) with little at Cz (−3.1 %),
matching the hand representation on the lateral motor strip. Foot imagery leaves C3 almost unchanged (−0.4 %) and gives
only a marginally larger ERD at Cz than hands do (−4.6 % vs −3.1 %), so the contrast is carried mainly by whether the hand
area desynchronises. Cz is weaker than expected because the foot area lies deep on the mesial wall with partly tangential
sources; a Laplacian check (`erd_plus.py`) confirms this — it makes feet mu ERD at Cz weaker, not sharper (−4.6 % → +2.8 %).

---

## Part 1 — Task 3: what is the model using?

### Decision rule (written and committed before running any channel-subset model)

Protocol for every row:
- 5-fold subject-level CV (`--split cv`), seeds 0, 1, 2;
- the per-seed pooled accuracy is over all 450 test trials;
- "above chance" means ≥ 0.540 (one-sided binomial p < .05, n = 450);
- "close to `all`" means within 5 points of `all` (pooled 0.566, mean of 3 seeds).

| Result under the fixed protocol | Reading |
|---|---|
| `frontal` clearly above chance in all 3 seeds and close to `all` | the model can solve the task from eye-region signals, so the cue/eye confound is a viable shortcut |
| `frontal` near chance, `motor` close to `all` | evidence that the model uses motor cortex |
| anything in between | report as mixed, give the numbers, don't over-claim |

Probes, to tell eye movements apart from brain activity:
- **`frontal --crop 1 4`** removes the first second after the cue, where the saccade to the target and the visual evoked
  response are. If frontal accuracy collapses, the frontal signal was cue-locked.
- **`--fband 8 30`** removes the < 8 Hz band, where most EOG (saccade/blink) energy lies. If frontal drops but motor holds,
  the frontal signal was ocular.

### Results

Protocol: 5-fold subject CV, seeds 0/1/2, pooled over 450 test trials per seed; chance threshold 0.540. Rows are tagged
`t3_*` in `results.jsonl`.

| Run | Channels | Crop (s) | Band (Hz) | Pooled acc, seed 0 / 1 / 2 | Mean | Best | Worst | Seeds above chance |
|---|---|---|---|---|---|---|---|---|
| all | 64 | 0–4 | 1–40 | 0.584 / 0.540 / 0.573 | 0.566 | 0.584 | 0.540 | 3/3 |
| **frontal** | Fp1 Fpz Fp2 AF7 AF3 AFz AF4 AF8 | 0–4 | 1–40 | 0.607 / 0.622 / 0.622 | **0.617** | 0.622 | 0.607 | 3/3 (p ≤ 3.5e-6) |
| motor | 21 FC/C/CP | 0–4 | 1–40 | 0.602 / 0.602 / 0.616 | 0.607 | 0.616 | 0.602 | 3/3 |
| frontal, cue second removed | frontal | **1–4** | 1–40 | 0.533 / 0.513 / 0.504 | 0.517 | 0.533 | 0.504 | 0/3 |
| frontal, mu/beta only | frontal | 0–4 | **8–30** | 0.484 / 0.478 / 0.487 | 0.483 | 0.487 | 0.478 | 0/3 |
| motor, mu/beta only | motor | 0–4 | **8–30** | 0.516 / 0.520 / 0.507 | 0.514 | 0.520 | 0.507 | 0/3 |

Evidence plot `part1_diagnose/frontal_eog.png` (`eog_probe.py`, Fp1/Fp2 mean, 1–15 Hz, baseline −0.5…0 s):
- From about 0.2 s after the cue, the frontal signal goes **positive for hands (target at the top)** and **negative for feet
  (target at the bottom)**: +11.8 vs −15.9 µV at 0.2–0.5 s, and +8.0 vs −10.3 µV at 0.5–1 s.
- Hands > feet in 0.2–1 s in **10/10 subjects** (sign test p ≈ 0.001).
- The difference has gone by 1–2 s (−0.2 vs −2.4 µV).

### Answer

The 8 electrodes nearest the eyes decode fists vs feet better than all 64 channels (0.617 vs 0.566 across subjects),
so the model can solve the task without motor cortex. That information sits entirely in the first second after the cue
and below 8 Hz — the signature of a vertical eye movement (EOG) towards the target, whose polarity flips with class in
10/10 subjects. Motor channels reach 0.607 at 1–40 Hz but 0.514 (chance) when restricted to the mu/beta band, so
cross-subject the pipeline's accuracy came from the cue, not from motor-imagery ERD.

---

## Part 1 — Task 4: short answers

1. **58 % on 45 trials:**
   - **Not convincingly above chance.** 26/45 = 0.578 with a one-sided binomial test against 0.5 gives p = 0.186; 29/45
     (64.4 %, p = 0.036) is needed, and the 95 % CI for 26/45 is 0.42–0.72.
   - **How to check properly:** run a label-permutation test with this exact pipeline. Retrain about 100+ times on
     shuffled labels to get the empirical null distribution, which also accounts for class imbalance and dependence
     between trials.
2. **The depthwise layer `Conv2d(F1, F1*D, (n_ch, 1), groups=F1)`:** each of the F1 temporal filters is
   frequency-selective. For each one, the layer learns D **spatial filters**, which are weightings across all electrodes,
   applied with no mixing between bands. It learns which electrodes to combine to expose a band-power pattern — for
   example, contrasting C3/C4 against Cz to pick up mu ERD differences between hands and feet.
3. **One change that would stop the model using the eye signal:**
   - **Experiment:** make the cue identical in location for both classes, for example a central symbol or an auditory cue,
     so gaze no longer depends on the class. Alternatives:
     - counterbalance or randomise the cue position across blocks;
     - use a delayed cue and analyse only after the cue has disappeared.
   - **Analysis:** drop the first ~1 s after the cue **and** band-pass to 8–30 Hz, and/or remove ocular components (ICA or
     regression on Fp channels).

---

## Part 1 — Bug evidence at a glance

| Bug | How I showed it was wrong | Which way it moved accuracy |
|---|---|---|
| A, trial split | Reproduced the split: all 10 subjects in train, val and test (overlap [1..10]) | Down when fixed (inflated before): 0.667 → 0.643 |
| C, z-score on all trials | The statistics were computed at starter L57-60 before the split at L65, over 420 trials including 84 test trials; per-channel std ratio 0.92–1.27 | Small inflation removed: 0.643 → 0.643 (0 at seed 0) |
| B, test-selected checkpoint | Starter log: chosen epoch 17 = first argmax of test (0.667); val peaked elsewhere (epochs 14, 15, 28) and was never used | Down when fixed: 0.643 → 0.583 |
| D, executed runs | MNE `eegbci` / PhysioNet run table: 5/9/13 = execution, 6/10/14 = imagery; the docstring and Part 2 say imagery | Down when fixed: 0.583 → 0.548 |
| E, post-cue ERD baseline | Epochs start at −2.0 s; the (0, 0.5) s window contains the +50–85 % cue-locked power peak at ~0.25 s; ERD numbers under both baselines compared | None on accuracy; ERD was inflated and uniform before |
| F, trials dropped at run end | `drop_log`: 3 trials per subject (`BAD boundary` ×2, `TOO_SHORT`); run 124.99 s, last cue 120.40 s, so +5 s overruns; epoching on 0–4 s keeps 45/45 | ~None (within noise): 0.548 → 0.522, n_test 84 → 90 |

**What this still doesn't rule out:**
- A trustworthy *evaluation* is not the same as a model that uses *motor imagery*: the fixed pipeline's 0.566 comes from the
  eye-movement shortcut (Task 3).
- Without the shortcut, motor imagery decodes within subjects but not reliably across them, and frontal channels still carry
  some within-subject information (Task 3, EOG-free checks).
- The fixed {9, 10} test split (90 trials, 0.522, p = 0.38) is too small to read a single number from; use the CV numbers.

---

## Part 2 — Build & Defend (`part2_transformer/`)

Code: `part2_transformer/main.py` (`YourModel`). Numbers: `part2_transformer/results.jsonl`. Full write-up:
`part2_transformer/README.md`. Reproduce: `cd part2_transformer && python -u main.py --epochs 20 --seeds 0 1 2`.

> **Reproducibility note:** reported Part 2 numbers use `python main.py --epochs 20 --seeds 0 1 2`; the script's
> own `--epochs` default is still 40 (the starter's original value), so running `python main.py` with no flags
> will not reproduce these exact numbers.

### Part 2 at a glance

- **The bet:** attend across 64 **electrode tokens** (log band-power, placed by 3-D scalp geometry) instead of 481
  timesteps. The attention matrix shrinks from 482² to 65².
- **Fixed split (test subjects 9–10, 3 seeds):** geom had the highest mean (0.574), but within one SE of vanilla (0.530) and
  EEGNet (0.526). No conclusive win, and seed variance was not lower.
- **All 10 subjects rotated (validation-selected checkpoints):** every neural model is at chance (geom 0.496,
  best EEGNet 0.527). Real vs shuffled geometry makes no difference (−0.004, p = 0.55), and removing attention changes
  nothing. The fixed-split lead and the rim attention map were noise from one split.
- **Lesson:** with 6–8 training subjects the bottleneck is the shift between people, not the attention axis.

### 1. Hypothesis (committed in `07411fc`, before the model code in `0c465b7`)

> Fists-vs-feet is encoded in *where* mu/beta power drops (lateral hand areas vs the medial vertex), not in *when*
> individual samples occur. Attending across 64 electrode tokens that carry log band-power and know their 3-D scalp
> position should beat the vanilla transformer with lower seed variance, and its attention should concentrate on
> central (FC/C/CP, Cz) rather than frontal electrodes.

**The one bias:** move attention from the time axis to the space axis (electrode tokens), with each token located by
scalp geometry.

### 2. What I changed and why

Each electrode becomes one token: a shared temporal filter bank (16 × 0.4 s kernels) is applied, squared, averaged
over 0.5 s windows and logged, yielding phase-invariant log band-power. Attention runs across the 64 electrodes
(not 481 timesteps), shrinking the attention matrix from 232,324 to 4,225 entries per head (≈ 55×). An MLP on
standardised xyz gives each token its scalp location; ablations: `shuffled` assigns xyz to the wrong electrodes,
`learned` replaces geometry with a learned per-electrode embedding.

> **Note on scope:** YourModel combines three changes — log band-power tokenization, electrode-as-token attention
> axis, and geometric positional encoding. The electrode/space-vs-time attention axis is the actual bet being tested;
> band-power tokenization and the geometric PE are scaffolding needed to make that axis meaningful, not separate
> hypotheses.

### 3. What happened

Final-epoch test accuracy on subjects 9–10 (n = 90), seeds 0/1/2, 20 epochs. Chance is 0.5; p < .05 needs 0.600;
SE is 5.3 pts.

| Model | params | final, seed 0 / 1 / 2 | best | worst | mean ± std | peak mean |
|---|---|---|---|---|---|---|
| **YourModel (geom)** | 76,930 | 0.478 / 0.644 / 0.600 | 0.644 | 0.478 | 0.574 ± 0.070 | 0.611 |
| YourModel (shuffled) | 76,930 | 0.456 / 0.544 / 0.422 | 0.544 | 0.422 | 0.474 ± 0.052 | 0.526 |
| YourModel (learned) | 78,786 | 0.433 / 0.556 / 0.533 | 0.556 | 0.433 | 0.507 ± 0.053 | 0.552 |
| EEGNet (supplied) | 2,610 | 0.522 / 0.589 / 0.467 | 0.589 | 0.467 | 0.526 ± 0.050 | 0.541 |
| EEGNet-P1 (`kern=80, p=0.5`) | 2,738 | 0.533 / 0.411 / 0.467 | 0.533 | 0.411 | 0.470 ± 0.050 | 0.537 |
| PatchTransformer (starter) | 232,898 | 0.511 / 0.489 / 0.511 | 0.511 | 0.489 | 0.504 ± 0.010 | 0.593 |
| VanillaTransformer | 135,746 | 0.533 / 0.522 / 0.533 | 0.533 | 0.522 | 0.530 ± 0.005 | 0.552 |

- **No conclusive win.** geom is the only model with seeds at or above 0.600 (seeds 1 and 2) and has the highest mean,
  but seed 0 is at chance and the mean is only 4.4 pts above vanilla and 4.8 above EEGNet: inside one SE.
- **"Lower seed variance" is contradicted.** geom has the widest spread (std 0.070); vanilla is flat at chance (0.005).
- **The ablation points the right way but is not conclusive:** shuffled 0.474 and learned 0.507 vs geom 0.574.
  These are 7–10 pt gaps on 3 seeds and one 90-trial test set, and neither ablation is above chance.
- **Both EEGNets are at chance here** (0.526, 0.470). That is consistent with Part 1 Task 3: in the mu/beta band
  (Part 2 filters 7–30 Hz) the low-frequency eye shortcut is gone, and so is cross-subject accuracy.
- **Reproducibility:** retraining geom seed 0 reproduced final 0.478 / peak 0.567 exactly (deterministic CPU).
- **Supplied files untouched:** `git diff 86e1619` on `models.py`, `data.py` and `train.py` is empty.

#### Subject rotation (all 10 subjects)

5-fold subject CV (same folds as Part 1), validation-selected checkpoints, seeds 0–2, 40 epochs, on Part 2's 7–30 Hz,
0–3 s data (`part2_transformer/cv.py`, rows tagged B1/B6/B7 in `results.jsonl`, `figure_cv.png`). The p < .05 line on 450
pooled trials is 0.540. "p vs 0.5" is a Wilcoxon test over the 10 subjects, uncorrected; "geom − model" is paired over
the same subjects. Vanilla was not run under rotation; on the fixed split it was at chance, 0.530.

| Model | pooled accuracy, seed 0 / 1 / 2 | best run | worst run | p vs 0.5 | geom − model (p) |
|---|---|---|---|---|---|
| **YourModel (geom)** | 0.516 / 0.511 / 0.462 | 0.567 | 0.411 | 0.576 | — |
| YourModel (shuffled per seed) | 0.516 / 0.509 / 0.478 | 0.567 | 0.411 | 0.221 | -0.004 (0.547) |
| YourModel (learned) | 0.487 / 0.502 / 0.511 | 0.600 | 0.400 | 0.697 | -0.004 (1.000) |
| YourModel (no position) | 0.496 / 0.502 / 0.507 | 0.589 | 0.422 | 0.867 | -0.005 (0.453) |
| TokensNoAttn | 0.473 / 0.498 / 0.491 | 0.578 | 0.389 | 0.645 | +0.009 (0.086) |
| EEGNet (supplied) | 0.522 / 0.538 / 0.522 | 0.678 | 0.400 | 0.477 | -0.031 (0.492) |
| EEGNet-P1 | 0.504 / 0.496 / 0.493 | 0.556 | 0.389 | 0.471 | -0.001 (0.652) |
| PatchTransformer | 0.489 / 0.498 / 0.500 | 0.533 | 0.444 | 0.754 | +0.001 (0.922) |

- **Every neural model is at chance.** None reaches 0.540 in any seed (geom 0.496, best EEGNet 0.527).
- **The fixed-split lead was noise from the single subjects 9–10 split.** geom vs shuffled is −0.004 (p = 0.55) and vs
  EEGNet −0.031 (p = 0.49); the fixed-split ablation gap was 0.574 vs 0.474.
- **Attention adds nothing shown:** geom is +0.009 ahead of TokensNoAttn, which removes attention (p = 0.086).

### 4. Figure (`part2_transformer/figure.png`)

Panel (a) shows accuracy per seed for every model. Panels (b) and (c) show the geom seed-0 model's last-layer
CLS→electrode attention.

That attention sits on the scalp rim, on periocular and temporal sites, not on the motor strip (C3/Cz/C4 are near the
uniform 1/64):
- fists: Fp2, F8, AF8, Fpz, AF3, Fp1;
- feet: AF8, T10, Fp1, T9, TP8.

So the **attention-location half of the hypothesis failed**. The map is consistent with the Part 1 cue/eye (EOG) and
muscle confound. Two caveats: it comes from a seed that is itself at chance, and topomap interpolation exaggerates the
rim. The t-SNE (`figure_tsne.png`) shows no class separation, and would not be evidence of mechanism anyway.

**Under rotation:** the rim attention did not replicate: region attention is about uniform (0.955–1.03× the
uniform share), seed-to-seed Spearman is 0.006, and it is unrelated to region occlusion (−0.08). The original map is best
read as noise from a single at-chance model.

### What failed / next

- **Failed:** attention location (rim, not FC/C/CP); stability (highest variance); a clear margin over vanilla or EEGNet.
- **Limits:** 2 test subjects (the two hardest in Part 1's CV), 90 trials, 3 seeds, 20 epochs.
- **Checked:** subject rotation (see "Subject rotation (all 10 subjects)"); attention vs region occlusion (see
  "4. Figure").
- **Not run:**
  - retraining without the rim or the first second after the cue;
  - Euclidean alignment per subject for the neural models;
  - a scalp-distance attention bias (−λ·distance on the logits), so geometry *constrains* attention;
  - a learning curve (train on 2, 4, 6 subjects).

### Walkthrough notes

- **√d:** q·k sums d_head products, so its variance grows ∝ d_head. Without scaling the softmax saturates and its
  gradients vanish; /√d_head keeps the logits at about unit variance.
- **Sequence length / cost:** vanilla has 482 tokens, ours 65. Cost is O(N²·d) per layer: 232,324 vs 4,225 attention
  entries per head.
- **80 subjects:** spatial priors matter less with more data, but phase-invariance still matters: induced-rhythm
  sample amplitudes never align across trials. A class-neutral cue location would matter more than more subjects.
- **Where it attends:** the fixed-split seed-0 map showed the periocular/temporal rim, but under rotation attention is about
  uniform and unstable across seeds (Spearman 0.006). That map is best read as noise from a single at-chance model.
