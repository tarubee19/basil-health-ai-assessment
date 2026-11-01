# ANSWERS — EEG × Deep Learning live exam

### Part 1 at a glance

- **Six bugs found and fixed, one commit each.** Accuracy went from 0.667 (inflated) to 0.522 on the fixed test split, and
  0.566 with 5-fold subject-level cross-validation (3 seeds).
- **ERD matches motor anatomy for hands** (C3/C4 mu ERD). The foot effect at Cz is weak because the foot area is deep on the
  mesial wall.
- **The remaining 0.566 is an eye-movement shortcut.** 8 frontal channels alone score 0.617. That signal sits in the first
  second after the cue and below 8 Hz, and it disappears with 8–30 Hz / 1–4 s ("EOG-free").
- **Without the shortcut**, motor imagery is decodable within subjects (CSP 0.711 on motor channels; EEGNet 0.553).
  Across subjects it is only suggestive (best 0.633, not significant after correction).

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

**Is it above chance?** One-sided binomial test vs 0.5 (`scipy.stats.binomtest`):
- Fixed split {9, 10}: 47/90 = 0.522, p = 0.376. **Not above chance**; 54/90 = 0.600 would be needed.
- CV, pooled over all 450 trials (every subject tested once): seed 0 263/450 = 0.584, p = 0.0002; seed 1 243/450 = 0.540,
  p = 0.049; seed 2 258/450 = 0.573, p = 0.001. That is **weakly but reliably above chance**; 243/450 = 0.540 is needed.
- Why the two disagree: subjects 9 and 10 are the two hardest in CV (mean per-subject accuracy over seeds 0.43 and 0.50; the
  best is subject 7 at 0.68). Two test people and 90 trials (SE ≈ 5 points) are too few to read a single number. The fold sd
  (0.06–0.07) shows how much the choice of test people matters.

**Notes on the path:**
- Most of the inflation came from A + B + D: 0.667 → 0.548 on the fixed split.
- C changed nothing at seed 0, which is expected for a small scale leak.
- F moved 0.548 → 0.522, the "wrong" way for a fix that adds data, but by 2.6 points on 90 trials, well inside noise.
- With B fixed, the val-selected and last-epoch numbers no longer agree perfectly. That is the honest situation: on ~90 test
  trials, they differ by noise.

**Why the final number can be trusted:**
- every test person is unseen in training and in model selection;
- the checkpoint is chosen on separate validation people;
- normalisation uses training-subject statistics only;
- the data are the imagined runs the brief describes;
- no trials are silently lost.

**What it still doesn't rule out:** the classifier may be reading a non-brain signal that is locked to the class. The fists
target appears at the top of the screen and the feet target at the bottom, so vertical eye movements (EOG) and the visual
evoked response separate the classes. Task 3 tests this.

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

**Short answer.** Hand imagery gives a clear mu ERD over the lateral hand areas (C3 −12.6 %, C4 −7.8 %) and little at Cz
(−3.1 %), which matches the hand representation on the lateral motor strip. Foot imagery leaves C3 almost unchanged (−0.4 %)
and gives only a slightly larger ERD at Cz than hands do (−4.6 % vs −3.1 %), so the contrast is carried mainly by whether
the hand area desynchronises. Cz is weaker than expected because the foot area lies deep on the mesial wall with partly
tangential sources, so little of its activity reaches the scalp; a Laplacian check (below) supports this.

**Details.**
- **Hands:** mu ERD that starts about 0.5 s after the cue and is clearly strongest over the lateral hand areas: C3 −12.6 %
  (8/10 subjects negative) and C4 −7.8 %, with only −3.1 % at Cz. This matches the hand representation on the lateral
  convexity of the motor cortex.
- **Feet:** almost no ERD over the hand areas (C3 −0.4 %, with short ERS bursts). The ERD at Cz is only marginally larger
  than for hands (−4.6 % vs −3.1 %; 7/10 subjects negative). So the hands-vs-feet contrast is carried mainly by *where the
  hand area desynchronises*, not by a strong foot ERD at the vertex.
- **Why Cz is weaker than expected:**
  - the foot area lies on the mesial wall, deep inside the interhemispheric fissure;
  - its sources are partly tangential, and the two feet's fields partly cancel, so little reaches the scalp at Cz;
  - the reference is unknown and there is no spatial filter (Laplacian/CSP), so Cz is blurred by its neighbours;
  - averaging subjects with different mu peak frequencies flattens the curves further.
  Beta was not clearer for feet here (Cz −6.4 % feet vs −6.0 % hands).
- *Laplacian check (`erd_plus.py`, `erd_plus_mu.png`):* a small Laplacian makes feet mu ERD at Cz weaker, not sharper (−4.6 % →
  +2.8 %, 7/10 → 4/10 subjects negative). So the "blurred by its neighbours" bullet is not supported, and the deep, mesial,
  tangential source is the better explanation: a Laplacian favours superficial radial sources.

**What else the plots show:**
- **Cue response, not motor:** a sharp +50 to +85 % mu power peak at about 0.25 s at all three channels and for both
  classes. This is cue-locked evoked activity (visual/eye), which is exactly why the old 0–0.5 s baseline was wrong.
- **Beta rebound:** after the trial ends (4–5 s, task offset at 4.1 s) for hands: C3 +9.6 %, C4 +11.8 %.
- **Baseline sanity check:** the pre-cue window fluctuates ±10–20 % around 0 with no trend; this level of noise comes from
  ~225 trials per condition.

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

**Short answer.** The 8 electrodes nearest the eyes decode fists vs feet better than all 64 channels (0.617 vs 0.566 across
subjects), so the model can solve the task without motor cortex. That information sits entirely in the first second after
the cue and below 8 Hz, and its polarity flips with class in 10/10 subjects (positive when the target is at the top). That is
the signature of a vertical eye movement (EOG) towards the cue, plus visual evoked activity. With the eye signal removed,
EEGNet falls to chance across subjects, so the pipeline's cross-subject accuracy came from the cue, not from motor imagery.

**Details.**
- **Which rule row applies:** the first one. The 8 electrodes nearest the eyes decode hands vs feet *better* than all
  64 channels (0.617 vs 0.566, above chance in 3/3 seeds).
- **Where that information sits:** entirely in the first second after the cue and below 8 Hz. Dropping 0–1 s
  (0.517) or keeping only 8–30 Hz (0.483) puts frontal back at chance.
- **Which non-brain signal explains it:** that time course and band are the signature of a **vertical eye movement (EOG)**
  towards the target. The fists target is at the top and the feet target at the bottom, so the polarity flips with class
  and matches the corneo-retinal dipole: looking up brings the positive cornea towards Fp1/Fp2. Blinks and frontalis EMG
  also contribute.
- **The worrying part:** the motor-strip control does not rescue a "brain" reading. Motor channels reach 0.607 at
  1–40 Hz, but **0.514 (chance) when restricted to the mu/beta band** where the Task 2 ERD lives. So cross-subject, the
  model's above-chance accuracy is explained by slow, cue-locked activity: EOG spreading back over the scalp (visible at
  C3/Cz/C4 in the raw traces) and/or visual evoked potentials, not by motor-imagery ERD.

### Is there a brain signal without the eye shortcut?

Setting: "EOG-free" = 8–30 Hz **and** crop 1–4 s; same subject folds as `--split cv`. Statistics are per subject (n = 10):
mean, one-sided Wilcoxon p vs 0.5, and the number of subjects above 0.5. Classical baselines are in `classical.py`
(CSP+LDA, tangent space + logistic regression), statistics in `stats.py` (`stats_table.md`, `accuracy_grid.png`), and
the rows in `results.jsonl` (tags `a1_*`, `a2_*`, `a3_*`, `c3_*`).

| EOG-free | CSP+LDA | TS+LR | EEGNet (seeds 0–2) |
|---|---|---|---|
| within-subject, motor | **0.711** (p = 0.018, 7/10) | 0.711 (p = 0.026, 7/10) | 0.553 (p = 0.19, 5/10) |
| within-subject, all 64 | 0.749 (p = 0.003, 9/10) | 0.744 (p = 0.001, 10/10) | 0.542 (p = 0.10, 6/10) |
| within-subject, frontal | 0.571 (p = 0.030, 7/10) | 0.527 (p = 0.33) | – |
| cross-subject, motor | 0.573 (p = 0.24, 5/10) | 0.553 (p = 0.09, 7/10) | 0.511 (p = 0.40) |
| cross-subject, all 64 | 0.547 (p = 0.27) | 0.600 (p = 0.014, 7/10); with re-centering 0.633 (p = 0.005, 9/10) | 0.513 (p = 0.31) |
| cross-subject, frontal | 0.504 (p = 0.61) | 0.476 (p = 0.99) | – |

- **Within subjects, a motor-imagery signal is decodable without the eye shortcut:** CSP motor 0.711 vs EEGNet 0.553.
  CSP and TS each beat EEGNet in 9/10 subjects (sign test p = 0.021, all channels).
- **Across subjects, transfer is suggestive only.** The best result is TS with per-subject re-centering (RPA), all channels:
  0.633, 9/10 subjects. RPA uses the test subject's unlabelled trials. Its p does not survive Holm correction over the
  25 EOG-free cross-subject configs tried (Holm p = 0.12).
- **Frontal channels** are at chance cross-subject but still give 0.571 within-subject. So the motor result is "mostly
  sensorimotor": some non-motor information may survive the filter.
- **Conclusion.** The eye signal was the shortcut behind the pipeline's cross-subject accuracy. Motor-imagery ERD is real and
  decodable within subjects, where simple covariance models reach ~0.71. EEGNet is data-hungry and does not learn it from
  ~36 training trials per subject (within) or 6 training people (across).

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
   applied with no mixing between bands. It works like a per-band CSP or beamformer: it learns which electrodes to add or
   subtract to expose a source's band-power pattern, for example contrasting C3/C4 against Cz for mu ERD, or, as Task 3
   shows, contrasting Fp electrodes to pick up a vertical EOG dipole.
3. **One change that would stop the model using the eye signal:**
   - **Experiment:** make the cue identical in location for both classes, for example a central symbol or an auditory cue,
     so gaze no longer depends on the class. Alternatives:
     - counterbalance or randomise the cue position across blocks;
     - use a delayed cue and analyse only after the cue has disappeared.
   - **Analysis:** drop the first ~1 s after the cue **and** band-pass to 8–30 Hz, and/or remove ocular components (ICA or
     regression on Fp channels).
   - Task 3 showed each of these alone already removes the frontal shortcut (frontal → 0.517 and 0.483).

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

### Part 2 at a glance

- **The bet:** attend across 64 **electrode tokens** (log band-power, placed by 3-D scalp geometry) instead of 481
  timesteps. The attention matrix shrinks from 482² to 65².
- **Fixed split (test subjects 9–10, 3 seeds):** geom had the highest mean (0.574), but within one SE of vanilla (0.530) and
  EEGNet (0.526). No conclusive win, and seed variance was not lower.
- **All 10 subjects rotated (validation-selected checkpoints):** every neural model is at chance (geom 0.496,
  best EEGNet 0.527). Real vs shuffled geometry makes no difference (−0.004, p = 0.55), and removing attention changes
  nothing. The fixed-split lead and the rim attention map were noise from one split.
- **Lesson:** with 6–8 training subjects the bottleneck is the shift between people, not the attention axis. Only
  covariance re-centering (TS+RPA, which uses unlabelled test-subject trials) gets above chance (0.618).

### 1. Hypothesis (committed in `07411fc`, before the model code in `0c465b7`)

> Fists-vs-feet is encoded in *where* mu/beta power drops (lateral hand areas vs the medial vertex, an
> anterior–posterior/medial contrast), not in *when* individual samples occur; per-timestep attention compares
> random-phase amplitudes it can't generalise from 8 subjects, so attending across 64 electrode tokens that carry log
> band-power and know their 3-D scalp position should beat the vanilla transformer with lower seed variance, and its
> attention should concentrate on central (FC/C/CP, Cz) rather than frontal electrodes.

**The one bias:** move attention from the time axis to the space axis (electrode tokens), with each token located by
scalp geometry.

### 2. What I changed and why

- **The weakness:** per-timestep attention has no locality prior and compares single samples whose phase is random
  across trials. MI ERD is induced, not phase-locked, so on 8 subjects there is little to generalise.
- **Tokens:** each electrode is one token. A shared temporal filter bank (16 × 0.4 s kernels) is applied, squared,
  averaged over 0.5 s windows and logged. That gives log band-power, which is phase-invariant like ERD itself.
- **Attention axis:** attention runs across the 64 electrodes, because the task contrast is spatial and
  anterior–posterior/medial (hand areas vs foot area near Cz), so a left/right prior would not help.
- **Position:** an MLP on standardised xyz gives each token its scalp location.
- **Cost:** the attention matrix shrinks from 482² = 232,324 to 65² = 4,225 entries per head (≈ 55×).
- **Ablation:** `shuffled` gives the same xyz values to the wrong electrodes; `learned` replaces geometry with a learned
  per-electrode embedding.

### 3. What happened

Final-epoch test accuracy on subjects 9–10 (n = 90), seeds 0/1/2, **20 epochs for every model**
(the starter default is 40). The harness's "best" is the max over the test curve, i.e. selected on test (Part 1's bug B), so
final is the headline and peak is shown alongside. Chance is 0.5; p < .05 needs 0.600 (54/90); SE is 5.3 pts.

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
| CSP+LDA, 6 subjects | 0.509 (all seeds) | 0.622 | 0.433 | 0.988 | -0.013 (0.922) |
| CSP+LDA | 0.571 (all seeds) | 0.822 | 0.422 | 0.539 | -0.075 (0.250) |
| CSP+LDA, motor | 0.529 (all seeds) | 0.600 | 0.467 | 0.189 | -0.033 (0.344) |
| TS+LR | 0.542 (all seeds) | 0.622 | 0.489 | 0.127 | -0.046 (0.186) |
| TS+LR, motor | 0.536 (all seeds) | 0.578 | 0.511 | 0.182 | -0.039 (0.770) |
| TS+LR+RPA (transductive) | 0.618 (all seeds) | 0.689 | 0.522 | 0.006 | -0.121 (0.008) |
| TS+LR+RPA, motor (transductive) | 0.600 (all seeds) | 0.711 | 0.533 | 0.010 | -0.104 (0.027) |

- **Every neural model is at chance.** None reaches 0.540 in any seed (geom 0.496, best EEGNet 0.527).
- **The fixed-split lead was noise from the single subjects 9–10 split.** geom vs shuffled is −0.004 (p = 0.55) and vs
  EEGNet −0.031 (p = 0.49); the fixed-split ablation gap was 0.574 vs 0.474.
- **Attention adds nothing shown:** geom is +0.009 ahead of TokensNoAttn, which removes attention (p = 0.086).
- **Classical references:**
  - TS+LR+RPA pools 0.618 and beats geom (−0.121, p = 0.008). It is transductive, and its p vs 0.5 does not survive
    Holm over these 15 rows (0.088).
  - CSP+LDA pools 0.571, but its per-subject test is p = 0.54, driven by S7 (0.89) and S2 (0.76).
  - Classical rows fit on all 8 non-test subjects unless marked; neural models train on 6, so they are not like for like
    (CSP+LDA pools 0.509 with 6).

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
