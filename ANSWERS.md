# ANSWERS — EEG × Deep Learning live exam

All numbers below come from `part1_diagnose/results.jsonl` (one row per run, labelled by `tag`) or from a quoted command
output. Training is deterministic on CPU: re-running seed 0 reproduced the untouched number exactly (0.667, best epoch 17).

---

## Part 1 — Task 1: the six defects

Line numbers refer to the untouched starter (`git show starter:part1_diagnose/<file>`).

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
