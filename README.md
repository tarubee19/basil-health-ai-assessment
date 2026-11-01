# EEG × Deep Learning: live exam submission

Motor-imagery decoding (imagined **both fists vs both feet**) on PhysioNet EEGBCI, subjects 1–10, 64 channels, 160 Hz.
The full write-up, with every number, is in **[ANSWERS.md](ANSWERS.md)**.

| Part | Question | Folder |
|---|---|---|
| **1. Diagnose & Trust** | The starter pipeline runs cleanly but prints a wrong accuracy. Find the bugs, read the ERD, and work out what the model really uses. | [`part1_diagnose/`](part1_diagnose) |
| **2. Build & Defend** | Can self-attention beat a conv net on tiny EEG data if we give it one EEG-specific inductive bias? | [`part2_transformer/`](part2_transformer) |

## Main findings

**Part 1**
- **Six defects fixed, one commit each:** trial-level split, test-set checkpoint selection, normalisation leak, executed
  instead of imagined runs, post-cue ERD baseline, and trials dropped at run ends. Accuracy fell from 0.667 (inflated) to
  0.522 on the fixed split, and 0.566 with 5-fold subject-level cross-validation.
- **ERD:** hand imagery desynchronises mu over C3/C4. The foot effect at Cz is weak, which fits the deep mesial foot area.
- **The model was using the cue:** the fists target is at the top of the screen and the feet target at the bottom. 8 frontal
  electrodes alone (0.617) beat all 64 (0.566). The signal sits in the first second and below 8 Hz, and its polarity flips
  with class in 10/10 subjects: a vertical eye movement (EOG).
- **Follow-up, without the eye signal (8–30 Hz, 1–4 s):** motor imagery is decodable within subjects (CSP 0.711 on motor
  channels vs EEGNet 0.553). Across subjects it is only suggestive.

**Part 2**
- **The bet:** attend across 64 **electrode tokens** (log band-power, located by 3-D scalp geometry) instead of 481
  timesteps. The attention matrix shrinks from 482² to 65².
- **Result:** no conclusive win over EEGNet or the vanilla transformer. Best and worst runs across seeds, the ablations, and
  what failed are reported in ANSWERS.md.

## Reproduce

```bash
pip install -r part1_diagnose/requirements.txt    # or: conda env create -f environment.yml

cd part1_diagnose
python train.py                                   # fixed pipeline, seed 0 -> 0.522
python train.py --split cv --seed 0               # 5-fold subject-level CV
python train.py --split cv --channels frontal     # Task 3: frontal-only
python erd.py --band mu                           # Task 2 -> erd_mu.png
python classical.py --help                        # follow-up: CSP / tangent-space baselines
python stats.py --plot                            # follow-up: subject-level statistics -> stats_table.md, accuracy_grid.png

cd ../part2_transformer
python -u main.py --epochs 20 --seeds 0 1 2       # reported numbers use 20 epochs (default is 40)
```

The data (~80 MB) downloads automatically through MNE on the first run. Everything runs on a laptop CPU, and training is
deterministic on CPU, so seeds reproduce exactly.

## Repository map

```
ANSWERS.md                  write-up for both parts
part1_diagnose/
  data.py, erd.py, train.py, model.py   fixed pipeline (each fix commented "# FIX (A)" ... "# FIX (F)")
  eog_probe.py                          frontal EOG evidence -> frontal_eog.png
  classical.py, align.py, stats.py      follow-up baselines and statistics
  erd_plus.py                           follow-up ERD (Laplacian, topomaps)
  results.jsonl                         one row per run, labelled by tag
  erd_mu.png, erd_beta.png, frontal_eog.png, accuracy_grid.png, erd_plus_*.png, erd_topomaps.png
part2_transformer/
  main.py                    YourModel (electrode tokens + scalp geometry) and the comparison harness
  models.py, data.py, train.py           supplied, unchanged
  results.jsonl, figure.png, figure_tsne.png
```

The commit history is kept unsquashed: one commit per bug fix, and the Part 2 hypothesis was committed before the model code.
