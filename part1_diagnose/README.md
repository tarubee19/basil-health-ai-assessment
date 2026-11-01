# Part 1: Diagnose & Trust

Hands-vs-feet motor-imagery decoding with EEGNet. The starter ran without errors but printed an inflated accuracy; the six
fixes are marked in the code as `# FIX (A)` … `# FIX (F)`. The write-up is in [`../ANSWERS.md`](../ANSWERS.md) (Part 1).

| File | What it does |
| --- | --- |
| `data.py` | Downloads the imagined runs (6/10/14) via MNE, filters, epochs, and returns trials, labels and subject ids |
| `train.py` | Trains EEGNet on held-out subjects (`--split fixed/cv/within`, `--channels all/frontal/motor`, `--crop`, `--fband`), appends to `results.jsonl` |
| `model.py` | EEGNet |
| `erd.py` | Mu/beta ERD at C3, Cz, C4 with a pre-cue baseline -> `erd_mu.png`, `erd_beta.png` |
| `eog_probe.py` | Cue-locked Fp1/Fp2 signal, hands vs feet -> `frontal_eog.png` (Task 3 evidence) |
| `classical.py`, `align.py` | CSP+LDA and tangent-space baselines on the same subject folds, optional alignment |
| `stats.py` | Subject-level statistics -> `stats_table.md`, `accuracy_grid.png` |
| `erd_plus.py` | ERD with confidence intervals, Laplacian, topomaps -> `erd_plus_*.png`, `erd_topomaps.png` |
| `results.jsonl` | One row per run, labelled by `tag` |

```bash
pip install -r requirements.txt
python train.py                                 # fixed split, seed 0 -> 0.522
python train.py --split cv --channels frontal   # Task 3
python erd.py --band mu                         # Task 2
python stats.py --plot                          # subject-level statistics
```

The first run downloads about 80 MB to `~/mne_data`. A CPU is enough.
