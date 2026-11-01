# Part 2 — resume notes (stopped at the time limit, 15:23)

**Done (committed):**
- hypothesis `07411fc` (before the model code `0c465b7`);
- YourModel = electrode tokens + xyz PE;
- results for YourModel geom/shuffled/learned, EEGNet (supplied) and PatchTransformer, seeds 0–2, 20 epochs, in
  `results.jsonl`;
- `figure.png`, `figure_tsne.png`;
- README write-up with the results table.

**Not done:**
1. **VanillaTransformer seeds 0–2 and EEGNet-P1 seeds 0–2:** not run, because of CPU contention with the Part 1 job.
   `python -u main.py --epochs 20 --only vanilla eegnet_p1` (appends to `results.jsonl`), then refresh the README table.
2. **`figure.png` layout:** the committed file was drawn by the running process before the layout fix in `5688eb1`, so
   the (b)/(c) titles overlap. Redraw with `python -u main.py --epochs 20 --figure-only` (~35 s on a free CPU; it
   retrains geom seed 0 and prints its final/peak, which should equal results.jsonl: final 0.478, peak 0.567).
3. **Reproducibility check:** not done. Rerun everything into a fresh log and compare with `results.jsonl`:
   `python -u main.py --epochs 20 --log results_repro.jsonl`.
4. **Starter-file check:** `git diff 86e1619 -- part2_transformer/models.py part2_transformer/data.py part2_transformer/train.py`
   should be empty.
5. **Root `ANSWERS.md` Part 2 merge:** waiting for the Part 1 session to confirm it has finished. Then copy the README
   write-up and commit `docs: ANSWERS.md Part 2`.
6. **Push:** the repo has no git remote configured. Add one, then `git push`.
