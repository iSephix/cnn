# RC13 — Learning output structure

| Local split | RC12 two attempts | RC13 first attempt | RC13 two attempts | RC13 candidate coverage |
|---|---:|---:|---:|---:|
| ARC-AGI-1 public evaluation | 154/400 | 143/400 | 156/400 | 175/400 |
| ARC-AGI-2 public evaluation | 1/120 | 1/120 | 1/120 | 2/120 |
| First supplementary sample | 10/100 | 6/100 | 11/100 | 17/100 |
| Second supplementary sample | 9/100 | 8/100 | 9/100 | 14/100 |
| Remaining supplementary tasks | 1/33 | 0/33 | 0/33 | 4/33 |

Two searches propose grids to the RC12 selector. The first infers up to three lossless factorizations shared by all demonstrated outputs. It solves for a smaller grid and reconstructs the output by reflecting, repeating or scaling each axis. A separate rotated search covers orientation-sensitive programs. The second search learns a Boolean expression for each output color from input-derived masks including outlines, spans, diagonals, holes, offsets and distance rings. It tests both literal colors and colors ranked by frequency. Both searches run on stripped tasks. The selector is fitted on the 400 official ARC-1 training tasks.

On ARC-1 the release gained three tasks (817e6c09, a406ac07 and d47aa2ff) and lost one (e88171ec) relative to RC12. All test inputs in a task must be solved within two guesses. Candidate coverage counts tasks for which a scorer with access to the answers could select a correct candidate for every query. The audit files use the RC11 proposal pair as their internal prior. The repository tracks local development results on previously examined public splits. An independent leaderboard score remains unmeasured.

Run from the repository root using the [RC12 dependencies](../rc12/README.md).

- Test the new searches with `python3 rc13/test.py`.
- Predict with `bash rc13/predict.sh BLIND_TASK_DIR PREDICTIONS.json BUILD_CACHE`.
- Refit with `bash rc13/train.sh ARC1_DATA/training WORK_DIR`.
- Reproduce the five scores with `bash rc13/reproduce.sh ARC1_DATA ARC2_DATA NEW_WORK_DIR`. The first build fetches pinned upstream sources. Timeouts can affect reproducibility on slower machines.
