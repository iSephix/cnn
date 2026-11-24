# RC12 — A trained portfolio of grid programs

| Public or supplementary split | RC11 | RC12 first attempt | RC12 two attempts | RC12 pool coverage |
|---|---:|---:|---:|---:|
| ARC-AGI-1 public | 102/400 | 139/400 | 154/400 | 171/400 |
| ARC-AGI-2 public | 0/120 | 1/120 | 1/120 | 2/120 |
| First supplementary sample | 7/100 | 7/100 | 10/100 | 16/100 |
| Second supplementary sample | 8/100 | 8/100 | 9/100 | 13/100 |
| Remaining disjoint tasks | 1/33 | 0/33 | 1/33 | 4/33 |

These local scores require every query in a task to match exactly. The selector gained 55 ARC-1 tasks and lost three. Pool coverage is a retrospective upper bound that chooses the correct grid per query using answer keys. The 50% target was 200 tasks. Even a perfect choice among the released candidates would require another 29 tasks with correct candidates. All evaluation splits had been examined during earlier releases. The selector was fitted on official ARC-1 training tasks only.

The normalized two-step native search alone solved 111/400 ARC-1 tasks in its first two guesses. On an exploratory three-second CPU budget, the experimental Hodel program pool contained correct grids on 23/400 ARC-1 tasks and added zero whole tasks to released pool coverage. Additional uncapped-node three-step trials used wall time limits and are excluded from release results.

The predictor pools RC11's two guesses, competing outputs from all six RC11 families, and candidates from Johan Sokrates Wind's Icecuber program synthesizer. Four native search settings vary color normalization, geometrical augmentation and search depth. The three-step search limits which nodes can be expanded at depth two. Two additional searches operate on edit masks and on latent grids assembled from repeated output tiles. A logistic model decides which two distinct outputs to return. It uses program agreement, rank, complexity, demonstration fit, and consistency of colors and shapes.

The C++ adapter reads demonstrations and one unlabeled query on stdin. It builds only the generic synthesis components from the pinned upstream revision. Dataset files, evaluation scripts and the task-indexed solver source in that repository are outside the build. Each search process has a 1.5 GiB address-space limit and a wall-clock timeout. A failed search contributes no guesses. The two-attempt baseline and candidate coverage are measured separately. Training features use the same previously trained RC2 and RC8 banks as inference, so fitting accuracy on their source tasks would be optimistic.

Run from the repository root with Linux, bash, git, patch, g++, Python, NumPy, SciPy, scikit-learn and Node.js. Run `bash rc12/predict.sh BLIND_TASK_DIR PREDICTIONS.json BUILD_CACHE` for unlabeled prediction, or `bash rc12/train.sh ARC1_DATA/training WORK_DIR` to refit the selector on the official training set. Run `bash rc12/reproduce.sh ARC1_DATA ARC2_DATA NEW_WORK_DIR` for the five evaluation splits using RC11's pinned datasets. Reproduction needs network access for the initial source clone. CPU speed can affect candidate availability near the timeout.

The experimental library miner reuses shared intermediate expressions in Michael Hodel's 400 training-task programs. It can be run separately with `python rc12/library.py BLIND_DIR ARC_DSL_CHECKOUT OUTPUT_DIR --seconds 3`. This experiment is excluded from the released model.

## Upstream sources

- Johan Sokrates Wind, *DSL solution to the ARC challenge* (2020), MIT. [Pinned source and license](https://github.com/top-quarks/ARC-solution/tree/9407072659de1270358c2ba34c527785214dd68b).
- Michael Hodel, *ARC-DSL*, MIT. [Pinned experimental source](https://github.com/michaelhodel/arc-dsl/tree/635de4902a5fb4e376f27333feaa396d3f5dfdcb).
