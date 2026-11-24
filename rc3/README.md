# RC3 · Object selection and relational recoloring

Public benchmark progress on ARC-AGI-1 is 13 → 14 → 21 fully solved tasks across RC1, RC2 and RC3. RC3 earned 20/400 on the first attempt and 21/400 with two attempts. It retained every RC2 solve. On ARC-AGI-2 the result is 0/120.

## Acquisition

For each task, the learner tries alternative segmentations using single-color or multicolor connected components, four or eight neighbors, and zero or the most common input color as background. It fits selectors on area, shape repetition, palette size, and other descriptors, renders an object or bounding box and fits a color mapping. A second family learns categorical pixel rules on geometric, adjacency and object attributes, optionally conditioned on the cell's color. A copy rule can use a neighboring object's color or the nearest visible color in a direction.

All candidate predictions must explain the demonstrations. Complexity ranks the resulting candidates. RC2's fitted attempts have priority. The remaining slots take distinct predictions in rank order. The new extractors, rule families and complexity costs are hand-written. Task-specific selectors and readouts are inferred. RC3 currently fits those rules afresh on each task.

## Evidence

The seven gains comprise three object selections, three per-cell mappings, and one nearest-object color copy. See results/arc1-change.json for the exact queries and the source of each correct prediction. These are attributions of the emitted answers. A successful trace does not establish uniqueness of the inferred rule.

ARC-AGI-1 and ARC-AGI-2 public evaluation were used in prior releases. These are repeated public measures. No evaluation answer is passed to prediction. The scorer and retrospective coverage audit read query labels in separate processes. Six new result files include task hits, prediction hashes, gained/lost task IDs and new-family coverage. Timing here is unrestricted and there is no matched-compute efficiency claim.

## Run

Requires Python 3 with NumPy (tested with 2.3.5), Node.js, curl, unzip and bash. Run from the repository root.

- `python3 rc3/test.py` checks novel-color copying, extraction, and query-label rejection.
- `bash rc3/reproduce.sh` downloads pinned data, runs the full solver and compares score reports including prediction hashes and paired changes.
- `bash rc3/benchmark.sh PATH_TO_ARC1_DATA PATH_TO_ARC2_DATA OUTPUT_DIR` retains all predictions and rule traces locally.
- `python3 rc3/coverage.py PATH_TO_EVALUATION` checks whether any new-family candidate fits and executes on every query, then whether one such rule explains all hidden answers retrospectively.
