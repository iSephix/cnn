# RC11 — Intermediate grids and object-centered programs

RC11 solves 102 of 400 ARC-AGI-1 public tasks with two attempts. Three new tasks used a renderer trained in object-relative coordinates. The fourth, `fd096ab6`, used a shape found in the query itself as a template. The solver selected the object with the richest palette, aligned its footprint with other objects, and filled that footprint with each object's own color.

| Local benchmark | RC10, two attempts | RC11, one attempt | RC11, two attempts |
|---|---:|---:|---:|
| ARC-AGI-1 public | 98/400 | 88/400 | 102/400 |
| ARC-AGI-2 public | 0/120 | 0/120 | 0/120 |
| First supplementary ARC-2 training sample | 7/100 | 4/100 | 7/100 |
| Second supplementary ARC-2 training sample | 7/100 | 6/100 | 8/100 |
| Remaining disjoint ARC-2 training tasks, evaluated after freeze | 1/33 | 1/33 | 1/33 |

A task is solved when every test grid matches exactly within the attempt budget. All four new ARC-1 solves came on the second attempt. These are local scores on public data. Both 100-task samples had already been evaluated by RC10. After freezing RC11, we compared both solvers on all 33 remaining ARC-2 training tasks after excluding earlier splits by task ID and content hash. Both solved the same task.

## What composes

- **Inspect and paint.** Search for an intermediate panel, object mask, ray field or crop and learn a color or tile lookup over it.
- **Partition and render.** Assign cells to the nearest object, measure position and color in that object's coordinate system, then fit a decision tree on the demonstrations.
- **Learn an alphabet of tiles.** Factor the demonstrated outputs into repeated motifs. Fit an encoder that predicts the motif indices, then expand the predicted index grid through the learned dictionary. This solved `87ab05b8` on the second supplementary sample.
- **Complete from an exemplar.** Extract a reference shape from the current input, match partial objects against it, then draw the inferred completions.
- **Simulate motion or replay object-relative edits.** Search for shared motion and stopping rules, or learn a dictionary of edits expressed relative to individual objects. These two final stages added no tasks on the public evaluation splits.

RC11 searches bounded program families. Demonstrations select operations and fit parameters. New proposals may fill empty attempts or replace a second RC10 tree guess. For that replacement, lookups require supporting keys across demonstrations, trees require exact held-out demonstration predictions, and the exemplar and motion programs require an exact shared demonstration fit. The motif dictionary and output-size hypotheses are inferred using all demonstrations.

## Use and verify

Run `bash rc11/predict.sh BLIND_TASK_DIRECTORY predictions.json` with a directory of unlabeled ARC tasks. The script writes the two guesses per query and a trace alongside them. It uses the CPU dependencies described in RC10. Run `python3 rc11/test.py` for checks of intermediate grids, motif decoding, exemplar completion and object-relative edits.

Run `bash rc11/reproduce.sh` to download pinned data and reproduce all five score files from the stored training banks. With cached RC10 predictions and their traces, run `bash rc11/bench.sh BLIND_DIR RC10_PREDICTIONS LABELED_DIR OUTPUT_PREFIX`. The labeled directory is used only for scoring.

The retrospective candidate audit hashes the complete vector of test predictions for each program before comparison with the answer keys. This checks whether any single generated program solved all queries for a task. An RC11 program produced the complete correct answer for 14 ARC-1 tasks missed by the final two guesses. Adding those retrospectively to the 102 actual solves gives 116/400 candidate coverage. No RC11 family generated a complete correct answer on ARC-2 public. Use `python3 rc11/probe.py emit BLIND_DIR pool.json` followed by `python3 rc11/probe.py score LABELED_DIR pool.json FINAL_SCORE.json`.

[Task scores, gain traces and candidate coverage](results/)
