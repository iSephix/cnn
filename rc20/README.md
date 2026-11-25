# RC20 — Learn object-specific actions

The default predictor solves **164/400 (41.0%)** on public ARC-AGI-1 within two attempts. It adds e74e1818, a task outside the previous candidate pool, and retains all RC19 solves. The target is **400/400**.

| Local split | RC20 first attempt | RC20 two attempts | Expanded candidate coverage |
|---|---:|---:|---:|
| ARC-1 public | 149/400 | 164/400 | 179/400 |
| ARC-2 public | 1/120 | 1/120 | 2/120 |
| Supplementary 1 | 6/100 | 11/100 | 17/100 |
| Supplementary 2 | 9/100 | 10/100 | 16/100 |
| Remaining supplementary tasks | 0/33 | 0/33 | 5/33 |

The object solver finds local actions that reproduce each demonstrated edit. It learns when to apply an action from object attributes including size rank, position, color, and shape multiplicity. The same policy is applied to the query. The default predictor retains the RC19 first attempt and offers a unique object-rule answer as the second attempt when every query has exactly one object candidate. This selection policy was adopted after the initial public development comparison.

An opt-in expanded predictor learns dimensions from input sizes, object extents, connected components, and other measurements. It searches up to three inferred output shapes per query with a seven-second native timeout on each search. It also learns cell rules from resampled input grids, including mirrored, tiled, and scaled views. The expanded verifier was fitted on the 400 official ARC-1 training tasks. Its two-attempt result was 162/400 on public ARC-1 and 2/33 on the remaining supplementary tasks. The extra dimension and projection searches added zero ARC-1 tasks beyond the 179 covered by the default candidate sources. The audit JSON files measure coverage using the expanded pool.

The default ARC-1 failure audit used answer keys only after prediction. Of the 236 unsolved tasks, 15 had a correct grid somewhere in the pool, 212 had a candidate with the right dimensions for each query but lacked correct contents for at least one query, and nine lacked the right dimensions for at least one query. All scores concern local development splits.

Run from the repo root using RC19's dependencies.

```sh
python3 rc20/test.py
bash rc20/predict.sh BLIND_DIR OUTPUT_JSON BUILD_CACHE
bash rc20/expanded.sh BLIND_DIR OUTPUT_JSON BUILD_CACHE
bash rc20/train.sh ARC1_DATA/training WORK_DIR
bash rc20/reproduce.sh ARC1_DATA ARC2_DATA WORK_DIR
```

The training script writes the experimental verifier to `WORK_DIR/ranker.b85`. The expanded predictor uses the checked-in model.
