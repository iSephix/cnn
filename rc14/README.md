# RC14 — Scene concepts

RC14 searches for three families of rules across a task's demonstrations.

- Infer a repeating lattice, staggered tile, or mirror axis from visible cells. Fill a damaged field or return the reconstructed patch.
- Align up to eight panels and learn an output color for each observed combination of panel colors or occupancy flags.
- Group connected objects by shape, optionally allowing rotations and reflections. Learn which objects to retain, erase, or recolor from shape, size, or the number of congruent copies.

When these rules agree on a single candidate, RC14 promotes it to the first attempt. With multiple candidates it considers the simplest for the second attempt. Rules must fit all demonstrations. The selector uses no task IDs.

| Local split | First attempt | Two attempts | Pool coverage |
|---|---:|---:|---:|
| ARC-1 public | 144/400 | 156/400 | 175/400 |
| ARC-2 public | 1/120 | 1/120 | 2/120 |
| Supplementary 1 | 6/100 | 11/100 | 17/100 |
| Supplementary 2 | 8/100 | 9/100 | 14/100 |
| Supplementary remainder | 0/33 | 0/33 | 4/33 |

The new families solved 23 ARC-1 public tasks in isolation. RC13 already solved all 23 within two attempts. Promotion added one first-attempt solve. These previously examined public splits measure development progress. Pool coverage uses the answers retrospectively.

Run `python3 rc14/test.py` for synthetic checks, `bash rc14/predict.sh BLIND_TASK_DIR OUTPUT_JSON BUILD_CACHE` for prediction, or `bash rc14/reproduce.sh ARC1_DATA ARC2_DATA WORK_DIR` to reproduce the public scores with the RC13 dependencies.
