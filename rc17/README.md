# RC17 — Spatial stencil transfer and count-based outputs

RC17 solves **159/400 (39.75%)** on public ARC-AGI-1 within two attempts. Its candidate pool contained correct outputs for **176/400** tasks, up from 175. The new solve, 2753e76c, encodes color counts as bars. RC17 retained all 158 RC15 solves.

| Local split | First attempt | Two attempts | Retrospective candidate coverage |
|---|---:|---:|---:|
| ARC-1 public | 144/400 | 159/400 | 176/400 |
| ARC-2 public | 1/120 | 1/120 | 2/120 |
| Supplementary 1 | 6/100 | 11/100 | 17/100 |
| Supplementary 2 | 8/100 | 9/100 | 14/100 |
| Remaining supplementary tasks | 0/33 | 0/33 | 4/33 |

The chart generator learns to render bars using pixel counts, component counts, or the number of rows or columns occupied by each color. It fits the order, orientation, alignment, and output width to the demonstrations. The motion generator independently fits translations and copies of color masks.

The stencil miner extracts offset sets from demonstrations in 300 ARC-1 training tasks. It stores geometry independently of the anchor and paint colors, which it binds on a new puzzle. The miner produced four stencils under a cap of 32. They yielded a correct candidate for one of the other 100 ARC-1 training tasks and one public ARC-1 task. A directional control with 32 stencils solved neither. A neighborhood control added after inspection reproduced both results with the same four geometries. This establishes that the extracted rules transfer across puzzles. It leaves an advantage over a hand-designed neighborhood library unproven. The stencil candidates added zero tasks to the full RC15 pool.

The RC17 ranker trains on official ARC-1 training tasks. The first public test of the refitted ranker gained one task and lost one. The final selector uses the RC17 ranker on tasks for which a new source contributes a candidate and uses RC15 otherwise. This gate was added after the initial public run. All published scores are local development results.

Run from the repository root with RC15's dependencies.

```sh
python3 rc17/test.py
python3 rc17/run.py train ARC1_DATA/training stencil-model.json --holdout
python3 rc17/run.py compare ARC1_DATA/training stencil-model.json transfer.json --holdout
bash rc17/predict.sh BLIND_DIR OUTPUT_JSON BUILD_CACHE
bash rc17/train.sh ARC1_DATA/training WORK_DIR
bash rc17/reproduce.sh ARC1_DATA ARC2_DATA WORK_DIR
```

Training the ranker writes `WORK_DIR/model.json`. Prediction uses the checked-in `rc17/ranker.json`.
