# RC18 — Object relations and compressed-scene search

RC18 solves **160/400 (40.0%)** on the public ARC-AGI-1 split within two attempts, up from 159. It retained all RC17 solves across the five evaluated splits. The candidate pool added two previously uncovered ARC-1 tasks, raising coverage to 178/400.

| Local split | First attempt | Two attempts | Retrospective candidate coverage |
|---|---:|---:|---:|
| ARC-1 public | 144/400 | 160/400 | 178/400 |
| ARC-2 public | 1/120 | 1/120 | 2/120 |
| Supplementary 1 | 6/100 | 12/100 | 17/100 |
| Supplementary 2 | 8/100 | 9/100 | 15/100 |
| Remaining supplementary tasks | 0/33 | 0/33 | 4/33 |

- The relation search fits a shared rule for which object pairs interact. It can bridge gaps, draw lines between centers, or render the bounding box of a pair. Its bridge proposal solved 55059096.
- The object renderer learns which symbols to draw around objects selected by shape, size, position, or contacts. It fits up to three paint colors against the demonstrations.
- The compressed-scene search reruns native synthesis after cropping to foreground, collapsing repeated rows and columns, or removing uniform separator lines. The target grids stay in their original coordinates. This source found the correct candidate for 19bb5feb, outside the previous pool. The selector ranked it below the two attempts.

The ranker was fitted on official ARC-1 training examples. After the first public run, the selector was updated to use the second attempt for unanimous relation predictions and keep its first attempt. The top relation candidate was correct for all four training tasks on which the generator produced candidates for every query. That sample is small. These are public development scores.

The additional native searches run up to three compressed scenarios per query, each with a nine-second wall timeout. Runtime affects available candidates near the timeout. Run from the repository root with RC17's dependencies.

```sh
python3 rc18/test.py
bash rc18/predict.sh BLIND_DIR OUTPUT_JSON BUILD_CACHE
bash rc18/train.sh ARC1_DATA/training WORK_DIR
bash rc18/reproduce.sh ARC1_DATA ARC2_DATA WORK_DIR
```
