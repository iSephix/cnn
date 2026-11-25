# RC23 — Verified detector portfolio

RC23 solves **243/400 (60.75%)** on public ARC-AGI-1 in two attempts, up from RC21's 167/400. The first-attempt score rose from 149 to 232. RC23 gains 77 tasks and loses one. Retrospective coverage across both candidate pools reaches 252/400.

| Split | RC21 two attempts | RC23 first attempt | RC23 two attempts |
|---|---:|---:|---:|
| ARC-1 public evaluation | 167/400 | 232/400 | 243/400 |
| ARC-2 public evaluation | 1/120 | 2/120 | 2/120 |
| Supplementary 1 | 11/100 | 6/100 | 9/100 |
| Supplementary 2 | 10/100 | 9/100 | 9/100 |
| Remaining supplementary tasks | 0/33 | 0/33 | 0/33 |

The portfolio adds Tanmay Bisen's [321 MIT-licensed symbolic detectors](https://github.com/tanmaybisen31/arc-agi-solver/tree/e151937e34c8b34f953833a0dab75797fc737ba4) at a pinned revision. The adapter independently verifies every returned transformation against every visible demonstration. It tests orientations, maps predictions back to the original coordinates, and pools evidence by detector and orientation. A logistic selector trained on official ARC-1 training labels ranks distinct candidates from the new bank and RC21 using agreement and grid consistency. The detector bank alone solves 206 public ARC-1 tasks within two attempts.

**Evaluation exposure.** Upstream developed detectors against the public ARC-1 evaluation set, and some source files explicitly describe individual public puzzles. The adapter passes grids and demonstrations to the detectors. Evaluation answers are loaded separately by the scorer. The resulting 60.75% measures performance on a public development set. A fixed selector with the full bank in the original orientation scores 243/400. Excluding the 15 modules whose source mentions IDs from that split reduces that same comparison to 189/400. This ablation measures sensitivity to those modules. The upstream library also used the public split more broadly. The supplementary regressions and small ARC-2 gain show limited measured transfer.

The input-view and output-factorization experiment in views.py independently solved 32 already covered public ARC-1 tasks. It produced zero additional correct grids on the evaluated splits and is outside the released predictor. The checked-in model is fitted on 400 official ARC-1 training tasks. The calibration report shows five-fold task-grouped model selection using the fixed RC21 predictor.

Run from the repository root with RC21's dependencies. The build caches the pinned detector checkout. Python timeouts and native search timeouts can affect borderline candidates.

```sh
python3 rc23/test.py
bash rc23/predict.sh BLIND_DIR OUTPUT_JSON BUILD_CACHE
bash rc23/train.sh ARC1_DATA/training WORK_DIR BUILD_CACHE
bash rc23/reproduce.sh ARC1_DATA ARC2_DATA WORK_DIR
```
