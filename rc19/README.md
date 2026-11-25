# RC19 — Compose concept outputs and verify them together

RC19 solves **163/400 (40.75%)** within two attempts on public ARC-AGI-1, up from 160. It gains four tasks and loses one. The first-attempt score rises from 144 to 149.

| Local split | First attempt | Two attempts | Retrospective candidate coverage |
|---|---:|---:|---:|
| ARC-1 public | 149/400 | 163/400 | 178/400 |
| ARC-2 public | 1/120 | 1/120 | 2/120 |
| Supplementary 1 | 6/100 | 11/100 | 17/100 |
| Supplementary 2 | 9/100 | 10/100 | 15/100 |
| Remaining supplementary tasks | 0/33 | 0/33 | 5/33 |

The compositor uses a common mask representation for pixel fields, selected object shapes, and relationships between objects. It can intersect, subtract, or combine masks from different sources into an edit. Each edit must account for every observed cell change in its color while painting only cells consistent with the demonstrations. This composes up to three masks per color and up to four paint colors.

The panel solver learns a program for each half of an output and assembles their predictions, optionally restoring a separator. The packing solver extracts objects and learns their order, alignment, and spacing. An optional experiment in `planes.py` assembles independent color masks from the native synthesizer. That experiment found one ARC-1 candidate that was already solved and is outside the released portfolio.

The final verifier ranks all sources together. It checks geometric and color invariants from the demonstrations, such as changes in component count, foreground bounds, and symmetry. It also uses candidate disagreement, source evidence, and the legacy score. The final selector restores RC18's second-attempt rule for unanimous relation predictions after the first public comparison.

The gradient-boosting weight was chosen with five task-grouped folds on 400 official ARC-1 training tasks. The legacy ranker was fixed across folds and was previously trained on this same split, so the fold results measure internal model selection. The verifier selected 247 tasks within two attempts versus 237 for legacy scoring on the same candidates. An ablation that refits the verifier without the new generators selected 245 in the internal folds and scored 163 on public ARC-1, with 151 first attempts. The public two-attempt improvement is therefore achievable through ranking alone. The compositor increased coverage from four to five on the 33-task supplementary split by proposing a correct grid on fc10701f. That proposal ranked outside the two attempts.

These are development scores. Native search timeouts can affect candidates near the runtime budget. Run from the repository root with RC18's dependencies.

```sh
python3 rc19/test.py
bash rc19/predict.sh BLIND_DIR OUTPUT_JSON BUILD_CACHE
bash rc19/train.sh ARC1_DATA/training WORK_DIR
bash rc19/reproduce.sh ARC1_DATA ARC2_DATA WORK_DIR
```

Training writes `WORK_DIR/ranker.b85`. Prediction uses the checked-in model, which holds feature names and tree parameters in a compressed text format.
