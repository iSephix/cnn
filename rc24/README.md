# RC24 — Held-out demonstration verification

RC24 reaches **244/400 (61.0%)** on public ARC-AGI-1, one more than RC23, with five more first-attempt solves. The supplementary second sample gains one solve. Full retrospective ARC-1 coverage remains 252 tasks.

| Split | RC23 two attempts | RC24 first attempt | RC24 two attempts | Single-orientation two attempts |
|---|---:|---:|---:|---:|
| ARC-1 public | 243/400 | 237/400 | 244/400 | 244/400 |
| ARC-2 public | 2/120 | 2/120 | 2/120 | 2/120 |
| Supplementary 1 | 9/100 | 7/100 | 9/100 | 9/100 |
| Supplementary 2 | 9/100 | 9/100 | 10/100 | 9/100 |
| Remaining supplementary | 0/33 | 0/33 | 0/33 | 0/33 |

The verifier repeatedly fits detectors with one demonstration excluded. Each resulting program must then reproduce every demonstration, including the excluded one. Candidates record how many distinct folds support the predicted grid. A second experiment permutes the palette of the input and output grids together so detectors can operate on colors ordered by their frequency. It reverses that permutation after prediction. Both procedures ignore query answers.

The selector learns how much to trust both kinds of evidence using official ARC-1 training tasks. The five-fold task-grouped training comparison scores 269/400 for RC23, for the full RC24 ranker, and for each source ablation. On public ARC-1, removing palette evidence yields 244 solves and 236 first-attempt solves. Removing held-out evidence yields 243 solves and 228 first-attempt solves. Neither new source expands the benchmark candidate pool. The new verifier's gains come from selection.

The optional single-orientation mode uses the RC23 selector and evaluates each external detector in one orientation rather than up to eight. It reaches the same 244 two-attempt solves on public ARC-1 with 226 first attempts and solves a different additional task. These are public development results. The pinned detector library was developed in part against public ARC-1 tasks.

```sh
python3 rc24/test.py
bash rc24/predict.sh BLIND_DIR OUTPUT_JSON BUILD_CACHE
bash rc24/predict.sh BLIND_DIR OUTPUT_JSON BUILD_CACHE single-orientation
bash rc24/train.sh ARC1_DATA/training WORK_DIR BUILD_CACHE
bash rc24/reproduce.sh ARC1_DATA ARC2_DATA WORK_DIR
```

The first run fetches the same pinned dependencies as RC23. Timing can affect candidates near detector and native timeouts. Training writes the replacement selector to WORK_DIR/model.json.
