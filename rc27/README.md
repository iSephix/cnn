# RC27 — Relational actions and geometric constraints

RC27 solves **265/400 (66.25%)** public ARC-AGI-1 tasks within two attempts, up from RC25's 255/400. First-attempt solves rise from 237 to **247**. Ten additional tasks are solved with zero regressions across the five measured benchmark splits.

| Split | RC25 first | RC25 two | RC27 first | RC27 two |
|---|---:|---:|---:|---:|
| ARC-AGI-1 public development | 237/400 | 255/400 | **247/400** | **265/400** |
| ARC-AGI-2 public development | 2/120 | 3/120 | 2/120 | 3/120 |
| Supplementary 1 | 7/100 | 11/100 | 7/100 | 11/100 |
| Supplementary 2 | 9/100 | 10/100 | 9/100 | 10/100 |
| Remaining supplementary | 0/33 | 0/33 | 0/33 | 0/33 |
| RC26 acquisition holdout | 65/100 | 70/100 | 65/100 | 70/100 |

The supplementary scores and acquisition holdout show no improvement. Representative public ARC-AGI-1 demonstrations informed the new operators. These measurements quantify progress on public development tasks. The inherited external detector bank may already encode knowledge of official training and public development tasks.

## New executable concepts

| Concept | Operation | New ARC-1 solve |
|---|---|---|
| Geometric template transfer | Match an enlarged or rotated monochrome silhouette to a colored exemplar and transfer its paint | 103eff5b |
| Connected label broadcast | Carry a unique marker color through a connected structure and paint its morphological interior | 09c534e7 |
| Object-relative staircase | Place each object relative to the preceding object's extent | 03560426 |
| Partial panel correspondence | Find a scaled exemplar inside another panel's foreground and paint the corresponding subset | 15113be4 |
| Scene-relative topology | Fit paint actions to intrinsic neighborhood features while allowing the foreground color to change | 15663ba9 |
| Conserved-piece exact cover | Tile enclosed cavities with available rectangular pieces, conserving their area and color | 184a9768 |
| Spatial legend correspondence | Match a legend's relative marker positions to separate scene objects | 1da012fc |
| Projected marker coordinates | Use external row and column pins to define a rectangle inside an object | 1c02dbbe |
| Object grid reconstruction | Group spatially separated icons into rows and assemble a compact two-dimensional layout | 1990f7a8 |
| Cavity insertion | Recognize an object's ability to fill an opening, including bodies that extend beyond the opening | 1acc24af |

Each operator searches shared parameters across demonstrations and then executes the fitted program on unlabeled queries. Per-scene objects and geometric relations are recomputed from the query. Inference uses the demonstration grids and query inputs.

The exact-cover solver is bounded by 40,000 search nodes and 16 distinct solutions. Task parameters include background interpretation, piece rotation, excluded distractor colors, and a solution rank shared across demonstrations.

Scene-relative topology reuses RC25's intrinsic neighborhood fields. It learns either preservation or a target color for a combination of features. Local decision trees provide another paint-rule generator. Both paint generators must reproduce every held-out demonstration when fitted to the others before their predictions are eligible for release.

## Arbitration and ablations

RC27 combines the new predictions with RC25. A complete candidate specifies every query in the task. When all eligible candidates agree on one complete prediction, the selector promotes it to the first attempt and carries RC25's first attempt as the second. Conflicting predictions retain RC25's pair.

| ARC-AGI-1 selection | First attempt | Two attempts |
|---|---:|---:|
| RC25 | 237 | 255 |
| New verified prediction in second attempt | 237 | 265 |
| Released promotion policy | **247** | **265** |
| Admit paint rules without held-out verification | 227 | 261 |

Retrospective candidate coverage rises from **261 to 271 tasks**. Six tasks have a correct candidate outside the selected pair. Another 129 tasks lack a complete correct candidate in the audited pool.

The periodic reconstruction and empty-space normalization operators add no ARC-1 coverage beyond RC25. Per-family reports include their results alongside the successful operators.

## Reproduce

Run from the repository root with Python, NumPy, SciPy, and scikit-learn. The first prediction run fetches RC25's pinned dependencies.

```sh
python3 rc27/test.py
bash rc27/predict.sh BLIND_DIR OUTPUT_JSON BUILD_CACHE
bash rc27/reproduce.sh ARC1_DATA ARC2_DATA WORK_DIR
```

To inspect candidates independently of the full predictor, select one of the six generator families.

```sh
python3 rc27/emit.py BLIND_DIR CACHE_DIR --family constraints
python3 rc27/emit.py BLIND_DIR CACHE_DIR --family topological
python3 rc27/arbitrate.py BLIND_DIR BASELINE_JSON OUTPUT_JSON CACHE_DIR --promote
```

Generator families are `relational`, `local`, `patterns`, `topological`, `constraints`, and `layout`. Candidate records include fitted parameters and complete query predictions. Paint records include held-out demonstration counts.

The [per-task audit](results/arc1-audit.json), [coverage diagnosis](results/arc1-diagnosis.json), and [scores](results) were computed after inference. The predictor rejects query answers. Native search and detector timeouts inherited from RC25 can affect reproduction near their time limits.
