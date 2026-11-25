# RC22 — Explain transformations through object correspondence

RC22 is an experimental candidate generator. It adds **no new public ARC-AGI-1 solves** over RC21. The released predictor remains RC21 at **167/400** within two attempts, with correct candidates for 182 tasks.

The experiment fits executable explanations of what happened to an object between a demonstration's input and output. It matches shapes across translations, rotations, and reflections. It can identify matching pixels in a merged component, take a color from its source or anchor, and infer whether to move the original or copy it. Placement can be absolute or defined by another object's edges. Separate programs may account for different parts of the transformation, provided their combined writes reproduce the demonstrated grids exactly.

The extraction experiment handles smaller outputs. It learns which object to crop by comparing shape repetition, position, enclosure, distance to a marker, and other attributes. It can fit an orientation and a shared color mapping.

| Dataset | Tasks | Correct candidate from correspondence | Correct candidate from extraction | Additional over RC21 |
|---|---:|---:|---:|---:|
| ARC-1 training split | 400 | 11 | 2 | |
| ARC-1 public evaluation | 400 | 1 | 0 | 0 |
| ARC-2 public evaluation | 120 | 0 | 0 | 0 |
| Supplementary 1 | 100 | 0 | 0 | 0 |
| Supplementary 2 | 100 | 0 | 0 | 0 |
| Remaining supplementary tasks | 33 | 0 | 0 | 0 |

The only public ARC-1 match was ef26cbf6, which RC21 already solves. The placement solver produced candidates for 13 training tasks and one public ARC-1 task. Extraction produced candidates for four training tasks and zero public ARC-1 tasks. The low activation rate points to a narrow hypothesis space. Matching only rigid shapes or uniformly recolored shapes leaves many scene transformations outside it.

The candidate generator reads demonstrations and unlabeled queries. Its optional trace records the chosen object selector, operation, orientation, anchor and placement rule, and the observed object origins before and after each demonstrated edit. A trace explains an exact fit to demonstrations. Multiple programs can fit the same observations, and the trace shows the least-cost recorded explanation per distinct set of outputs.

Run from the repository root with Python, NumPy, and SciPy.

```sh
python3 rc22/test.py
python3 rc22/emit.py BLIND_DIR CANDIDATE_DIR --traces TRACE_DIR
python3 rc22/emit.py BLIND_DIR EXTRACTION_DIR --family extraction --traces EXTRACTION_TRACE_DIR
python3 rc22/evaluate.py LABELED_DIR CANDIDATE_DIR EXTRACTION_DIR --baseline RC21_PREDICTIONS.json
```

The evaluator reads query answers only after predictions have been generated. A task is covered if each expected output appears among the candidates for that query.
