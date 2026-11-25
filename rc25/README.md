# RC25 — Structural programs and native trace repair

RC25 solves **255/400 (63.75%)** of the public ARC-AGI-1 development tasks within two attempts, up from 244/400. It keeps RC24's first attempt and adds 11 solves without losing any RC24 solves across the five measured splits.

| Split | RC24 two attempts | RC25 first attempt | RC25 two attempts |
|---|---:|---:|---:|
| ARC-AGI-1 public | 244/400 | 237/400 | **255/400** |
| ARC-AGI-2 public | 2/120 | 2/120 | **3/120** |
| Supplementary 1 | 9/100 | 7/100 | **11/100** |
| Supplementary 2 | 10/100 | 9/100 | 10/100 |
| Remaining supplementary | 0/33 | 0/33 | 0/33 |

The extra ARC-AGI-2 solve is a task shared with ARC-AGI-1. The two releases have the same query and expected answer for that task.

## What changed

- Structural search learns routes through uniquely colored markers, fills maximal free rectangles, shifts bands with periodic offsets, and recolors an object's pixels by local topology. Every candidate must reproduce every demonstration.
- Layout search unpacks mosaics, predicts the next frame of a grid sequence, and sorts rows or columns on latent editable lattices. The sequence search fits affine changes to object bounds or learns a color-constrained spread from transitions already present in the input.
- The widened native search now returns predicted grids for the demonstrations as well as the query. Residual search fits small geometric or context-dependent recoloring rules on the demonstration errors, then applies the rule to the query. The second attempt can use a unique exact native answer or agreement between a unique residual repair and the leading native trace. A native candidate cannot displace an existing second attempt also present in the trace. A unique symbolic answer takes precedence.

On ARC-AGI-1, the structural and layout programs contribute seven solves. Native traces contribute four more. Two of the native solves were already present in the RC24 candidate pool. The combined retrospective pool grows from **252 to 261 tasks**, with six correct answers still outside the two selected attempts and 139 tasks missing a complete correct candidate.

## Iterations and evaluation

On ARC-AGI-1, the symbolic-only selector reaches 251 solves. The two learned arbiter variants reach 251 with the first attempt pinned and 244 with both attempts ranked. The learned variants were fit on official ARC-AGI-1 training tasks using scores and predictions from the existing engine on those same tasks. The released selector uses fixed rules and achieved 255.

The geometry-first paint decoder, lattice projection with a general solver, and a pinned [ARC 2024 specialist bank](https://github.com/jiaxuan030331/ARC2024-solution/tree/9984bf5c7b3c7f47ddfad21dbacf4f26f8de2d00) were tested separately. They added no new ARC-AGI-1 candidate coverage and contribute only to the experiment report.

[Per-split scores and ablations](results/experiments.json) and the [ARC-AGI-1 coverage audit](results/arc1-diagnosis.json) use the same splits as RC24. Some RC25 operators were designed while examining public development demonstrations. The inherited detector bank was also developed partly on public ARC-AGI-1. These scores are public development measurements. Query labels were used by the scorer and coverage audit.

## Run

```sh
python3 rc25/test.py
bash rc25/predict.sh BLIND_DIR OUTPUT_JSON BUILD_CACHE
bash rc25/reproduce.sh ARC1_DATA ARC2_DATA WORK_DIR
bash rc25/train.sh ARC1_DATA/training WORK_DIR BUILD_CACHE
```

The optional fourth prediction argument selects `verified` for symbolic-only arbitration, `reserve` for learned second-attempt ranking with a symbolic override, or `learned` for learned ranking of both attempts. The default is `consensus`. A first build fetches pinned RC12 and RC23 dependencies. Native timeouts can affect results near the budget limit.
