# RC21 — Route colors, bind objects, and interpret regions

RC21 solves **167/400 (41.75%)** on public ARC-AGI-1 within two attempts, up from 164. All previous solves survive. Three independent generators each add a task beyond RC20's candidate pool.

| Generator | ARC-1 tasks with a correct candidate from this source | Newly covered ARC-1 task |
|---|---:|---|
| Conditional color routing | 13 | 45bbe264 |
| Object-to-object color binding | 3 | 2a5f8217 |
| Region boundary rules | 4 | 62ab2642 |
| Symmetry-orbit repair | 5 | |
| Marker-guided template copying | 0 | |
| Object placement beside an anchor | 0 | |

Color routing learns a decision tree that directly optimizes for pointing to a source with the correct color. Candidate sources include local neighbors, rays, symmetry references, and scene measurements. Training can choose different sources for different cells. Each demonstration is then held out in turn to test transfer.

Object binding learns which objects interact and whether another object's color should be applied. It can select another object based on position, congruence, proximity, or use as a marker. Region rules inspect connected areas and learn whether to preserve their color, paint a fixed color, or borrow their boundary's dominant or least common color. Symmetry repair propagates observations around orbits induced by up to two geometric transforms. Marker and placement searches transfer a template or align an object against another object's edge.

The default predictor retains the old first attempt and admits a unanimous new candidate as the second. Routing additionally requires an exact prediction on every held-out demonstration with at most seven tree leaves. The final selector was chosen after an initial public development run. The routing gate admitted a correct candidate on 23 of 23 ARC-1 training tasks and nine of nine public ARC-1 tasks where it fired. This is a small sample.

| Local split | Default first attempt | Default two attempts | Candidate coverage |
|---|---:|---:|---:|
| ARC-1 public | 149/400 | 167/400 | 182/400 |
| ARC-2 public | 1/120 | 1/120 | 2/120 |
| Supplementary 1 | 6/100 | 11/100 | 17/100 |
| Supplementary 2 | 9/100 | 10/100 | 15/100 |
| Remaining supplementary tasks | 0/33 | 0/33 | 5/33 |

The optional refitted verifier scores 155 first attempts and 167 two-attempt solves on public ARC-1. Its supplementary scores are 13/100, 9/100, and 3/33. The default uses the conservative selector to retain all prior solves. Coverage is a retrospective upper bound using the answer keys after prediction. All results are local development scores.

Run from the repository root with RC20's dependencies.

```sh
python3 rc21/test.py
bash rc21/predict.sh BLIND_DIR OUTPUT_JSON BUILD_CACHE
bash rc21/expanded.sh BLIND_DIR OUTPUT_JSON BUILD_CACHE
bash rc21/train.sh ARC1_DATA/training WORK_DIR
bash rc21/reproduce.sh ARC1_DATA ARC2_DATA WORK_DIR
```

The training script writes the optional verifier to `WORK_DIR/ranker.b85`. The expanded predictor uses the checked-in model.
