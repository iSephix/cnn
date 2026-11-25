# RC26 — Compositional search and concept acquisition experiment

RC26 adds executable grid programs, backward constraints from demonstration outputs, and a library of learned program fragments. The experiment finds useful compositions, but it has **not improved the released ARC-AGI scores**. RC25 scores 255/400 on public ARC-AGI-1.

| Acquisition holdout, 100 tasks | First candidate | First two candidates | Complete candidate coverage |
|---|---:|---:|---:|
| Forward search | 7 | 7 | 7 |
| Search with backward goals | 7 | 8 | 8 |
| Backward goals and learned fragments | 7 | 7 | 7 |
| Scene program followed by verified detector | 10 | 10 | 10 |

The holdout contains the first 100 filenames of official ARC-AGI-1 training. Fragment acquisition uses the other 300 tasks and reads their demonstrations. Every holdout prediction is fitted to that task's demonstrations with its queries unlabeled. This controls exposure during RC26 acquisition. The inherited external detector library may already encode knowledge of official training tasks.

## Executable concepts

Programs share a sequence representation with named, parameterized operations. They reuse RC15's scene, color, panel, and object selections and add grid cropping, removal of empty rows or columns, geometric transforms, panel combination, scaling, tiling, and mirrored extension. A final color map is fitted across all demonstrations. Search states are deduplicated across demonstrations and queries together.

The backward search reverses known output rendering operations into intermediate goals. A color-renaming invariant indexes those goals. Only a consistent color map can connect a forward state to a goal. This constrains intermediate grids using output structure while preserving exact demonstration fits.

Acquisition preserves alternative program explanations even when their query predictions coincide. It simplifies adjacent geometric transforms, extracts recurring sequences, and counts support once per source task. The saved library has 16 fragments, each supported by at least two acquisition tasks. It is a library within this fixed grammar. Concept generation beyond that grammar is still open.

The learned search uses fragment support to order expansions and executes fragments as reusable operations. Prefix caches share intermediate results. Fixed and learned searches each allow at most 768 distinct generated states, a beam of 32, and three expansion rounds. A fragment can execute several atomic operations. These budgets constrain states, and do not equalize runtime or atomic work.

The detector bridge considers at most eight distinct intermediate scene programs per task. It fits the pinned RC23 detector bank on each intermediate representation and requires exact demonstration reproduction. A single detector must produce an answer for every query. This uses a separate budget from the fixed and learned searches.

## Results and limits

On public ARC-AGI-1, fixed and learned search each cover nine tasks. The bridge covers 29. Every covered task was already in RC25's retrospective candidate pool. A policy that replaces the second attempt only for a unique complete prediction scores 255/400 with either search arm and 253/400 with the detector bridge.

The frequency-based library failed to improve the acquisition holdout. Its most frequent fragments select a small object and apply a geometric transform. The current grammar and acquisition objective provide limited evidence for abstractions that transfer to harder tasks.

The experiment supplies a reproducible foundation for further work on concept acquisition. The next measurable capability would be discovering a reusable procedure that increases coverage on tasks excluded from acquisition. Public development scores and acquisition holdout results are recorded in [results](results).

## Run

Use Python with NumPy and SciPy for the compositional engine. The detector bridge uses RC23's pinned dependency.

```sh
python3 rc26/test.py
python3 rc26/search.py predict BLIND_DIR CANDIDATE_DIR
python3 rc26/search.py predict BLIND_DIR CANDIDATE_DIR --model rc26/model.json
python3 rc26/search.py predict BLIND_DIR CANDIDATE_DIR --forward-only
python3 rc26/search.py train ACQUISITION_TASK_DIR MODEL_JSON
python3 rc26/bridge.py BLIND_DIR CANDIDATE_DIR rc26/model.json DETECTOR_CACHE
python3 rc26/predict.py BLIND_DIR CANDIDATE_DIR BASELINE_JSON OUTPUT_JSON
bash rc26/reproduce.sh ARC1_DATA ARC2_DATA WORK_DIR
```

The optional experimental wrapper runs RC25 and merges learned-search predictions using the unique-prediction policy.

```sh
bash rc26/predict.sh BLIND_DIR OUTPUT_JSON BUILD_CACHE
```

The evaluator reads answer keys after candidate generation. The predictor rejects labeled queries. Detector timeouts can affect reproduction near the time budget.
