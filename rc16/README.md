# RC16 — Learning which programs to reuse

This experiment mines programs from ARC-1 training tasks. The grammar combines up to two grid operations with a color mapping fitted to each new task. Programs can use literal colors or frequency ranks, select a largest or smallest object, and bind geometric directions to grid axes. Each source puzzle contributes up to twelve of its shortest exact programs. Programs that fit at least two source puzzles receive priority on new puzzles, ordered by their source frequency.

Every evaluation arm tests **384 program templates per task**. Arms share the grammar, interpreter, candidate generation, and two-attempt limit. The default searches in grammar order. Three controls shuffle programs of equal length with fixed seeds. Learning changes the search order. Program count controls the search budget, and individual programs can differ in runtime.

| Split | Default | Shuffled 0 | Shuffled 1 | Shuffled 2 | Learned |
|---|---:|---:|---:|---:|---:|
| First held-out ARC-1 training fold | 6/100 | 7/100 | 7/100 | 6/100 | 7/100 |
| Four-fold ARC-1 training cross-validation | 25/400 | 27/400 | 33/400 | 27/400 | 28/400 |
| Public ARC-1 evaluation | 5/400 | 5/400 | 5/400 | 5/400 | 5/400 |
| Public ARC-2 evaluation | 0/120 | 0/120 | 0/120 | 0/120 | 0/120 |

The first fold is part of the cross-validation total. Each fold uses a library mined from the other 300 tasks. The shipped library comes from the first fold's 300 source tasks. Public evaluation uses that same library. Each reported solve requires one of the first two distinct sets of predictions to match all queries. Full candidate coverage equaled the two-attempt result in these runs.

Compared with grammar order, learning gained three cross-validation solves and lost zero. One shuffled control solved all 28 tasks the learned ordering solved, plus another five. Adding the learned candidates to the full RC15 pool left its retrospective public coverage at 175/400 for ARC-1 and 2/120 for ARC-2. RC16 is available as an experimental candidate generator. RC15 remains the released predictor.

Run from the repository root with Python, NumPy, and SciPy. Use the canonical ARC dataset directories.

```sh
python3 rc16/learn.py train ARC1_DATA/training model.json
python3 rc16/learn.py evaluate ARC1_DATA/training model.json holdout.json --holdout
python3 rc16/learn.py cross-validate ARC1_DATA/training model.json cross-validation.json
python3 rc16/learn.py evaluate ARC1_DATA/evaluation model.json arc1.json
python3 rc16/learn.py evaluate ARC2_DATA/evaluation model.json arc2.json
python3 rc16/learn.py predict BLIND_DIR model.json CANDIDATE_DIR
```
