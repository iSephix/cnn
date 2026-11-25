# RC15 — Composing scene selections and learned transformations

RC15 solves **158/400 (39.5%)** on public ARC-AGI-1 within two attempts. It retained every RC14 solve and added 9110e3c5 and f5aa3634. The first-attempt score is 144/400.

The search selects intermediate grids from panels, object or color bounding boxes, frame interiors and compressed rows and columns. It can then rotate or reflect the intermediate grid and learn a replacement tile for each color. Selected panels and objects can also feed RC14's classification, panel combination and repair rules. A separate stencil search learns sets of offsets around a repeated anchor color. Programs fit the task demonstrations and produce candidate query grids. A logistic selector trained on the official ARC-1 training split ranks those candidates alongside the earlier portfolio.

| Local split | First attempt | Two attempts | Retrospective pool coverage |
|---|---:|---:|---:|
| ARC-1 public evaluation | 144/400 | 158/400 | 175/400 |
| ARC-2 public evaluation | 1/120 | 1/120 | 2/120 |
| Supplementary 1 | 6/100 | 11/100 | 17/100 |
| Supplementary 2 | 8/100 | 9/100 | 14/100 |
| Remaining supplementary tasks | 0/33 | 0/33 | 4/33 |

The composed search solved five public ARC-1 tasks within its first two candidates in isolation. The stencil search solved one. The full pool still covers the same 175 tasks as RC14. The net gain came from selection. All splits were previously examined during development.

Run from the repository root with the RC12 dependencies. Use `python3 rc15/test.py` for synthetic checks, `bash rc15/predict.sh BLIND_DIR OUTPUT_JSON BUILD_CACHE` for prediction, `bash rc15/train.sh ARC1_DATA/training WORK_DIR` to refit the selector, or `bash rc15/reproduce.sh ARC1_DATA ARC2_DATA WORK_DIR` to reproduce the five local scores.
