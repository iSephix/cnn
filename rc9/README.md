# RC9 — Composing spatial relationships

| Split | RC8, two attempts | RC9, first attempt | RC9, two attempts |
|---|---:|---:|---:|
| ARC-AGI-1 public evaluation | 61/400 | 68/400 | 71/400 |
| ARC-AGI-2 public evaluation | 0/120 | 0/120 | 0/120 |
| Supplementary ARC-AGI-2 training sample | 4/100 | 3/100 | 4/100 |

We count a task when every test grid matches exactly. All ten ARC-1 gains were additions. These are local scores on public data used during development. The supplementary sample uses the existing RC4 protocol.

## What earned the gains

1. **Pairs of stored rules**. From the frozen RC8 bank, extract threshold tests and decision paths. On a new task, search for two regions from cards with disjoint training origins. The output color depends on the input color and membership in those regions. Both regions must be needed to fit the demonstrations. Neither the single-region control nor the paired search added a public solve over RC8.
2. **Spatial relationships**. Construct regions from rows, columns, diagonals, bounding boxes, holes, and distances to colored cells. Search singles and pairs and fit their color mapping on demonstration grids. This stage added eight ARC-1 solves. Six used two relations. For example, task 319f2597 needed both the rows containing a zero and the columns containing a zero to fit the chosen rule.
3. **Panel reasoning**. Parse the grid using uniform separator lines or equal subdivisions. Infer how to summarize each panel into a cell or select a panel by an extremal statistic. This added two ARC-1 solves. Task 7039b2d7 used the most frequent color in each panel. Task c3202e5a selected the unique panel with the fewest distinct colors.

Each stage keeps prior nonidentity guesses first and fills available attempts with distinct predictions. Candidate selection uses demonstrations and unlabeled query inputs. The spatial relations and panel statistics are supplied primitives. The engine learns their combinations and output mappings within this vocabulary. We have yet to demonstrate an accuracy gain from composing two stored RC8 cards.

## Reproduction

From the repository root run `bash rc9/reproduce.sh` with Python 3, NumPy, SciPy, scikit-learn, Node.js, curl and unzip. It downloads the pinned datasets, rebuilds the previous solvers and checks score files including prediction hashes. For a cached RC8 run use `bash rc9/bench.sh BLIND_DIR RC8_PREDICTION.json LABELED_TASK_DIR OUTPUT_PREFIX`. Only the scorer receives the labeled directory. The matching RC8 trace file must be next to the prediction file. Run `bash rc9/transfer.sh ARC1_DATA ARC2_DATA OUT_DIR` for the supplementary comparison.

[Stage scores](results/stages.json) · [ARC-1 gain traces](results/arc1-change.json) · [Regression checks](test.py)
