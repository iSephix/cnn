# RC10 — Learning where to read and draw

RC10 learns spatial programs from each task's demonstrations. Some programs decide what color to paint, some decide where to read a color from the query, and others infer the output dimensions and how to sample or summarize the input. All six program families receive the same RC9 predictions. Their first two distinct proposals are combined by agreement across families, with existing nonidentity guesses kept first.

| Tasks solved exactly | RC9, two attempts | RC10, one attempt | RC10, two attempts |
|---|---:|---:|---:|
| ARC-AGI-1 public evaluation | 71/400 | 88/400 | 98/400 |
| ARC-AGI-2 public evaluation | 0/120 | 0/120 | 0/120 |
| Existing ARC-2 training sample | 4/100 | 4/100 | 7/100 |
| Fresh ARC-2 training sample, selected after freeze | 3/100 | 6/100 | 7/100 |

All 71 previous ARC-1 solves were retained. We scored every test grid in a task for an exact match within the attempt budget. The public evaluation data has been used during development. These are local results. The ARC-1 mean fraction of correct test grids per task is 24.625%. The stricter whole-task rate shown in the table is 24.5%.

For the fresh check, we froze the six families and merge order, then hash-sampled 100 ARC-2 training tasks after excluding ARC-1 training and evaluation, ARC-2 public evaluation, and the previous sample by ID and content digest. Both versions received the same stripped inputs. RC10 gained four tasks and retained all three RC9 solves. The small sample and the difference from ARC-2 public evaluation limit how far that improvement can be extrapolated.

## Which capabilities helped?

| New family added alone to RC9 | ARC-1 solves | What the demonstrations determine |
|---|---:|---|
| Paint at a learned output size | 83 | Output dimensions and a decision tree over output coordinates, sampled colors, block statistics or scene counts |
| Learn spatial paint rules | 82 | A threshold tree over local offsets, distances to colors, or connected-component measurements |
| Select a source color | 77 | A tree whose leaves can copy from a nearby cell, a ray, a reflected position, or a line |
| Sample a source window | 75 | Region, orientation, output size, per-axis index mapping and palette |
| Draw from landmarks | 73 | A set of displacements or regularly spaced rays whose effects jointly explain all required edits |
| Assemble an object catalog | 72 | Grouping, sorting, alignment and rendering sprites or bars from measured objects |

Each row is a separate comparison against the same baseline. On task `b7999b51` the catalog program grouped cells by color, measured each color's vertical extent and drew bars sorted by decreasing height. For `f45f5ca7` the solver inferred a different horizontal displacement for each of four colors and drew the shifted cells on a blank canvas.

The search still uses a supplied vocabulary of measurements, drawing operations and color sources. It fits programs within that vocabulary. The tree searches rank exact demonstration fits by holding out one demonstration at a time during color-rule fitting. The expanded search also spends more compute than RC9.

A retrospective oracle could recover 102/400 ARC-1 tasks by choosing among all guesses exported by the six families, including guesses dropped in the merge. Its coverage on ARC-2 public was 0/120. This diagnostic uses answer keys only after prediction. On ARC-2 the exported candidate grids themselves need to improve.

## Run

From the repository root, run `bash rc10/predict.sh BLIND_TASK_DIRECTORY predictions.json`. Each JSON file contains paired training inputs and outputs plus test inputs. The predictor writes two guesses per test grid and a separate trace file. It uses the stored RC2 and RC8 banks. A CPU is sufficient. The measured environment used NumPy 2.3.5, SciPy 1.17.0 and scikit-learn 1.8.0, plus Node.js and bash.

Run `python3 rc10/test.py` for regression checks. Run `bash rc10/reproduce.sh` to download pinned ARC data, reconstruct the RC9 baseline and predictions, and compare final score files including prediction hashes across all four splits. For an existing RC9 run, use `bash rc10/bench.sh BLIND_DIR RC9_PREDICTIONS LABELED_DIR OUTPUT_PREFIX`. Labels are used by the scorer.

[Per-task results, ablations and retrospective coverage](results/)
