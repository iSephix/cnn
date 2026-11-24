# RC4 · Placement and panel induction

RC4 solved 35 of 400 ARC-AGI-1 public tasks with two guesses and 34 with one. RC3 solved 21 and 20 respectively. The paired evaluation gained 14 tasks and lost zero. On ARC-AGI-2 public evaluation both systems solved zero of 120. Full-task success requires every query to match exactly.

## What the demonstrations teach

- Rigid placement searches for an object whose translation or copy exactly reconstructs the output. It infers which object moves, which object anchors it, and a row and column rule shared by all demonstrations. Coordinates may attach to an edge, align bounding boxes or use a displacement. The twelve generated query grids in the transfer test change both positions and colors while keeping the learned rule fixed.
- Layout induction partitions the output into input-sized blocks. It learns which reflection, rotation or blank block belongs at each position. Eight new ARC tasks used this readout.
- Panel induction partitions an input into equal panels with optional separator lines. It tries cell occupancy counts, joint occupancy masks and joint colors. It learns which output color corresponds to each observed category and rejects candidates when a query contains an unseen category. Six new ARC tasks used this readout.
- Additional candidates compact or extend lines and stamp a transformed template. The rigid mover and these additional candidates contributed zero incremental official-split solves in this release.

The operators and ranking costs are hand-written, with task-specific rules fitted from demonstrations. Each non-identity RC3 prediction is offered first. A maximum of two distinct guesses are offered per query, with identity as fallback. The public evaluation data have been scored repeatedly. The placement and layout experiment was scored first (29/400), then panel induction was added. This is a development result with no independent or private score implied.

## Reproduce

Requires Node.js, Python 3 with NumPy, bash, curl and unzip. Run `bash rc4/reproduce.sh` from repository root. It downloads the pinned datasets, rebuilds earlier predictions, runs four regression tests, and compares per-task scores, prediction hashes and the paired change reports. Query answers are stripped before either model runs. No task ID enters the learner.

To keep predictions and traces locally, run `bash rc4/benchmark.sh ARC1_DATA ARC2_DATA OUTPUT_DIR`. The results directory in this release contains score summaries, prediction hashes and the successful rule for every newly solved task. No matched CPU-budget comparison was run for this release.

A supplementary check runs the frozen RC3 and RC4 solvers on 100 ARC-AGI-2 training tasks chosen by deterministic hash order, excluding matching IDs or exact canonical task JSON from ARC-AGI-1 and ARC-AGI-2 public evaluation. Run `bash rc4/transfer.sh ARC1_DATA ARC2_DATA OUTPUT_DIR`. This is a public training-set diagnostic, not another official evaluation split. The overlap filter cannot detect transformed variants.

On that supplementary sample RC3 solved 1/100 and RC4 solved 3/100.
