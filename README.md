# Concept Engine

[RC18](rc18/README.md) reaches **160/400 (40.0%)** on public ARC-AGI-1 with object relations and compressed-scene search.

[RC17](rc17/README.md) reaches **159/400 (39.75%)** on public ARC-AGI-1 and adds count-based rendering and transferable spatial stencils.

[RC16](rc16/README.md) tests whether programs mined from solved training puzzles transfer to new tasks under a matched search budget.

[RC15](rc15/README.md) reaches **158/400 (39.5%)** on public ARC-AGI-1 with two attempts.

[RC14](rc14/README.md) promotes demonstration-consistent scene concepts, solving **144/400 on the first attempt** and **156/400 within two attempts** on public ARC-AGI-1.

[RC13](rc13/README.md) solves **156/400 (39.0%)** on ARC-AGI-1 public evaluation, including 143 on the first attempt. New searches learn repeated output structure and Boolean rules for individual colors.

[RC12](rc12/README.md) reaches **154/400 (38.5%)** on ARC-AGI-1 public evaluation and **1/120** on ARC-AGI-2 by ranking native search and Concept Engine proposals together.

[RC11](rc11/README.md) solves **102/400 (25.5%)** ARC-AGI-1 public tasks with object-relative rendering and within-grid exemplar completion.

[RC10](rc10/README.md) solves **98/400 (24.5%)** ARC-AGI-1 public tasks, including 88 on the first attempt. The standalone predictor accepts unlabeled ARC tasks.

[RC9](rc9/README.md) solves 71/400 (17.75%) ARC-AGI-1 public tasks, including 68 on the first attempt, after adding spatial relationships and panel reasoning.

[RC8](rc8/README.md) reaches 61/400 (15.25%) on ARC-AGI-1 public evaluation by reusing learned numerical rules across tasks.

[RC7](rc7/README.md) solves 56/400 ARC-AGI-1 public tasks (14%) and 0/120 ARC-AGI-2 public tasks. New cell rules compile into executable numerical threshold cards.

[RC6](rc6/README.md) searches the order of grid operations and solves 46/400 ARC-AGI-1 public tasks with two attempts.


[RC5](rc5/README.md) reaches 38/400 on ARC-AGI-1 public evaluation with compositional panel search, and 0/120 on ARC-AGI-2.


RC4 solves **35/400 (8.75%)** on ARC-AGI-1 public evaluation and **0/120** on ARC-AGI-2. It infers how to place transformed tiles and how aligned panels combine. Its anchor-based mover also transfers to new positions and colors in controlled tests. [RC4 experiment and reproduction](rc4/README.md).

The [previous object-selection and recoloring experiment](rc3/README.md) solved 21/400 ARC-AGI-1 tasks.

## ARC RC2

Can a grid transformation discovered in one task become an executable part of the next solution? RC2 mines two-step spatial programs from the 400 ARC-AGI-1 training tasks. It saves their projection sequence and learned threshold weights. Six training tasks that no single base projection solved yielded a validated program.

For example, one card crops the nonzero bounding box and enlarges each surviving cell into a 2×2 block. Each saved card can be run independently. On a new task the engine can replay its exact weights, fit a new color readout on its geometry, or use its frozen output as input to a further learned transformation. Geometry is implemented by fixed JavaScript primitives. Language understanding and the discovery of entirely new primitives remain open.

## Results

Local public evaluation, all queries in a task must be correct, up to two guesses per query. All credited ARC tasks were solved on guess one.

| Search | ARC-AGI-1 | ARC-AGI-2 |
|---|---:|---:|
| RC1, 51 single projections | 13 / 400 | 0 / 120 |
| RC2, 51 singles plus 64 more trials, fresh or reuse | 13 / 400 | 0 / 120 |
| RC2, 51 singles plus 512 more trials, fresh or reuse | 14 / 400 | 0 / 120 |
| RC2, full pair search, fresh or reuse | 14 / 400 | 0 / 120 |

Each treatment shares a learned search order trained on ARC-AGI-1. Fresh search fits new projections from scratch. Reuse spends some of the same trial allowance testing stored cards, frozen prefixes and retrieved paths. A trial can have different execution costs. In a separate timing control, both scored 14/400 in each of three runs with a deadline of 10 ms of process CPU time per task. A result completed after the deadline cannot be added. Setup, file I/O and scoring are outside the deadline. Runtime and timing results are machine-dependent.

ARC-AGI-1 task bf699163 accounts for the additional solve at 512 trials. Fresh search found a crop around color 7 followed by a crop to the largest component against the most frequent background. The saved training cards did not increase the aggregate ARC score. The evaluation split was already used for RC1 reporting. Query answers were removed before inference.

After freezing the solver, an oracle audit of 51 singles and 2,051 admissible pairs found at least one demonstration fit for 18 of 400 ARC-AGI-1 tasks. A single fitted program could generalize to all queries on 15. On ARC-AGI-2 both counts were zero. The audit excludes three-stage frozen prefixes and never feeds labels to prediction.

## Transfer experiment

For each mined card, generate eight downstream tasks with a new output color permutation and independently sampled input grids. Each task has five demonstrations and three queries. Exact success requires all three queries. With 64 additional candidate trials, fresh search solved 14/48 and reuse solved 37/48, with 23 gains and zero losses. This is a positive control deliberately constructed to share the six learned operations. The 48 tasks span six source concepts and are correlated. It measures transfer under known shared structure. It offers no estimate of transfer frequency in naturally occurring tasks.

## Run

Requires Node.js 24, bash, curl, unzip. No npm dependencies, external model or GPU.

- Run `bash rc2/reproduce.sh` for deterministic training, benchmark predictions, transfer experiments and byte comparisons against the packaged results. Uses the pinned ARC dataset commits from RC1.
- Run `bash rc2/cpu.sh /path/to/ARC-AGI/data/evaluation` to repeat the machine-dependent CPU deadline control.
- Run `node rc2/coverage.js /path/to/evaluation output.json` for the retrospective coverage audit.
- Run `node rc2/card.js rc2/results/library.json f25fbde4 grid.json` to apply the saved crop-and-enlarge card. The file grid.json contains one rectangular integer grid.
- Read `RC1.md` for the base runtime and single-projection experiment. Version-specific source, predictions, traces, transfer records and score summaries are under `rc2/`.

The draft community leaderboard entry is in submissions/concept-engine/submission.yaml. Public-split scores are self-reported. Publication and acceptance are separate from the benchmark run.
