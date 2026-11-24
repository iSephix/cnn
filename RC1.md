# Concept Engine · ARC research candidate 1

A standalone research branch of Concept Engine v0.8, using its original sparse threshold evaluator. Run with Node.js (tested on 24.22.3). No external model, API or npm dependencies. The CSV workbench is distributed separately.

## Reproduce

Run `bash reproduce.sh` with Node, bash, curl and unzip. It downloads the two official pinned dataset snapshots, learns from the 400 ARC-AGI-1 training tasks, removes all evaluation query answers before inference and compares new predictions and scores byte-for-byte with this release. The scoring process has access to query answers. No task ID enters the solver. The public splits were scored locally. No private evaluation or leaderboard acceptance is recorded.

## Method

Each candidate combines a hand-written spatial projection with a color transition fitted from the task's demonstrations. A projection produces one categorical feature per output cell. The learned readout maps 10 single-color or 100 paired-color categories to 10 output channels via a sparse threshold layer. Unseen categories copy the first color. Cards serialize to JSON and execute in the v0.8 numerical core for readout. Shape, crop, connected-component and alignment logic still execute in JavaScript. The 51 projections include symmetries, two-panel relationships, crops, repeats, color frequency and adjacent-cell rules. Only one projection is fitted at a time.

The policy uses kernel regression on nine task descriptors to rank projections by exact solution frequency on similar training tasks. Rewards come from the training split's labelled queries. It stores task descriptors and route successes. The global baseline ignores the descriptors, ranking by aggregate training success. The budget counts candidate routes tried, not CPU time. Declared fixed order starts with symmetries. An identity fallback is available to all methods. Predictions allow up to two distinct grids per input.

## Local results

| Public split | Budget | Search order | Tasks fully solved |
|---|---:|---|---:|
| ARC-AGI-1 | 12 | Fixed | 4 / 400 |
| ARC-AGI-1 | 12 | Global frequency | 6 / 400 |
| ARC-AGI-1 | 12 | Conditioned on demonstrations | 11 / 400 |
| ARC-AGI-1 | 51 | Fixed or conditioned | 13 / 400 |
| ARC-AGI-2 | 51 | Frozen ARC-AGI-1 policy | 0 / 120 |

Every credited task was solved on the first guess. The full ARC-AGI-1 model solved 16 of 419 individual queries. Full-task accuracy and mean within-task query accuracy agree for these runs. Results contain the predictions, per-task hits and SHA-256 hashes. No eval task-specific code or parameters were added after scoring.

Coverage audit, computed after inference: a card fit every demonstration on only 16 of 400 ARC-AGI-1 tasks and zero of 120 ARC-AGI-2 tasks. Perfect selection of a single fitted card could solve 13 and zero respectively. Identity alone solved zero on both splits. This isolates representational coverage as the main measured bottleneck here.

## Next research gate

Learn compositions of projections, object relations and cell operations from the training split. Save a successful composition as an independently executable card and compare reuse to fresh search on disjoint task families under the same compute budget. Keep the published RC1 results frozen. Reserve a new holdout for architecture selection. Language grounding and unbounded discovery are open research goals.

The community submission draft requires a real public source URL before publication. Replace the placeholder and rename submission.yaml.in to submission.yaml in the leaderboard fork. Its maintainers decide acceptance.

Benchmark source: https://github.com/fchollet/ARC-AGI and https://github.com/arcprize/ARC-AGI-2. Submission procedure: https://github.com/arcprize/ARC-AGI-Community-Leaderboard/blob/main/CONTRIBUTING.md
