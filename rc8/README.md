# RC8 — Reusing learned numerical rules

The solver mines 189 numerical cards from 91 ARC-AGI-1 training tasks. On a new task it applies each stored rule to the demonstration inputs in eight orientations and tries both literal colors and frequency-based color roles. It can learn a new mapping from the card's output and the input color to the target color. Only candidates that reproduce the demonstrations qualify. The solver preserves earlier nonidentity predictions and fills the two available attempts with distinct grids when possible.

| Evaluation | RC7, two attempts | RC8, first attempt | RC8, two attempts |
|---|---:|---:|---:|
| ARC-AGI-1 public | 56/400 | 59/400 | 61/400 |
| ARC-AGI-2 public | 0/120 | 0/120 | 0/120 |
| Supplementary ARC-AGI-2 training sample | 4/100 | 3/100 | 4/100 |

A task counts as solved when every test grid matches exactly in the allowed attempts. These are local measurements. Prediction runs receive test inputs with their outputs removed. We use the public evaluation scores during development.

All five gains came from adapting the output mapping of a stored card. No prior solve was lost. The additional spatial and rule-fragment search produced no further solves on these sets. The standalone frozen-card probe scored zero on the public ARC-1 split, the supplementary sample, and the disjoint 200-task training probe. A retrospective oracle found that input color plus a single stored threshold could explain four gains. For task 1c0d0a4b, none of the individual thresholds tested sufficed, while the card acquired from bdad9b1f succeeded. This comparison uses test answers for diagnosis and does not measure how well a simpler solver would rank those rules.

## Reproduction

Run `bash rc8/reproduce.sh` from the repository with Python 3, NumPy, Node.js, curl and unzip. It rebuilds the earlier predictions, regenerates the bank from pinned training data and compares both final score files, including prediction hashes. Run `bash rc8/transfer.sh ARC1_DATA ARC2_DATA OUT` for the supplementary sample and `bash rc8/crossfit.sh ARC1_TRAINING OUT` for the frozen-card training split probe. Dataset arguments to transfer.sh point to directories containing training and evaluation subdirectories.

[Task-level scores and gain traces](results/) · [Training cards and origins](bank.json)
