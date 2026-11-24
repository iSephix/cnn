# RC7 · Learning local grid rules

RC7 solves 56/400 ARC-AGI-1 public evaluation tasks with two guesses, including 55 on the first guess. It retains all 46 RC6 solves. ARC-AGI-2 remains 0/120. On the separate 100-task ARC-AGI-2 training sample defined in RC4, scores are 3/100 for RC6 and 4/100 for RC7. This sample filters exact overlaps and has been used in earlier releases.

Six gains on ARC-AGI-1 came from treating one color as a missing region, reconstructing it with compatible periodic or symmetric observations, and predicting either the whole scene or its reconstructed crop. Three came from induced cell-level decision trees. One came from learning which output tile corresponds to an input neighborhood. A fourth learner infers translated paint strokes around marker cells and added zero benchmark solves. See the change reports for per-task attributions.

The tree input is a hand-written set of scene features, positions, and reflected colors. Trees learn from all demonstration pixels. An entire rule family must also predict every held-out demonstration exactly in leave-one-demonstration-out refits to qualify. Accepted trees compile to sparse numerical comparisons and signed path conditions. Query inference runs the compiled card. A leaf may preserve the input color. The emitted trace contains the compiled tree and the rule family. Cards depend on the feature ordering in rules.py and rc3/scene.py. Portability here covers tree readout on that feature representation.

Each learner fits its task independently. Selected RC6 guesses have priority, followed by distinct RC7 candidates in ascending rule-cost order. Repeated public evaluation and an expanded hand-written operator set limit interpretation. This release has no matched-time ablation or private benchmark result.

## Reproduction

From repository root, run `bash rc7/reproduce.sh` with Node, bash, curl, unzip, Python, NumPy and scikit-learn. Tested with NumPy 2.3.5 and scikit-learn 1.8.0. It rebuilds the earlier numerical pipeline, strips answer keys, fits RC7 on the demonstrations, and compares deterministic prediction hashes, scores and change reports. `python3 rc7/test.py` compares serialized-card inference to the fitted tree on 6,000 fresh feature vectors. To repeat the supplementary check, run `bash rc7/transfer.sh ARC1_DATA ARC2_DATA OUTPUT_DIR` using the pinned datasets from the reproduction script.
