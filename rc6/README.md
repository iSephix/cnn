# RC6 · Searching operation order

The solver builds a grid program breadth-first from 18 available operations. It tests the same program on every demonstration and retains exact fits. Operations can repeat or occur in either order. The primitives, breadth-first traversal, and depth limit are supplied by the implementation. The inferred program order is task-specific. There is no trained cross-task ordering policy in this experiment.

The deterministic release adds this search to RC5 with 700 attempted expansions per task and a maximum depth of three. RC5's fitted guesses take priority. The search deduplicates states on both demonstration and unlabeled query inputs. It reads no query labels. Every task receives at most two distinct guesses, with input copying as fallback.

## Controls

Three separate runs share the same frozen RC4 predictions. In each, the additional search is allowed 20 ms of process CPU per task. The first uses RC5 panel inference, the second uses RC6 primitives in nondecreasing operator-index order, and the third allows arbitrary order with those same primitives. Fixed and free variants also share a 700-expansion ceiling. Timing excludes I/O and merging predictions. Equal allowances do not imply identical CPU consumption. Timed runs vary by machine. The unrestricted-time release uses the full RC5 prior and cannot isolate the effect of order by itself.

Run `bash rc6/reproduce.sh` from the repository root. The script rebuilds the two baselines, runs a mirror-then-crop regression check, measures both public evaluation splits, and compares deterministic prediction hashes and gains against the release report. Requires Node.js, Linux Python with NumPy, bash, unzip and curl. Public splits have been used for prior development. Scores are self-reported.
