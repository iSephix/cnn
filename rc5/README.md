# RC5 · Compositional panel search

RC5 fits programs from demonstrations in a bounded grammar. A program splits at monochrome separator lines or into equal panels, summarizes each panel or extracts an object, applies a symmetry, and joins the results or selects a panel. A shared categorical readout is fitted on demonstrations. RC4's non-identity predictions take priority within the two-guess budget.

On the repeatedly used public evaluation splits RC5 solved 38/400 ARC-AGI-1 tasks, including 37 on the first guess, and 0/120 ARC-AGI-2 tasks. The controlled test recovers a separator split followed by largest-object selection, reflection and horizontal reassembly on unseen colors. The per-task change reports identify the actual programs responsible for benchmark gains. The operation families and stage order are hand-written.

Run `bash rc5/reproduce.sh` from the repository root with NumPy, Python 3, Node, curl, unzip and bash. The script downloads pinned public data, strips query labels before prediction, and compares the resulting scores and prediction hashes. To inspect emitted rules run `python3 rc5/audit.py OUTPUT_DIR` on the output of rc5/benchmark.sh.
