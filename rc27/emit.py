"""Generate complete relational programs without reading query answers."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from concepts import fit
from local import fit as fit_local
from patterns import fit as fit_patterns
from topological import fit as fit_topological
from constraints import fit as fit_constraints
from layout import fit as fit_layout


def process(job):
    path, out, family = job
    task = json.loads(path.read_text())
    solvers = dict(relational=fit, local=fit_local, patterns=fit_patterns,
                   topological=fit_topological, constraints=fit_constraints, layout=fit_layout)
    candidates = list(solvers[family](task))
    def priority(row):
        support = row.get('folds', row.get('demonstrations', 1))/row.get('demonstrations', 1)
        return -support, row['cost']
    candidates.sort(key=priority)
    unique = {}
    for row in candidates:
        unique.setdefault(json.dumps(row['grids']), row)
    candidates = list(unique.values())
    with NamedTemporaryFile(mode='w', dir=out, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(dict(candidates=candidates[:24]), handle, separators=(',', ':'))
            handle.write('\n')
            handle.close()
            temporary.replace(out/path.name)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('blind', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--family', choices=('relational', 'local', 'patterns', 'topological', 'constraints', 'layout'), default='relational')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(process, [(path, args.output, args.family) for path in sorted(args.blind.glob('*.json'))]))
