"""Compose shared scene programs with an existing verified detector as a terminal."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

from programs import execute, signature, valid, variants
from search import priority

ROOT = Path(__file__).resolve().parents[1]


def init(detectors):
    global bank
    sys.path.insert(0, str(ROOT/'rc23'))
    import bank
    bank.init(detectors, (0,))


def compose(task, model, limit=8):
    if not task['train'] or not task['test'] or any('output' in q for q in task['test']):
        raise ValueError('Expected demonstrations and unlabeled queries')
    n = len(task['train'])
    grids = [np.asarray(e['input']) for e in task['train']+task['test']]
    targets = [np.asarray(e['output']) for e in task['train']]
    if any(not valid(grid) for grid in grids+targets):
        raise ValueError('Expected valid ARC grids')
    grids = [grid.astype(np.int16) for grid in grids]
    targets = [grid.astype(np.int16) for grid in targets]
    frames = [dict((op, g) for op, g in variants(grid) if valid(g)) for grid in grids]
    common = set.intersection(*(set(frame) for frame in frames))
    states = []
    seen = {signature(grids)}
    for op in sorted(common, key=repr):
        # Orientation ensembles already exist in RC25. This bridge spends its
        # budget on changed scene representations.
        if op[0] in ('orient', 'tile', 'scale', 'mirror append'):
            continue
        states.append(([list(op)], [frame[op] for frame in frames]))
    for macro in model['macros']:
        values = [execute(grid, macro['program']) for grid in grids]
        if all(g is not None for g in values):
            states.append((macro['program'], values))
    selected = []
    for program, values in states:
        key = signature(values)
        if key in seen:
            continue
        seen.add(key)
        distance = sum(abs(a-b) for grid, target in zip(values, targets)
                       for a, b in zip(grid.shape, target.shape))
        selected.append(((distance, -priority(program, model['macros']), repr(program)), program, values))
    selected.sort(key=lambda row: row[0])
    candidates = {}
    for _, program, values in selected[:limit]:
        latent = dict(train=[dict(input=g.tolist(), output=y.tolist()) for g, y in zip(values, targets)],
                      test=[dict(input=g.tolist()) for g in values[n:]])
        rows = bank.generate(latent, bank.DETECTORS, (0,))
        # A detector must produce a prediction for every query.
        names = set.intersection(*(set(name for rec in row for name in rec['detectors']) for row in rows))
        for name in sorted(names):
            options = [[rec for rec in row if name in rec['detectors']] for row in rows]
            if any(len(row) != 1 for row in options):
                continue
            outputs = [row[0]['grid'] for row in options]
            key = json.dumps(outputs, separators=(',', ':'))
            candidates.setdefault(key, dict(program=program, detector=name, grids=outputs,
                                             cost=len(program)+1))
    return dict(candidates=sorted(candidates.values(), key=lambda row: (row['cost'], row['detector'])),
                states=min(limit, len(selected)), budget=limit)


def process(job):
    path, output, model = job
    result = compose(json.loads(path.read_text()), model)
    (output/path.name).write_text(json.dumps(result, separators=(',', ':'))+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('blind', 'output', 'model', 'detectors'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = json.loads(args.model.read_text())
    with ProcessPoolExecutor(args.workers, initializer=init, initargs=(args.detectors.resolve(),)) as pool:
        list(pool.map(process, [(path, args.output, model) for path in sorted(args.blind.glob('*.json'))]))
