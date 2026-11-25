"""Merge a unique compositional prediction into an existing two-attempt submission."""
import argparse
import json
from pathlib import Path


def merge(task, result, baseline):
    if any('output' in query for query in task['test']):
        raise ValueError('Expected unlabeled queries')
    if len(task['test']) != len(baseline):
        raise ValueError('Baseline query count does not match the task')
    candidates = result['candidates']
    if any(len(row['grids']) != len(baseline) for row in candidates):
        raise ValueError('Candidate query count does not match the task')
    output = []
    for index, original in enumerate(baseline):
        row = dict(original)
        if len(candidates) == 1:
            grid = candidates[0]['grids'][index]
            if grid != original['attempt_1']:
                row['attempt_2'] = grid
        output.append(row)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('blind', 'cache', 'baseline', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    old = json.loads(args.baseline.read_text())
    result = {}
    for path in sorted(args.blind.glob('*.json')):
        result[path.stem] = merge(json.loads(path.read_text()),
                                 json.loads((args.cache/path.name).read_text()), old[path.stem])
    args.output.write_text(json.dumps(result, separators=(',', ':'))+'\n')
