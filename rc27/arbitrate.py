"""Admit a unique complete concept prediction, preserving the first attempt."""
import argparse
import json
from pathlib import Path


def eligible(record):
    if record['family'] in ('local', 'topological'):
        return record['folds'] == record['demonstrations'] and record['demonstrations'] >= 2
    return True


def select(task, baseline, sources, policy='verified', promote=False):
    if any('output' in query for query in task['test']):
        raise ValueError('Expected unlabeled queries')
    if len(baseline) != len(task['test']):
        raise ValueError('Baseline query count mismatch')
    choices = {}
    for data in sources:
        for record in data['candidates']:
            if len(record['grids']) != len(baseline):
                raise ValueError('Candidate query count mismatch')
            if policy == 'verified' and not eligible(record):
                continue
            key = json.dumps(record['grids'], separators=(',', ':'))
            choices.setdefault(key, record['grids'])
    selected = next(iter(choices.values())) if len(choices) == 1 else None
    rows = []
    for index, original in enumerate(baseline):
        row = dict(original)
        if selected is not None and selected[index] != original['attempt_1']:
            row['attempt_2'] = selected[index]
            if promote:
                row['attempt_1'], row['attempt_2'] = row['attempt_2'], row['attempt_1']
        rows.append(row)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('blind', type=Path)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('sources', nargs='+', type=Path)
    parser.add_argument('--policy', choices=('verified','all'), default='verified')
    parser.add_argument('--promote', action='store_true')
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    output = {}
    for path in sorted(args.blind.glob('*.json')):
        task = json.loads(path.read_text())
        sources = [json.loads((directory/path.name).read_text()) for directory in args.sources]
        output[path.stem] = select(task, baseline[path.stem], sources, args.policy, args.promote)
    args.output.write_text(json.dumps(output, separators=(',', ':'))+'\n')
