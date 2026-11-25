"""Promote a unanimous demonstration-consistent concept prediction."""
import json
import sys
from pathlib import Path


def merge(prior, directory):
    for task, rows in prior.items():
        records = json.loads((directory / (task + '.json')).read_text())
        if len(records) != len(rows):
            raise ValueError(f'Missing query for {task}')
        for row, record in zip(rows, records):
            if record['candidates']:
                grid = record['candidates'][0][-1]
                if len(record['candidates']) == 1 and grid != row['attempt_1']:
                    row['attempt_2'] = row['attempt_1']
                    row['attempt_1'] = grid
                elif grid != row['attempt_1']:
                    row['attempt_2'] = grid
    return prior


if __name__ == '__main__':
    prior, candidates, output = map(Path, sys.argv[1:])
    result = merge(json.loads(prior.read_text()), candidates)
    output.write_text(json.dumps(result, separators=(',', ':')) + '\n')
