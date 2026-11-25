"""Use the experimental ranking only on tasks with a new-family proposal."""
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def select(baseline, ranked, directories):
    result = {}
    for task, attempts in baseline.items():
        active = any(any(row['candidates'] for row in read(folder / f'{task}.json'))
                     for folder in directories)
        result[task] = ranked[task] if active else attempts
    return result


if __name__ == '__main__':
    baseline, ranked, output, *directories = map(Path,sys.argv[1:])
    output.write_text(json.dumps(select(read(baseline),read(ranked),directories),
                                 separators=(',',':'))+'\n')
