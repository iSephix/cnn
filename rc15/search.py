import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

from compose import generate
from scene import rules


def solve(task):
    rows=generate(task)
    n=len(task['train'])
    for cost,outputs in rules(task):
        for row,grid in zip(rows,outputs):
            candidate=[n-min(.99,.01*cost),n,grid.tolist()]
            row['candidates'].append(candidate)
    for row in rows:
        unique={}
        for item in sorted(row['candidates'],key=lambda c:-c[0]):
            unique.setdefault(json.dumps(item[-1]),item)
        row['candidates']=list(unique.values())
    return rows


def write(job):
    path,out=job
    (out/path.name).write_text(json.dumps(solve(json.loads(path.read_text())),separators=(',',':')))


if __name__=='__main__':
    source,out=map(Path,sys.argv[1:])
    out.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(4) as pool:
        list(pool.map(write,[(f,out) for f in sorted(source.glob('*.json'))]))
