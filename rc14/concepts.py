"""Emit demonstration-consistent predictions from the RC14 concept families."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import panels
import objects
import repair


def generate(task):
    if not task['train'] or not task['test'] or any('output' in q for q in task['test']):
        raise ValueError('Expected demonstrations and unlabeled queries')
    n=len(task['train'])
    answers=[{} for _ in task['test']]
    for cost,grids in sorted([*panels.rules(task),*repair.rules(task),*objects.rules(task)],key=lambda p:p[0]):
        for row,g in zip(answers,grids):
            row.setdefault((g.shape,g.tobytes()),[n-.01*cost,n,g.tolist()])
    return [dict(status='ok', candidates=list(row.values())) for row in answers]


def write(job):
    path,out=job
    rows=generate(json.loads(path.read_text()))
    (out/path.name).write_text(json.dumps(rows,separators=(',',':')))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('tasks',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(write,[(f,args.out) for f in sorted(args.tasks.glob('*.json'))]))
