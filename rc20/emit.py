"""Emit candidates from the compositional solver."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import objects
import project


def solve(job):
    path,out,family=job
    task=json.loads(path.read_text())
    if not task['train'] or not task['test'] or any('output' in q for q in task['test']):
        raise ValueError('Expected demonstrations and unlabeled queries')
    n=len(task['train'])
    candidates=[{} for _ in task['test']]
    for cost,outputs in sorted(family(task),key=lambda x:x[0]):
        for row,grid in zip(candidates,outputs):
            row.setdefault(json.dumps(grid),[n-min(.99,.01*cost),n,grid])
    (out/path.name).write_text(json.dumps([dict(status='ok',candidates=list(row.values())) for row in candidates]))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('blind',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('--family',choices=('object','projection'),default='object')
    p.add_argument('--workers',type=int,default=5)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    family={'object':objects.rules,'projection':project.rules}[args.family]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(solve,[(p,args.out,family) for p in sorted(args.blind.glob('*.json'))]))
