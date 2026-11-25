"""Generate candidates without access to query outputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import markers
import place
import bind
import repair
import regions
import route


def emit(job):
    path,dest,solver=job
    task=json.loads(path.read_text())
    if not task['train'] or not task['test'] or any('output' in q for q in task['test']):
        raise ValueError('Expected demonstrations and unlabeled queries')
    rows=[{} for _ in task['test']]
    n=len(task['train'])
    for cost,outputs in sorted(solver(task),key=lambda rec:rec[0]):
        for row,grid in zip(rows,outputs):
            row.setdefault(json.dumps(grid),[n-min(.99,cost*.01),n,grid])
    (dest/path.name).write_text(json.dumps([dict(status='ok',candidates=list(row.values())[:20]) for row in rows]))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    solvers={'markers':markers.rules,'repair':repair.rules,'route':route.rules,'bind':bind.rules,'regions':regions.rules,'place':place.rules}
    p.add_argument('family',choices=tuple(solvers))
    p.add_argument('blind',type=Path)
    p.add_argument('out',type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(5) as pool:
        list(pool.map(emit,[(path,a.out,solvers[a.family]) for path in sorted(a.blind.glob('*.json'))]))
