"""Emit exact structural programs from unlabeled tasks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from structured import structure
from layout import recipes


def process(job):
    path,output,solver=job
    task=json.loads(path.read_text())
    if not task['train'] or not task['test'] or any('output' in q for q in task['test']):
        raise ValueError('Expected demonstrations and unlabeled queries')
    rows=[{} for _ in task['test']]
    for family,cost,outs in sorted(solver(task),key=lambda rec:rec[1]):
        for row,answer in zip(rows,outs):
            row.setdefault(json.dumps(answer),dict(grid=answer,source=family,cost=cost))
    (output/path.name).write_text(json.dumps([list(r.values())[:18] for r in rows],separators=(',',':')))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('blind',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--family',choices=('structure','layout'),default='structure')
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    solver=structure if a.family=='structure' else recipes
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(process,[(p,a.output,solver) for p in sorted(a.blind.glob('*.json'))]))
