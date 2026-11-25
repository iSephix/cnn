"""Emit blind candidates and source programs for object correspondence."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from correspondence import fit
from extract import extract


def emit(job):
    path, dest, traces, solver = job
    task=json.loads(path.read_text())
    if not task['train'] or not task['test'] or any('output' in q for q in task['test']):
        raise ValueError('Expected demonstrations and unlabeled queries')
    candidates=[{} for _ in task['test']]
    explanations={}
    n=len(task['train'])
    for cost,outputs,trace in sorted(solver(task),key=lambda a:a[0]):
        key=json.dumps(outputs)
        explanations.setdefault(key,dict(cost=cost,outputs=outputs,rule=trace))
        for row,grid in zip(candidates,outputs):
            row.setdefault(json.dumps(grid),[n-min(.99,.01*cost),n,grid])
    (dest/path.name).write_text(json.dumps([dict(status='ok',candidates=list(row.values())[:20]) for row in candidates]))
    if traces is not None:
        (traces/path.name).write_text(json.dumps(list(explanations.values())[:20],indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('blind',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--family',choices=('correspondence','extraction'),default='correspondence')
    p.add_argument('--traces',type=Path)
    args=p.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    if args.traces:
        args.traces.mkdir(exist_ok=True,parents=True)
    paths=sorted(args.blind.glob('*.json'))
    solver=fit if args.family=='correspondence' else extract
    with ProcessPoolExecutor(6) as pool:
        list(pool.map(emit,[(p,args.output,args.traces,solver) for p in paths]))
