"""Write candidate grids for each answer-blind task."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import relations
import latent
import render


def run(job):
    path, folder, family, binary = job
    task=json.loads(path.read_text())
    if not task['train'] or not task['test'] or any('output' in q for q in task['test']):
        raise ValueError('Expected demonstrations and unlabeled queries')
    n=len(task['train'])
    rows=[{} for _ in task['test']]
    if family=='latent':
        for variant in latent.rules(task,binary):
            for row,result in zip(rows,variant):
                for score,count,grid in result['candidates']:
                    key=json.dumps(grid)
                    previous=row.get(key)
                    if previous is None or score>previous[0]:
                        row[key]=[score,count,grid]
    else:
        for cost,outputs in sorted({'relation':relations.rules,'render':render.rules}[family](task),key=lambda rule:rule[0]):
            for row,grid in zip(rows,outputs):
                row.setdefault(json.dumps(grid),[n-min(.99,.01*cost),n,grid])
    rows=[sorted(row.values(),key=lambda item:-item[0]) for row in rows]
    (folder/path.name).write_text(json.dumps([dict(status='ok',candidates=r) for r in rows]))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('blind',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--family',choices=('relation','latent','render'),default='relation')
    parser.add_argument('--binary',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.family=='latent' and args.binary is None:
        parser.error('--binary is required for latent search')
    binary=str(args.binary.resolve()) if args.binary else None
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(run,[(p,args.out,args.family,binary) for p in sorted(args.blind.glob('*.json'))]))
