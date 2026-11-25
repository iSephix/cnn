from pathlib import Path
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from stamp import rules


def write(job):
    path,out=job
    task=json.loads(path.read_text())
    if any('output' in q for q in task['test']):
        raise ValueError('Expected unlabeled queries')
    n=len(task['train'])
    rows=[{} for _ in task['test']]
    for cost,grids in sorted(rules(task),key=lambda v:v[0]):
        for row,g in zip(rows,grids):
            row.setdefault((g.shape,g.tobytes()),[n-min(.99,.01*cost),n,g.tolist()])
    (out/path.name).write_text(json.dumps([dict(status='ok',candidates=list(r.values())) for r in rows]))


if __name__=='__main__':
    source,out=map(Path,sys.argv[1:])
    out.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(4) as pool:
        list(pool.map(write,[(f,out) for f in sorted(source.glob('*.json'))]))
