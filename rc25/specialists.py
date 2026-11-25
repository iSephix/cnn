"""Audit additional open-source solvers as optional candidate producers."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import signal
import sys
import warnings

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'rc23'))
from bank import grid, timeout, write_rows


def init(path):
    global Task, TaskExample, SOLVERS
    sys.path.insert(0,str(path))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from arc_solver.data.task import Task, TaskExample
        from arc_solver.solvers import (ChessSolver, GridSolver, RepeatingSolver, SubmaskSolver,
            SubmatrixSolver, SymmetrySolver, TilingSolver, MLSolver)
    SOLVERS=(RepeatingSolver,GridSolver,ChessSolver,TilingSolver,SubmatrixSolver,SubmaskSolver,
             SymmetrySolver,MLSolver)
    signal.signal(signal.SIGALRM,timeout)


def candidates(solver, demonstrations, query):
    try:
        try:
            signal.setitimer(signal.ITIMER_REAL,1.25)
            training=[TaskExample(x.copy(),y.copy()) for x,y in demonstrations]
            raw=solver().solve(Task('unlabeled',training,[query.copy()]))
            result=[]
            for value in raw:
                a=np.asarray(value)
                if a.dtype.kind=='f' and a.ndim==2 and a.size and np.isfinite(a).all() and np.array_equal(a,np.rint(a)):
                    a=a.astype(int)
                value=grid(a)
                if value is not None:
                    result.append(value.tolist())
            return result
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
    except Exception:
        return []


def run(job):
    path,out=job
    task=json.loads(path.read_text())
    if not task['train'] or not task['test'] or any('output' in q for q in task['test']):
        raise ValueError('Expected demonstrations and unlabeled queries')
    demos=[(np.array(e['input']),np.array(e['output'])) for e in task['train']]
    queries=[np.array(q['input']) for q in task['test']]
    result=[{} for _ in queries]
    for solver in SOLVERS:
        guesses=[candidates(solver,demos,q) for q in queries]
        if not all(guesses):
            continue
        hits=sum(y.tolist() in candidates(solver,demos[:i]+demos[i+1:],x)
                 for i,(x,y) in enumerate(demos)) if len(demos)>1 else 0
        for row,options in zip(result,guesses):
            for rank,output in enumerate(options[:8]):
                k=json.dumps(output,separators=(',',':'))
                quality=hits/len(demos)
                record=dict(grid=output,solver=solver.__name__,held_out=quality,rank=rank)
                if k not in row or row[k]['held_out']<quality:
                    row[k]=record
    rows=[sorted(row.values(),key=lambda c:(-c['held_out'],c['rank']))[:20] for row in result]
    write_rows(out/path.name,rows)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('blind','output','upstream'):
        p.add_argument(key,type=Path)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(a.workers,initializer=init,initargs=(a.upstream.resolve(),)) as pool:
        list(pool.map(run,[(path,a.output) for path in sorted(a.blind.glob('*.json'))]))
