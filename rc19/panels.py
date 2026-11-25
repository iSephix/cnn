"""Solve output panels with existing native synthesis, then assemble them."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rc12'))
from native import search


def scenarios(outputs, inputs):
    for axis in (0,1):
        for gap in (0,1):
            if any(g.shape[axis]<5 or (g.shape[axis]-gap)%2 for g in outputs):
                continue
            halves=[np.split(g if axis==0 else g.T,
                             [(g.shape[axis]-gap)//2,(g.shape[axis]+gap)//2]) for g in outputs]
            separators=[int(mid.flat[0]) for _,mid,_ in halves if gap and np.all(mid==mid.flat[0])]
            if gap and (len(separators)!=len(outputs) or len(set(separators))!=1):
                continue
            if not gap and not all(
                a.shape==x.shape or a.T.shape==x.shape or len(set(a.ravel())^set(b.ravel()))>=2
                for (a,_,b),x in zip(halves,inputs)
            ):
                continue
            if all(np.array_equal(a,b) for a,_,b in halves):
                continue
            left=[a if axis==0 else a.T for a,_,_ in halves]
            right=[b if axis==0 else b.T for _,_,b in halves]
            yield axis,separators[0] if gap else None,left,right


def merge(left,right,axis,separator):
    a,b=np.asarray(left),np.asarray(right)
    if a.shape!=b.shape:
        return
    if a.shape[axis]*2+int(separator is not None)>30:
        return
    pieces=[a,b]
    if separator is not None:
        shape=list(a.shape)
        shape[axis]=1
        pieces.insert(1,np.full(shape,separator))
    return np.concatenate(pieces,axis).tolist()


def solve(task, binary):
    demos=task['train']
    n=len(demos)
    results=[{} for _ in task['test']]
    with TemporaryDirectory() as directory:
        path=Path(directory)/'panel.json'
        for axis,separator,*halves in scenarios([np.array(e['output']) for e in demos],
                                                 [np.array(e['input']) for e in demos]):
            parts=[]
            for panels in halves:
                sub=dict(train=[dict(input=e['input'],output=g.tolist()) for e,g in zip(demos,panels)],
                         test=task['test'])
                path.write_text(json.dumps(sub))
                records=search(path,binary,2,0,7)
                candidates=[[v for v in row['candidates'] if v[0]>n-1][:3] for row in records]
                if not all(candidates):
                    break
                parts.append(candidates)
            if len(parts)!=2:
                continue
            for row,aa,bb in zip(results,*parts):
                for a,b in product(aa,bb):
                    grid=merge(a[-1],b[-1],axis,separator)
                    if grid is not None:
                        key=json.dumps(grid)
                        score=(a[0]+b[0])/2-.02
                        if key not in row or score>row[key][0]:
                            row[key]=[score,n,grid]
    return [dict(status='ok',candidates=sorted(row.values(),key=lambda v:-v[0])[:16]) for row in results]


def process(job):
    path,output,binary,family=job
    task=json.loads(path.read_text())
    if not task['train'] or not task['test'] or any('output' in q for q in task['test']):
        raise ValueError('Expected demonstrations and unlabeled queries')
    if family=='planes':
        import planes
        rows=planes.solve(task,binary)
    else:
        rows=solve(task,binary)
    (output/path.name).write_text(json.dumps(rows,separators=(',',':')))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('blind','out','binary'):
        p.add_argument(name,type=Path)
    p.add_argument('--family',choices=('panels','planes'),default='panels')
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    binary=str(args.binary.resolve())
    with ProcessPoolExecutor(5) as pool:
        list(pool.map(process,[(q,args.out,binary,args.family) for q in sorted(args.blind.glob('*.json'))]))
