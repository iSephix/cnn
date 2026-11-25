"""Recheck external symbolic detectors and pool their predictions across orientations."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
import signal

import numpy as np


def orient(grid, code):
    return np.rot90(grid if code<4 else grid[:,::-1],code%4)


def restore(grid, code):
    unrotated=np.rot90(grid,-(code%4))
    return unrotated if code<4 else unrotated[:,::-1]


def grid(value):
    if value is None:
        return
    a=np.asarray(value)
    if a.ndim==2 and min(a.shape)>0 and max(a.shape)<=30 and a.dtype.kind in 'iub' and a.min()>=0 and a.max()<=9:
        return a.astype(np.int8)


def timeout(signum,frame):
    raise TimeoutError('Detector exceeded its time budget')


def init(path, transforms, exclusions=()):
    global DETECTORS, TRANSFORMS
    sys.path.insert(0,str(path))
    from registry import ALL
    signal.signal(signal.SIGALRM,timeout)
    DETECTORS=tuple(d for d in ALL if d.__module__ not in exclusions)
    TRANSFORMS=tuple(transforms)


def generate(task, detectors, transforms):
    if not task['train'] or not task['test'] or any('output' in q for q in task['test']):
        raise ValueError('Expected demonstrations and unlabeled queries')
    train=[(np.array(e['input']),np.array(e['output'])) for e in task['train']]
    queries=[np.array(e['input']) for e in task['test']]
    rows=[{} for _ in queries]
    seen=set()
    for code in transforms:
        demos=[(orient(a,code).copy(),orient(b,code).copy()) for a,b in train]
        tests=[orient(a,code).copy() for a in queries]
        signature=tuple((a.shape,a.tobytes()) for pair in demos for a in pair)
        signature+=tuple((a.shape,a.tobytes()) for a in tests)
        if signature in seen:
            continue
        seen.add(signature)
        for detector in detectors:
            try:
                try:
                    signal.setitimer(signal.ITIMER_REAL,1.5)
                    f=detector([(a.copy(),b.copy()) for a,b in demos])
                    if f is None or not all(np.array_equal(grid(f(a.copy())),b) for a,b in demos):
                        continue
                    for row,a in zip(rows,tests):
                        value=grid(f(a.copy()))
                        if value is None:
                            continue
                        result=restore(value,code)
                        key=(result.shape,result.tobytes())
                        evidence=row.setdefault(key,dict(grid=result.tolist(),detectors=Counter(),orientations=set(),identity_votes=0))
                        name=f'{detector.__module__}.{detector.__name__}'
                        evidence['detectors'][name]+=1
                        evidence['orientations'].add(code)
                        evidence['identity_votes']+=code==0
                finally:
                    signal.setitimer(signal.ITIMER_REAL,0)
            except Exception:
                # Each external detector is an independent optional hypothesis.
                continue
    data=[]
    for row in rows:
        choices=[]
        for value in row.values():
            value['detectors']=dict(value['detectors'])
            value['orientations']=sorted(value['orientations'])
            choices.append(value)
        choices.sort(key=lambda c:(-c['identity_votes'],-len(c['detectors']),-len(c['orientations'])))
        data.append(choices[:40])
    return data


def write_rows(destination,rows):
    with NamedTemporaryFile(mode='w',dir=destination.parent,prefix='.',delete=False) as handle:
        path=Path(handle.name)
        try:
            json.dump(rows,handle,separators=(',',':'))
            handle.close()
            path.replace(destination)
        finally:
            path.unlink(missing_ok=True)


def process(job):
    path,out=job
    task=json.loads(path.read_text())
    rows=generate(task,DETECTORS,TRANSFORMS)
    dest=out/path.name
    write_rows(dest,rows)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('blind','out','bank'):
        p.add_argument(name,type=Path)
    p.add_argument('--workers',type=int,default=7)
    p.add_argument('--skip-modules',type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--canonical-only',action='store_true')
    args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    transforms=(0,) if args.canonical_only else (0,1,2,3,4,5,6,7)
    paths=[p for p in sorted(args.blind.glob('*.json')) if not (args.resume and (args.out/p.name).exists())]
    exclusions=tuple(json.loads(args.skip_modules.read_text())) if args.skip_modules else ()
    with ProcessPoolExecutor(args.workers,initializer=init,initargs=(args.bank.resolve(),transforms,exclusions)) as pool:
        list(pool.map(process,[(p,args.out) for p in paths]))
