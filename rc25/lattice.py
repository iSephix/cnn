"""Fit a program on periodic editable sites and splice it into the original canvas."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def init(detectors):
    global bank, native
    from importlib.util import module_from_spec, spec_from_file_location

    sys.path[:0]=[str(ROOT/'rc23'),str(ROOT/'rc12')]
    import bank

    # Load RC12's helper without resolving the RC25 trace driver of the same name.
    spec=spec_from_file_location('rc12_lattice',ROOT/'rc12'/'native.py')
    native=module_from_spec(spec)
    spec.loader.exec_module(native)
    bank.init(detectors,(0,))


def origin(grid,origin_mode):
    if origin_mode=='grid':
        return 0,0
    bg=np.bincount(grid.ravel()).argmax()
    pts=np.argwhere(grid!=bg)
    return tuple(pts.min(axis=0)) if len(pts) else (0,0)


def lattices(grids,targets):
    n=len(targets)
    deltas=[np.argwhere(g!=y) for g,y in zip(grids,targets)]
    if any(not len(pts) for pts in deltas):
        return
    seen=set()
    fits=[]
    for anchor in ('grid','foreground'):
        origins=[origin(g,anchor) for g in grids]
        first=deltas[0][0]-origins[0]
        for sr,sc in product(range(1,7),repeat=2):
            if sr*sc<2:
                continue
            phase=first%[sr,sc]
            if any(np.any((points-base)%[sr,sc]!=phase) for points,base in zip(deltas,origins)):
                continue
            axes=[]
            for g,base in zip(grids,origins):
                r,c=(np.array(base)+phase)%[sr,sc]
                axes.append((slice(r,None,sr),slice(c,None,sc)))
            if any(min(g[a].shape)<2 for g,a in zip(grids,axes)):
                continue
            key=tuple((a.start,a.step,b.start,b.step) for a,b in axes)
            if key in seen:
                continue
            seen.add(key)
            factor=sum(g[a].size for g,a in zip(grids[:n],axes))/sum(g.size for g in grids[:n])
            fits.append((factor*(1+int(anchor=='foreground')*.12),axes,anchor))
    yield from sorted(fits,key=lambda v:v[0])[:4]


def predict(task,binary):
    demos=task['train']
    n=len(demos)
    grids=[np.array(e['input']) for e in demos+task['test']]
    targets=[np.array(e['output']) for e in demos]
    outputs=[{} for _ in task['test']]
    if any(g.shape!=y.shape for g,y in zip(grids,targets)):
        return [[] for _ in outputs]
    with TemporaryDirectory() as temp:
        path=Path(temp)/'latent.json'
        for compression,axes,anchor in lattices(grids,targets):
            small=[g[a] for g,a in zip(grids,axes)]
            task_view=dict(train=[dict(input=g.tolist(),output=y[a].tolist()) for g,y,a in
                                  zip(small,targets,axes)],test=[dict(input=g.tolist()) for g in small[n:]])
            path.write_text(json.dumps(task_view))
            guesses=native.search(path,binary,2,0,2)
            detectors=bank.generate(task_view,bank.DETECTORS,(0,))
            for row,raw,axis,generated,exact in zip(outputs,grids[n:],axes[n:],guesses,detectors):
                options=[(c['grid'],'detector') for c in exact]
                options.extend((c[-1],'native') for c in generated['candidates'] if c[0]>n-1)
                for g,source in options:
                    latent=np.array(g)
                    if latent.shape!=raw[axis].shape:
                        continue
                    answer=raw.copy()
                    answer[axis]=latent
                    key=answer.tobytes()
                    row.setdefault(key,dict(grid=answer.tolist(),anchor=anchor,
                                             source=source,compression=compression))
    return [list(row.values())[:24] for row in outputs]


def process(job):
    path,out,binary=job
    task=json.loads(path.read_text())
    native.blind(task)
    bank.write_rows(out/path.name,predict(task,binary))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('blind','output','binary','detectors'):
        p.add_argument(name,type=Path)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(a.workers,initializer=init,initargs=(a.detectors.resolve(),)) as pool:
        list(pool.map(process,[(p,a.output,str(a.binary.resolve())) for p in sorted(a.blind.glob('*.json'))]))
