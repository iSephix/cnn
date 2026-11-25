"""Search over compressed input scenes and reconstruct decoded outputs."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rc12'))
from native import search


def background(g):
    return np.bincount(g.ravel()).argmax()


def crop(g):
    pts=np.argwhere(g!=background(g))
    if not len(pts):
        return g
    lo,hi=pts.min(axis=0),pts.max(axis=0)+1
    return g[lo[0]:hi[0],lo[1]:hi[1]]


def compress(g):
    rows=g[np.r_[True,np.any(g[1:]!=g[:-1],axis=1)]]
    return rows[:,np.r_[True,np.any(rows[:,1:]!=rows[:,:-1],axis=0)]]


def separator(g):
    rows=~np.all(g==g[:,[0]],axis=1)
    cols=~np.all(g==g[[0],:],axis=0)
    return g[rows][:,cols] if rows.any() and cols.any() else g


def scenarios(task):
    demos=task['train']
    queries=task['test']
    inputs=[np.array(e['input']) for e in demos+queries]
    outputs=[np.array(e['output']) for e in demos]
    n=len(demos)
    seen=set()
    for transform in (crop,compress,separator):
        xs=[transform(g) for g in inputs]
        if not any(x.size<g.size for x,g in zip(xs[:n],inputs[:n])):
            continue
        if sum(x.size for x in xs[:n]) > .86*sum(g.size for g in inputs[:n]):
            continue
        key=tuple((x.shape,x.tobytes()) for x in xs)
        if key in seen:
            continue
        seen.add(key)
        yield dict(train=[dict(input=x.tolist(),output=y.tolist()) for x,y in zip(xs,outputs)],
                   test=[dict(input=x.tolist()) for x in xs[n:]])


def rules(task, binary):
    with TemporaryDirectory() as directory:
        p=Path(directory)/'task.json'
        for latent in scenarios(task):
            p.write_text(json.dumps(latent))
            yield search(p,binary,2,0,9)
