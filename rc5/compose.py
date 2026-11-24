"""Enumerate split, map, orient and assemble programs with a shared color readout."""
import sys
from pathlib import Path
from itertools import product
from functools import lru_cache
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rc4'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rc3'))
from scene import Scene
from pattern import orient
from learn import lookup


def spans(mask):
    indices=np.flatnonzero(~mask)
    return [run for run in np.split(indices,np.where(np.diff(indices)>1)[0]+1) if len(run)]


def partition(g,spec):
    if isinstance(spec,int):
        rs=spans(np.all(g==spec,axis=1))
        cs=spans(np.all(g==spec,axis=0))
    else:
        r,c=spec
        if g.shape[0]%r or g.shape[1]%c:
            return None
        rs=np.split(np.arange(g.shape[0]),r)
        cs=np.split(np.arange(g.shape[1]),c)
    if not 2<=len(rs)*len(cs)<=100:
        return None
    return [[g[np.ix_(r,c)] for c in cs] for r in rs]


@lru_cache(maxsize=8192)
def summarize(shape,data,op):
    a=np.frombuffer(data,dtype=np.uint8).reshape(shape)
    if op=='raw':
        return a
    colors,counts=np.unique(a,return_counts=True)
    if op in ('common','rare','ink','distinct','occupied'):
        if op=='common':
            v=colors[counts.argmax()]
        elif op=='rare':
            v=colors[counts.argmin()]
        elif op=='ink':
            v=next((k for k in colors[counts.argsort()[::-1]] if k),0)
        elif op=='distinct':
            v=len(colors)
        else:
            v=int(np.any(a!=0))
        return np.array([[v]])
    objects=Scene(a).objects((True,4,0))
    if not objects:
        return None
    o=(max if op=='largest' else min)(objects,key=lambda o:o.stats['area'])
    return o.patch


def mapped(blocks,op):
    return [[summarize(a.shape,a.astype(np.uint8).tobytes(),op) for a in row] for row in blocks]


def assemble(blocks,op,k):
    flat=[a for row in blocks for a in row]
    if any(a is None for a in flat):
        return None
    if op!='join':
        key=(lambda a:len(np.unique(a))) if 'palette' in op else np.count_nonzero
        a=(max if op.startswith('max') else min)(flat,key=key)
        return orient(a,k)
    try:
        return np.block([[orient(a,k) for a in row] for row in blocks])
    except ValueError:
        return None


def infer(xs,ys,qs):
    grids=xs+qs
    for spec in [*range(10),*product(range(1,4),repeat=2)]:
        parts=[partition(g,spec) for g in grids]
        if any(p is None for p in parts):
            continue
        for op in ('common','ink','rare','occupied','distinct','raw','largest','smallest'):
            blocks=[mapped(p,op) for p in parts]
            for output in ('join','max-palette','min-palette','max-ink','min-ink'):
                for k in range(1 if op in ('common','ink','rare','occupied','distinct') else 8):
                    demo=[]
                    for b,y in zip(blocks,ys):
                        a=assemble(b,output,k)
                        if a is None or a.shape!=y.shape:
                            break
                        demo.append(a)
                    else:
                        mapping=lookup(demo,ys)
                        if mapping is None:
                            continue
                        answers=[]
                        for b in blocks[len(xs):]:
                            a=assemble(b,output,k)
                            if a is None or max(a.shape)>30:
                                break
                            answers.append(np.vectorize(lambda v:mapping.get(v,v),otypes=[int])(a).tolist())
                        else:
                            yield (sum(x!=y for x,y in mapping.items()),int(k!=0)),[spec,op,k,output],answers
