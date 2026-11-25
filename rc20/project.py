"""Learn output-cell rules over input projections and inferred output dimensions."""
from itertools import combinations, product

import numpy as np

from shape import rank_sizes


def mirror(offset, length):
    period=2*length
    index=offset%period
    return np.minimum(index,period-index-1)


def fields(g, shape):
    h,w=g.shape
    oh,ow=shape
    r,c=np.indices(shape)
    yield 'checker',(r+c)%2
    yield 'row parity',r%2
    yield 'column parity',c%2
    yield 'row edge',np.minimum(r,oh-r-1)
    yield 'column edge',np.minimum(c,ow-c-1)
    yield 'row',r
    yield 'column',c
    if g.size==oh*ow:
        yield 'reshape',g.reshape(shape)
    for swapped in (False,True):
        a=g.T if swapped else g
        ah,aw=a.shape
        tag='transposed' if swapped else 'plain'
        coordinates=[('scale',r*ah//oh,c*aw//ow),('tile',r%ah,c%aw),
                     ('mirror',mirror(r,ah),mirror(c,aw))]
        if oh%ah==0 and ow%aw==0:
            coordinates.append(('spread',r//(oh//ah),c//(ow//aw)))
        for kind,rr,cc in coordinates:
            yield (tag,kind),a[rr,cc]
            if kind in ('tile','scale'):
                yield (tag,kind,'vertical flip'),a[ah-1-rr,cc]
                yield (tag,kind,'horizontal flip'),a[rr,aw-1-cc]
                yield (tag,kind,'opposite'),a[ah-1-rr,aw-1-cc]
    mask=g!=np.bincount(g.ravel()).argmax()
    yy,xx=np.where(mask)
    if len(yy):
        box=g[yy.min():yy.max()+1,xx.min():xx.max()+1]
        bh,bw=box.shape
        yield 'object scale',box[r*bh//oh,c*bw//ow]
        yield 'object tile',box[r%bh,c%bw]


def fit(frames, labels, keys):
    result={}
    for frame,y in zip(frames,labels):
        values=np.column_stack([frame[k].ravel() for k in keys])
        uni,inverse=np.unique(values,axis=0,return_inverse=True)
        table=np.empty(len(uni),dtype=y.dtype)
        table[inverse]=y.ravel()
        if not np.array_equal(table[inverse],y.ravel()):
            return
        for key,color in zip(map(tuple,uni),table):
            if result.setdefault(key,color)!=color:
                return
    if len(result)<=24:
        return result


def rules(task):
    demos=task['train']
    train=[np.array(e['input']) for e in demos]
    target=[np.array(e['output']) for e in demos]
    if all(x.shape==y.shape for x,y in zip(train,target)):
        return
    query=[np.array(e['input']) for e in task['test']]
    widths=[rank_sizes(train,target,x)[:3] for x in query]
    if not all(widths):
        return
    frames=[dict(fields(x,y.shape)) for x,y in zip(train,target)]
    shared=sorted(set.intersection(*(set(f) for f in frames)),key=repr)
    hypotheses=[]
    seen=set()
    for keys in [*((k,) for k in shared),*combinations(shared,2)]:
        table=fit(frames,target,keys)
        if table is None:
            continue
        score=len(keys)+.1*len(table)
        hypotheses.append((score,keys,table))
    for indices in product(*(range(len(sizes)) for sizes in widths)):
        if sum(indices)>3:
            continue
        fs=[dict(fields(x,sizes[i])) for x,sizes,i in zip(query,widths,indices)]
        for cost,keys,table in hypotheses:
            outputs=[]
            for f in fs:
                if any(k not in f for k in keys):
                    break
                cells=zip(*(f[k].ravel() for k in keys))
                colors=[table.get(k,-1) for k in cells]
                if -1 in colors:
                    break
                outputs.append(np.array(colors).reshape(f[keys[0]].shape).tolist())
            else:
                signature=tuple(tuple(map(tuple,g)) for g in outputs)
                if signature not in seen:
                    seen.add(signature)
                    yield 4+sum(indices)+cost,outputs
