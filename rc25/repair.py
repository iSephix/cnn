"""Induce a small residual program that turns a solver trace into an exact fit."""

import numpy as np


def views(grid):
    yield 'unchanged',grid
    for k in range(1,4):
        yield ('rotate',k),np.rot90(grid,k)
    yield 'horizontal',grid[:,::-1]
    yield 'vertical',grid[::-1]
    yield 'transpose',grid.T
    bg=int(np.bincount(grid.ravel()).argmax())
    y,x=np.where(grid!=bg)
    if len(x):
        yield 'crop',grid[y.min():y.max()+1,x.min():x.max()+1]


def translate(grid,dy,dx):
    h,w=grid.shape
    out=np.zeros(grid.shape,dtype=np.int16)
    if abs(dy)<h and abs(dx)<w:
        out[max(0,dy):min(h,h+dy),max(0,dx):min(w,w+dx)]=grid[max(0,-dy):min(h,h-dy),max(0,-dx):min(w,w-dx)]
    return out


def context(reference, proposal):
    h,w=proposal.shape
    r,c=np.indices((h,w))
    yield 'none',np.zeros((h,w),np.int16)
    yield 'row parity',r%2
    yield 'column parity',c%2
    yield 'parity',(r+c)%2
    yield 'edge distance',np.minimum.reduce([r,c,h-1-r,w-1-c])
    yield 'half row',np.sign(r*2-h+1)+1
    yield 'half column',np.sign(c*2-w+1)+1
    if reference.shape==proposal.shape:
        yield 'original',reference
        yield 'original symmetry',reference[::-1,::-1]
        yield 'original mirrored columns',reference[:,::-1]
        yield 'original mirrored rows',reference[::-1]
        yield 'was changed',reference!=proposal
    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
        yield ('neighbor',dy,dx),translate(proposal,dy,dx)


def fit(inputs, outputs, proposals, query, prediction):
    """Yield exact demonstration fits and their query result."""
    all_views=[dict(views(g)) for g in [*proposals,prediction]]
    n=len(outputs)
    emitted=set()
    for transform in all_views[0]:
        if any(transform not in frame for frame in all_views):
            continue
        images=[frame[transform] for frame in all_views]
        if any(a.shape!=b.shape for a,b in zip(images,outputs)):
            continue
        correctness=[np.mean(a==b) for a,b in zip(images,outputs)]
        if min(correctness)<.52 or np.mean(correctness)<.76:
            continue
        differences=[a!=b for a,b in zip(images,outputs)]
        if not any(d.any() for d in differences):
            if transform!='unchanged':
                yield 2,images[-1],('geometry',transform),1.
            continue
        frames=[dict(context(a,b)) for a,b in zip([*inputs,query],images)]
        common=sorted(set.intersection(*(set(f) for f in frames)),key=repr)
        colors=np.concatenate([y.ravel() for y in outputs])
        for field in common:
            codes=[10*frame[field].astype(np.int16)+g for frame,g in zip(frames,images)]
            train=np.concatenate([a.ravel() for a in codes[:n]])
            length=max(int(train.max()),int(codes[-1].max()))+1
            table=np.arange(length)%10
            table[train]=colors
            if not np.array_equal(table[train],colors):
                continue
            edits=np.flatnonzero(table!=np.arange(length)%10)
            if not 1<=len(edits)<=12:
                continue
            result=table[codes[-1]]
            key=result.shape,result.tobytes()
            if key in emitted:
                continue
            emitted.add(key)
            cost=2+len(edits)+int(field!='none')+int(transform!='unchanged')
            yield cost,result,(transform,field,[(int(k),int(table[k])) for k in edits]),float(np.mean(correctness))
