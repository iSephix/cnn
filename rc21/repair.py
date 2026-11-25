"""Infer missing cells by enforcing shared geometric orbits."""
from itertools import combinations
import numpy as np


def orbit_maps(shape):
    h,w=shape
    r,c=np.indices(shape)
    yield 'vertical',(h-1-r,c)
    yield 'horizontal',(r,w-1-c)
    yield 'half-turn',(h-1-r,w-1-c)
    if h==w:
        yield 'transpose',(c,r)
        yield 'anti-diagonal',(w-1-c,h-1-r)
        yield 'quarter-turn',(c,h-1-r)


def complete(grid, unknown, trans):
    proposal=grid.copy()
    known=~unknown.copy()
    while not known.all():
        count=np.count_nonzero(known)
        for coords in trans:
            other=proposal[coords]
            available=known[coords]
            if np.any(known&available&(proposal!=other)):
                return
            fill=~known&available
            proposal[fill]=other[fill]
            known[fill]=True
        if count==np.count_nonzero(known):
            return
    if all(np.array_equal(proposal,proposal[coords]) for coords in trans):
        return proposal


def variants(grid, ink):
    unknown=grid==ink
    if not unknown.any() or unknown.all():
        return {}
    maps=dict(orbit_maps(grid.shape))
    options={}
    yy,xx=np.where(unknown)
    for names in [*((n,) for n in maps),*combinations(maps,2)]:
        output=complete(grid,unknown,[maps[n] for n in names])
        if output is None:
            continue
        options[names,'whole']=output
        options[names,'patch']=output[yy.min():yy.max()+1,xx.min():xx.max()+1]
    return options


def rules(task):
    demos=task['train']
    xs=[np.array(e['input']) for e in demos+task['test']]
    ys=[np.array(e['output']) for e in demos]
    common=sorted(set.intersection(*(set(g.ravel()) for g in xs[:len(demos)])))
    for ink in common:
        if any(x.shape==y.shape and (np.any((x!=ink)&(x!=y)) or np.array_equal(x,y)) for x,y in zip(xs,ys)):
            continue
        frames=[variants(g,ink) for g in xs]
        keys=set.intersection(*(set(f) for f in frames))
        for k in sorted(keys):
            if all(np.array_equal(f[k],y) for f,y in zip(frames,ys)):
                yield 2+len(k[0]),[f[k].tolist() for f in frames[len(demos):]]
