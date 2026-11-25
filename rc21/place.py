"""Move or copy objects by aligning their edges to marker objects."""
from itertools import product
import numpy as np

from bind import objects


def transform(grid, parts, mover, reference, direction, align, mode):
    bg=int(np.bincount(grid.ravel()).argmax())
    out=grid.copy()
    sources=[item for item in parts if item['color']==mover]
    anchors=[item for item in parts if item['color']==reference]
    if not sources or not anchors:
        return
    placements=[]
    for source in sources:
        found=[]
        for anchor in anchors:
            delta=anchor['center']-source['center']
            if direction=='axial' and np.min(np.abs(delta))>=1:
                continue
            found.append((float(np.abs(delta).sum()),anchor))
        if not found:
            continue
        anchor=min(found,key=lambda rec:rec[0])[1]
        loc=source['points']
        other=anchor['points']
        lo,hi=loc.min(axis=0),loc.max(axis=0)
        lower,upper=other.min(axis=0),other.max(axis=0)
        delta=anchor['center']-source['center']
        axis=int(np.argmax(np.abs(delta)))
        shift=((lower+upper)-(lo+hi))//2 if align=='center' else lower-lo if align=='start' else upper-hi
        shift[axis]=lower[axis]-hi[axis]-1 if delta[axis]>=0 else upper[axis]-lo[axis]+1
        dest=loc+shift
        if np.any(dest<0) or np.any(dest>=grid.shape):
            return
        placements.append((loc,dest))
    if not placements:
        return
    if mode=='move':
        for loc,_ in placements:
            out[tuple(loc.T)]=bg
    for loc,dest in placements:
        out[tuple(dest.T)]=grid[tuple(loc.T)]
    return out


def rules(task):
    demos=task['train']
    n=len(demos)
    raw=[np.array(e['input']) for e in demos+task['test']]
    targets=[np.array(e['output']) for e in demos]
    if any(g.shape!=y.shape or set(np.unique(y))!=set(np.unique(g)) for g,y in zip(raw,targets)):
        return
    palettes=[set(g.ravel()) for g in raw[:n]]
    colors=sorted(set.intersection(*palettes))
    for diag in (1,2):
        scenes=[objects(g,True,diag) for g in raw]
        if any(not 2<=len(s)<=32 for s in scenes):
            continue
        for mover,anchor in product(colors,repeat=2):
            if mover==anchor:continue
            if not any(np.any((y==mover)&(g!=mover)) for g,y in zip(raw,targets)):
                continue
            for spec in product(('any','axial'),('center','start','end'),('move','copy')):
                if not all(np.array_equal(transform(g,s,mover,anchor,*spec),y)
                           for g,s,y in zip(raw,scenes,targets)):
                    continue
                outs=[transform(g,s,mover,anchor,*spec) for g,s in zip(raw[n:],scenes[n:])]
                if all(o is not None for o in outs):
                    yield 4+int(spec[0]!='any')+int(spec[1]!='center'),[o.tolist() for o in outs]
