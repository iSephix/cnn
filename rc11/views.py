"""Generate candidate intermediate scenes from a single observed grid."""
from itertools import product
import numpy as np
from scipy import ndimage as nd


def crop(g, mask):
    r,c = np.where(mask)
    if len(r):
        return g[r.min():r.max()+1,c.min():c.max()+1]


def transform(g,k):
    view = g.T if k&4 else g
    if k&1:
        view = view[::-1]
    if k&2:
        view = view[:,::-1]
    return view


def inspect(g):
    bg = int(np.bincount(g.ravel()).argmax())
    yield ('whole',),g,g
    def emit(name,mask):
        patch = crop(g,mask)
        if patch is not None:
            yield (name,'crop'),patch,patch
            selected = np.where(mask,g,bg)
            yield (name,'isolate'),selected,g
            yield (name,'boolean'),mask.astype(int),g
    for name,mask in [('ink',g!=bg),*[(int(c),g==c) for c in np.unique(g)]]:
        yield from emit(name,mask)
    for axis in (0,1):
        a = g if axis==0 else g.T
        constant = (a==a[:,:1]).all(axis=1)
        groups = np.split(np.flatnonzero(~constant),np.flatnonzero(np.diff(np.flatnonzero(~constant))>1)+1)
        for i,rows in enumerate(groups):
            if len(rows) and len(rows)!=len(a):
                p = a[rows] if axis==0 else a[rows].T
                yield ('panel',axis,i),p,p
        for count in (2,3,4):
            size,rem = divmod(len(a),count)
            if size and rem in (0,count-1):
                blocks = [a[i*(size+bool(rem)):i*(size+bool(rem))+size] for i in range(count)]
                for i,v in enumerate(blocks):
                    p = v if axis==0 else v.T
                    yield ('part',axis,count,i),p,p
                for i,j in ((0,1),(0,count-1)):
                    p,q = (blocks[i],blocks[j]) if axis==0 else (blocks[i].T,blocks[j].T)
                    yield ('overlay',axis,count,i,j),p,q
    for diag in (False,True):
        for mono in (False,True):
            objects = []
            for mask in ([g==c for c in np.unique(g) if c!=bg] if mono else [g!=bg]):
                labels,_ = nd.label(mask,nd.generate_binary_structure(2,2 if diag else 1))
                for i,box in enumerate(nd.find_objects(labels),1):
                    cells = labels[box]==i
                    rgb = g[box][cells]
                    color = int(np.bincount(rgb).argmax())
                    objects.append((box,cells,color))
            if not objects:
                continue
            areas = [int(m.sum()) for _,m,_ in objects]
            for typ,idx in [('largest',int(np.argmax(areas))),('smallest',int(np.argmin(areas)))]:
                box,m,c = objects[idx]
                p = g[box]
                yield (typ,diag,mono),p,p
                v = np.where(m,p,bg)
                yield (typ,diag,mono,'masked'),v,p
            for style in ('rect','interior','outline','holes','center','area-rank'):
                out = np.full_like(g,bg) if style!='area-rank' else np.zeros_like(g)
                ranks = {v:min(9,i+1) for i,v in enumerate(sorted(set(areas)))}
                for (box,m,c),area in zip(objects,areas):
                    target = out[box]
                    if style=='rect':
                        target[:] = c
                    elif style=='interior':
                        target[1:-1,1:-1] = c
                    elif style=='outline':
                        target[[0,-1],:] = c
                        target[:,[0,-1]] = c
                    elif style=='holes':
                        target[nd.binary_fill_holes(m)&~m] = c
                    elif style=='center':
                        target[(len(m)-1)//2,(m.shape[1]-1)//2] = c
                    else:
                        target[m] = ranks[area]
                yield (style,diag,mono),out,g
    if (g!=bg).any():
        idx = nd.distance_transform_edt(g==bg,return_distances=False,return_indices=True)
        yield ('nearest',),g[tuple(idx)],g
    for dy,dx in product((-1,0,1),repeat=2):
        if not (dy or dx):
            continue
        out = np.full_like(g,bg)
        for r in (range(len(g)) if dy>=0 else reversed(range(len(g)))):
            for c in (range(g.shape[1]) if dx>=0 else reversed(range(g.shape[1]))):
                if g[r,c]!=bg:
                    out[r,c] = g[r,c]
                elif 0<=r-dy<len(g) and 0<=c-dx<g.shape[1]:
                    out[r,c] = out[r-dy,c-dx]
        yield ('cast',dy,dx),out,g
