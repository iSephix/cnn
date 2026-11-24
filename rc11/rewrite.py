"""Learn a shared edit program from object-centered examples inside each demonstration."""
from itertools import product
import numpy as np
from scipy import ndimage as nd
from regions import partition
from views import transform


def coordinates(r,c,h,w,k):
    if k&4:
        r,c,h,w = c,r,w,h
    return (h-1-r if k&1 else r),(w-1-c if k&2 else c)


def describe(g,idx,normalize):
    bg = int(np.bincount(g.ravel()).argmax())
    if not idx.any():
        return []
    nearest = nd.distance_transform_edt(idx==0,return_distances=False,return_indices=True)
    owner = idx[tuple(nearest)]
    items = []
    for i,(rows,cols) in enumerate(nd.find_objects(idx),1):
        mask = idx[rows,cols]==i
        colors = np.bincount(g[rows,cols][mask],minlength=10)
        main = int(colors.argmax())
        choices = range(8) if normalize else (0,)
        def signature(k):
            m = transform(mask,k)
            return m.shape,m.tobytes()
        k = min(choices,key=signature)
        h,w = mask.shape
        (H,W),bits = signature(k)
        shape = (H,W,bits.hex())
        areas = int(mask.sum())
        keys = ('any',main,areas,(H,W),shape,(shape,main),(areas,main),len(np.flatnonzero(colors)))
        items.append((keys,k,rows.start,cols.start,h,w,owner==i,main,bg))
    return items


def stencil(item,g,y,anchor):
    _,k,top,left,h,w,territory,main,bg = item
    yy,xx = np.where(territory & (g!=y))
    r,c = coordinates(yy-top,xx-left,h,w,k)
    H,W = (w,h) if k&4 else (h,w)
    if anchor=='center':
        r,c = r-(H-1)//2,c-(W-1)//2
    elif anchor=='end':
        r,c = r-H+1,c-W+1
    vals = y[yy,xx]
    codes = np.where(vals==main,10,np.where(vals==bg,11,vals))
    return tuple(sorted(zip(map(int,r),map(int,c),map(int,codes))))


def draw(g,items,table,field,anchor):
    result = g.copy()
    for item in items:
        keys,k,top,left,h,w,_,main,bg = item
        steps = table.get(keys[field])
        if steps is None:
            return
        H,W = (w,h) if k&4 else (h,w)
        for r,c,code in steps:
            if anchor=='center':
                r,c = r+(H-1)//2,c+(W-1)//2
            elif anchor=='end':
                r,c = r+H-1,c+W-1
            if k&1:
                r = H-1-r
            if k&2:
                c = W-1-c
            if k&4:
                r,c = c,r
            r,c = r+top,c+left
            if 0<=r<len(g) and 0<=c<g.shape[1]:
                result[r,c] = main if code==10 else bg if code==11 else code
    return result


def infer(task):
    demos = task['train']
    if any(np.shape(e['input'])!=np.shape(e['output']) for e in demos):
        return
    grids = [np.array(e['input']) for e in demos+task['test']]
    ys = [np.asarray(e['output']) for e in demos]
    n = len(demos)
    seen = set()
    for mono,diag in product((True,False),(False,True)):
        idx = [partition(g,mono,diag) for g in grids]
        sig = tuple(a.tobytes() for a in idx)
        if sig in seen or not all(a.any() for a in idx):
            continue
        seen.add(sig)
        for normalize in (False,True):
            groups = [describe(g,a,normalize) for g,a in zip(grids,idx)]
            for anchor in ('start','center','end'):
                rows = [(item[0],stencil(item,g,y,anchor),i) for i,(group,g,y) in
                        enumerate(zip(groups,grids,ys)) for item in group]
                for field in range(8):
                    table,support = {},{}
                    for keys,steps,i in rows:
                        key = keys[field]
                        if table.setdefault(key,steps)!=steps:
                            break
                        support.setdefault(key,set()).add(i)
                    else:
                        if not any(table.values()):
                            continue
                        predicted = [draw(g,grp,table,field,anchor) for g,grp in zip(grids,groups)]
                        if any(p is None for p in predicted) or not all(np.array_equal(p,y) for p,y in zip(predicted,ys)):
                            continue
                        weak = sum(len(v)<2 for v in support.values())
                        cost = (weak,len(table),sum(len(v) for v in table.values()),field)
                        trace = ['rewrite',mono,diag,normalize,anchor,field,list(table.items())]
                        yield cost,trace,[p.tolist() for p in predicted[n:]]
