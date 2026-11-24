"""Split larger grids into aligned panels, then induce a cell-wise composition."""
from itertools import product
import numpy as np


def split(g,rows,cols,dy,dx):
    h,w=g.shape
    h,hr=divmod(h-(rows-1)*dy,rows)
    w,wr=divmod(w-(cols-1)*dx,cols)
    if hr or wr or min(h,w)<1:
        return None
    return np.stack([g[i*(h+dy):i*(h+dy)+h,j*(w+dx):j*(w+dx)+w]
                     for i in range(rows) for j in range(cols)])


def panels(xs,ys,qs):
    for rows,cols in product(range(1,6),repeat=2):
        if not 2<=rows*cols<=9:
            continue
        dy=0 if rows==1 else (xs[0].grid.shape[0]-rows*ys[0].shape[0])//(rows-1)
        dx=0 if cols==1 else (xs[0].grid.shape[1]-cols*ys[0].shape[1])//(cols-1)
        if not(0<=dx<=2 and 0<=dy<=2):
            continue
        def get(s):
            return split(s.grid,rows,cols,dy,dx)
        train=[get(s) for s in xs]
        if any(v is None or v.shape[1:]!=y.shape for v,y in zip(train,ys)):
            continue
        queries=[get(q) for q in qs]
        if any(v is None for v in queries):
            continue
        for bg in (0,'mode','panel'):
            def background(s,v):
                if bg=='panel':
                    return np.array([np.bincount(p.ravel()).argmax() for p in v])[:,None,None]
                return s.bg if bg=='mode' else bg
            def encode(s,v,kind):
                mask=v!=background(s,v)
                if kind=='count':
                    return mask.sum(axis=0)
                if kind=='mask':
                    return np.tensordot(2**np.arange(len(v)),mask,axes=1)
                return np.tensordot(10**np.arange(len(v)),v,axes=1)
            for kind in ('count','mask','colors'):
                mapping={}
                for s,v,y in zip(xs,train,ys):
                    encoded=encode(s,v,kind)
                    for k in np.unique(encoded):
                        vals=y[encoded==k]
                        if np.any(vals!=vals[0]) or mapping.get(int(k),vals[0])!=vals[0]:
                            break
                        mapping[int(k)]=int(vals[0])
                    else:
                        continue
                    break
                else:
                    outs=[]
                    for q,v in zip(qs,queries):
                        enc=encode(q,v,kind)
                        if any(int(k) not in mapping for k in np.unique(enc)):
                            break
                        outs.append(np.vectorize(mapping.__getitem__,otypes=[int])(enc))
                    if len(outs)==len(qs):
                        yield 3+len(mapping),('panels',rows,cols,dy,dx,bg,kind),outs
