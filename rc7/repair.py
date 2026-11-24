"""Fit the occluding color, symmetry or period, and the required output region."""
import numpy as np
from itertools import product

def restore(a,c,mode):
    mask=a==c
    if not np.any(mask):return None
    if mode=='period':
        # Search candidate periods by motif area; reject contradictory observations.
        h,w=a.shape
        y,x=np.indices(a.shape)
        seen=~mask
        for dy,dx in sorted(product(range(1,h+1),range(1,w+1)),key=lambda p:p[0]*p[1]):
            if dy*dx>seen.sum():break
            ids=(y%dy)*dx+x%dx
            lut=np.full(dy*dx,-1)
            lut[ids[seen]]=a[seen]
            if np.array_equal(lut[ids[seen]],a[seen]) and (lut[ids[mask]]>=0).all():
                return lut[ids]
        return None
    b=a.copy()
    for kind in mode:
        z={'x':b[:,::-1],'y':b[::-1],'r':b[::-1,::-1],'d':b.T}[kind]
        if z.shape!=b.shape:return None
        b=np.where(b==c,z,b)
    return b if not np.any(b==c) else None

def repairs(xs,ys,qs):
    for patch in (False,True):
        for c in np.unique(xs[0].grid):
            def region(a):
                y,x=np.where(a==c)
                if not len(y):return None
                return np.s_[y.min():y.max()+1,x.min():x.max()+1] if patch else np.s_[:,:]
            if any(region(s.grid) is None or s.grid[region(s.grid)].shape!=y.shape for s,y in zip(xs,ys)):continue
            if any(np.any(y==c) for y in ys):continue
            for mode in ('x','y','r','d','xy','dx','dy','period'):
                def run(s):
                    box=region(s.grid)
                    if box is None:return None
                    b=restore(s.grid,c,mode)
                    return None if b is None else b[box]
                if all(np.array_equal(run(s),y) for s,y in zip(xs,ys)):
                    out=[run(q) for q in qs]
                    if all(v is not None for v in out):
                        yield 1+int(patch),['repair',int(c),mode,patch],out
