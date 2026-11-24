"""Learn block layouts and cell placement operators for shared output structures."""
from itertools import product
import numpy as np


def orient(g,k):
    view=g.T if k&4 else g
    if k&1:
        view=view[::-1]
    if k&2:
        view=view[:,::-1]
    return view


def layouts(xs,ys,qs):
    g=xs[0].grid
    h,w=g.shape
    rows,cols=ys[0].shape[0]//h,ys[0].shape[1]//w
    if not (1<rows*cols<=36 and rows and cols):
        return
    if any(y.shape!=(s.grid.shape[0]*rows,s.grid.shape[1]*cols) for s,y in zip(xs,ys)):
        return
    if any(q.grid.shape[0]*rows>30 or q.grid.shape[1]*cols>30 for q in qs):
        return
    # Tile codes 0..7 select a symmetry, -1 selects the scene background.
    codes=[]
    for i,j in product(range(rows),range(cols)):
        valid=[]
        for k in (*range(8),-1):
            for s,y in zip(xs,ys):
                a,b=s.grid.shape
                tile=y[i*a:(i+1)*a,j*b:(j+1)*b]
                if k==-1:
                    equal=(tile==s.bg).all()
                else:
                    equal=np.array_equal(orient(s.grid,k),tile)
                if not equal:
                    break
            else:
                valid.append(k)
        if not valid:
            return
        codes.append(valid[0])
    outs=[]
    for q in qs:
        blocks=[orient(q.grid,k) if k>=0 else np.full_like(q.grid,q.bg) for k in codes]
        if any(b.shape!=q.grid.shape for b in blocks):
            return
        outs.append(np.block([blocks[i*cols:(i+1)*cols] for i in range(rows)]))
    yield 3+len(set(codes)),('layout',rows,cols,codes),outs


def sweep(g,bg,axis,reverse,kind):
    a=(g if axis==0 else g.T).copy()
    if reverse:
        a=a[::-1]
    for j in range(a.shape[1]):
        line=a[:,j]
        indices=np.flatnonzero(line!=bg)
        if not len(indices):
            continue
        if kind=='pack':
            vals=line[indices].copy()
            line[:]=bg
            line[:len(vals)]=vals
        elif kind=='extend':
            c=bg
            for i,v in enumerate(line):
                if v!=bg:
                    c=v
                else:
                    line[i]=c
        else:
            for left,right in zip(indices,indices[1:]):
                if line[left]==line[right]:
                    line[left+1:right]=line[left]
    if reverse:
        a=a[::-1]
    return a if axis==0 else a.T


def sweeps(xs,ys,qs):
    if any(s.grid.shape!=y.shape for s,y in zip(xs,ys)):
        return
    for axes,rev,kind,bg in product(((0,),(1,),(0,1),(1,0)),(False,True),('pack','extend','connect'),('mode',0)):
        def run(s):
            g=s.grid
            for axis in axes:
                g=sweep(g,s.bg if bg=='mode' else bg,axis,rev,kind)
            return g
        if all(np.array_equal(run(s),y) for s,y in zip(xs,ys)):
            yield 2+len(axes),('sweep',axes,rev,kind,bg),[run(q) for q in qs]


def stamps(xs,ys,qs):
    """A source cell determines which transformed copy fills its output block."""
    if not all(y.shape==tuple(n*n for n in s.grid.shape) for s,y in zip(xs,ys)):
        return
    if any(max(q.grid.shape)>5 for q in qs) or all(s.grid.size==1 for s in xs):
        return
    for mask_fg,k,recolor in product((True,False),range(8),(False,True)):
        def run(s):
            template=orient(s.grid,k)
            if template.shape!=s.grid.shape:
                return None
            block=template[None,None,:,:]
            color=s.grid[:,:,None,None]
            ink=np.where(block!=s.bg,color,s.bg) if recolor else block
            enabled=(color!=s.bg) if mask_fg else (color==s.bg)
            tiles=np.where(enabled,ink,s.bg)
            return tiles.transpose(0,2,1,3).reshape(tuple(n*n for n in s.grid.shape))
        if all(np.array_equal(run(s),y) for s,y in zip(xs,ys)):
            outs=[run(q) for q in qs]
            if all(o is not None for o in outs):
                yield 3+recolor,('stamp',mask_fg,k,recolor),outs
