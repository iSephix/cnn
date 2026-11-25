"""Transfer the shape of one object to marker positions elsewhere in the grid."""
from itertools import product
import numpy as np
from scipy import ndimage


def scene(grid, diagonal):
    bg=int(np.bincount(grid.ravel()).argmax())
    labels,count=ndimage.label(grid!=bg,ndimage.generate_binary_structure(2,diagonal))
    items=[]
    for index,box in enumerate(ndimage.find_objects(labels),1):
        patch=grid[box]
        active=labels[box]==index
        counts=np.bincount(patch[active],minlength=10)
        top=(box[0].start,box[1].start)
        items.append((top,patch,active,counts))
    return bg,items


def template(items, style):
    if style=='largest':
        return max(items,key=lambda o:o[2].sum())
    if style=='palette':
        return max(items,key=lambda o:(np.count_nonzero(o[3]),o[2].sum()))
    return max(items,key=lambda o:o[2].size)


def render(grid,sc,style,anchor,tone,keep,clip):
    bg,items=sc
    if not 2<=len(items)<=24:
        return
    donor=template(items,style)
    (_,patch,active,colors)=donor
    if active.sum()<2:
        return
    out=grid.copy() if keep else np.full_like(grid,bg)
    for item in items:
        if item is donor:
            continue
        (y,x),_,marks,hist=item
        if marks.sum()!=1:
            continue
        color=int(hist.argmax())
        if anchor=='color':
            hits=np.argwhere(active&(patch==color))
            if len(hits)!=1:
                continue
            ry,rx=hits[0]
        elif anchor=='center':
            ry,rx=np.array(patch.shape)//2
        else:
            ry,rx=0,0
        yy,xx=np.where(active)
        dy,dx=yy+y-ry,xx+x-rx
        valid=(dy>=0)&(dy<grid.shape[0])&(dx>=0)&(dx<grid.shape[1])
        if not clip and not valid.all():
            return
        dy,dx=dy[valid],dx[valid]
        values=patch[yy[valid],xx[valid]] if tone=='copy' else color
        out[dy,dx]=values
    return out


def rules(task):
    demos=task['train']
    n=len(demos)
    raw=[np.array(e['input']) for e in demos+task['test']]
    targets=[np.array(e['output']) for e in demos]
    if any(g.shape!=y.shape for g,y in zip(raw,targets)) or all(np.array_equal(g,y) for g,y in zip(raw,targets)):
        return
    for diag in (1,2):
        scenes=[scene(g,diag) for g in raw]
        if any(not 2<=len(s[1])<=24 for s in scenes):
            continue
        for spec in product(('largest','palette','extent'),('color','center','corner'),
                            ('copy','marker'),(True,False),(False,True)):
            if not all(np.array_equal(render(g,s,*spec),y) for g,s,y in zip(raw,scenes,targets)):
                continue
            outputs=[render(g,s,*spec) for g,s in zip(raw[n:],scenes[n:])]
            if all(g is not None for g in outputs):
                yield 3+int(spec[1]!='color')+int(spec[-1]),[g.tolist() for g in outputs]
