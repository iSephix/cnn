"""Use object detection to learn layout, ordering, and alignment of the output scene."""
from itertools import product
import numpy as np

from compose import relations


def patches(grid, split, group):
    bg,objects=relations.scene(grid,split)
    if not 2<=len(objects)<=18:
        return bg,[]
    result=[]
    for color,coords,lo,hi in objects:
        if group=='crop':
            patch=grid[lo[0]:hi[0]+1,lo[1]:hi[1]+1].copy()
        else:
            patch=np.full(tuple(hi-lo+1),bg)
            patch[tuple((coords-lo).T)]=color
        result.append((color,patch,len(coords),int(lo[0]),int(lo[1])))
    return bg,result


def place(scene,axis,spacing,align,order,reverse,fill):
    bg,items=scene
    if not items:
        return
    key={
        'position':lambda r:(r[3],r[4]),'column':lambda r:(r[4],r[3]),
        'color':lambda r:r[0],'area':lambda r:r[2],'height':lambda r:r[1].shape[0],
        'width':lambda r:r[1].shape[1],
    }[order]
    sorted_items=[r[1] if axis==1 else r[1].T for r in sorted(items,key=key,reverse=reverse)]
    h=max(g.shape[0] for g in sorted_items)
    w=sum(g.shape[1] for g in sorted_items)+spacing*(len(sorted_items)-1)
    if max(h,w)>30:
        return
    fill=bg if fill=='scene' else 0
    grid=np.full((h,w),fill)
    start=0
    for patch in sorted_items:
        ph,pw=patch.shape
        y=(h-ph)*align//2
        grid[y:y+ph,start:start+pw]=np.where(patch==bg,fill,patch)
        start+=pw+spacing
    return grid if axis==1 else grid.T


def rules(task):
    examples=task['train']
    raw=[np.array(e['input']) for e in examples+task['test']]
    targets=[np.array(e['output']) for e in examples]
    if all(np.array_equal(x,y) for x,y in zip(raw,targets)):
        return
    for split,group in product((False,True),('mask','crop')):
        scenes=[patches(g,split,group) for g in raw]
        if any(not a for _,a in scenes):
            continue
        for axis,spacing in product((0,1),(0,1,2)):
            if not all(sum(item[1].shape[axis] for item in items)+spacing*(len(items)-1)==target.shape[axis]
                       and max(item[1].shape[1-axis] for item in items)==target.shape[1-axis]
                       for (_,items),target in zip(scenes,targets)):
                continue
            for align,order,reverse,fill in product((0,1,2),('position','column','color','area','height','width'),
                                                   (False,True),('scene','zero')):
                spec=(axis,spacing,align,order,reverse,fill)
                if not all(np.array_equal(place(sc,*spec),target) for sc,target in zip(scenes,targets)):
                    continue
                outputs=[place(sc,*spec) for sc in scenes[len(examples):]]
                if all(out is not None for out in outputs):
                    yield 3+int(split)+int(spacing!=0)+int(reverse),[out.tolist() for out in outputs]
