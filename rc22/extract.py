"""Find which scene object corresponds to a smaller demonstrated output."""
from collections import Counter
from itertools import product

import numpy as np
from scipy import ndimage

from correspondence import local, objects, shape_key


def parse(grid,separated,diagonal):
    bg=int(np.bincount(grid.ravel()).argmax())
    entries=objects(grid,separated,diagonal)
    if len(entries)>36:
        return {}
    shapes=[]
    for item in entries:
        points=item['points']
        shapes.append(min(shape_key(local(points,k)) for k in range(8)))
    repeats=Counter(shapes)
    selected={}
    for i,item in enumerate(entries):
        points=item['points']
        lo,hi=points.min(axis=0),points.max(axis=0)
        mask=np.zeros(tuple(hi-lo+1),bool)
        mask[tuple((points-lo).T)]=True
        area=int(mask.sum())
        box=grid[lo[0]:hi[0]+1,lo[1]:hi[1]+1]
        distances=sorted(float(np.abs(item['center']-other['center']).sum())
                         for other in entries if other is not item)
        marker_distances=[float(np.abs(item['center']-other['center']).sum())
                          for other in entries if other is not item and other['area']==1]
        attrs=dict(area=area,holes=np.count_nonzero(ndimage.binary_fill_holes(mask)&~mask),
                   repeats=repeats[shapes[i]],density=area/mask.size,palette=np.unique(box[mask]).size,
                   height=mask.shape[0],width=mask.shape[1],top=int(lo[0]),left=int(lo[1]),
                   bottom=int(grid.shape[0]-1-hi[0]),right=int(grid.shape[1]-1-hi[1]),
                   nearest=distances[0] if distances else 0.,
                   marker=min(marker_distances,default=float('inf')))
        selected[i]=(bg,item['color'],box,mask,attrs)
    return selected


def choose(items):
    family={}
    if not items:
        return family
    fields=list(next(iter(items.values()))[-1])
    for attr in fields:
        for direction,compare in [('min',min),('max',max)]:
            best=compare(row[-1][attr] for row in items.values())
            tied=[i for i,row in items.items() if row[-1][attr]==best]
            if len(tied)==1:
                family[attr,direction]=items[tied[0]]
    counts=Counter(row[1] for row in items.values())
    for row in items.values():
        if counts[row[1]]==1:
            family['color',row[1]]=row
    if len(items)==1:
        family['only',]=next(iter(items.values()))
    return family


def patch(item,style,turn):
    bg,_,grid,mask,_=item
    if style=='isolated':
        grid=np.where(mask,grid,bg)
    return np.rot90(grid if turn<4 else grid[:,::-1],turn%4)


def learn_map(views,targets):
    lut=np.full(10,-1,dtype=int)
    for view,target in zip(views,targets):
        if view.shape!=target.shape:
            return
        previous=lut.copy()
        lut[view.ravel()]=target.ravel()
        if np.any((previous>=0)&(previous!=lut)) or np.any(lut[view]!=target):
            return
    return lut


def extract(task):
    demos=task['train']
    n=len(demos)
    images=[np.array(d['input']) for d in demos+task['test']]
    targets=[np.array(d['output']) for d in demos]
    if any(x.size<=y.size for x,y in zip(images,targets)):
        return
    seen=set()
    for mono,diag in product((True,False),(1,2)):
        scenes=[choose(parse(g,mono,diag)) for g in images]
        keys=set.intersection(*(set(sc) for sc in scenes))
        for key in sorted(keys,key=repr):
            for style,turn in product(('isolated','crop'),range(8)):
                windows=[patch(s[key],style,turn) for s in scenes]
                lut=learn_map(windows[:n],targets)
                if lut is None:
                    continue
                out=[]
                for view in windows[n:]:
                    if np.any(lut[view]<0):
                        break
                    out.append(lut[view].tolist())
                else:
                    sig=tuple(tuple(map(tuple,g)) for g in out)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    cost=4+int(turn!=0)+int(style=='isolated')+int(np.sum((lut>=0)&(lut!=np.arange(10))))
                    trace=dict(exact=True,extraction=dict(selection=key,orientation=turn,style=style,
                            color_components=mono,diagonal_connectivity=diag==2,color_map=lut.tolist()),
                        demonstrations=[dict(origin=[int(scene[key][-1]['top']),int(scene[key][-1]['left'])],
                                             shape=list(window.shape)) for scene,window in zip(scenes[:n],windows[:n])])
                    yield cost,out,trace
