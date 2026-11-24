"""Compile an ordered object catalog into a sequence of sprites or a bar chart."""
from itertools import product
import numpy as np
from scipy import ndimage as nd


def catalog(g,bg,kind):
    colors = np.unique(g[g!=bg])
    groups = []
    if kind=='palette':
        for c in colors:
            groups.append(np.where(g==c))
    else:
        masks = [g==c for c in colors] if kind.startswith('mono') else [g!=bg]
        for mask in masks:
            ids,_ = nd.label(mask,nd.generate_binary_structure(2,2 if kind.endswith('8') else 1))
            for i,box in enumerate(nd.find_objects(ids),1):
                y,x = np.where(ids[box]==i)
                groups.append((y+box[0].start,x+box[1].start))
    objects = []
    for y,x in groups:
        top,left = y.min(),x.min()
        p = g[top:y.max()+1,left:x.max()+1]
        patch = np.full_like(p,bg)
        patch[y-top,x-left] = g[y,x]
        h,w = p.shape
        objects.append(dict(top=int(top),left=int(left),area=len(y),height=h,width=w,
                            color=int(np.bincount(g[y,x]).argmax()),patch=patch))
    return objects


def render(objects,bg,rule):
    order,reverse,kind,axis,align = rule
    sequence = sorted(objects,key=lambda o:(o[order],o['top'],o['left']),reverse=reverse)
    if kind=='sprites':
        patches = [o['patch'].T if axis else o['patch'] for o in sequence]
        h,w = max(p.shape[0] for p in patches),sum(p.shape[1] for p in patches)
        if max(h,w)>30:
            return
        result = np.full((h,w),bg)
        x = 0
        for p in patches:
            y = (h-len(p))*align
            result[y:y+len(p),x:x+p.shape[1]] = p
            x += p.shape[1]
    else:
        values = [1 if kind=='unit' else o[kind] for o in sequence]
        h,w = max(values),len(values)
        if max(h,w)>30:
            return
        result = np.full((h,w),bg)
        for x,(v,o) in enumerate(zip(values,sequence)):
            y = (h-v)*align
            result[y:y+v,x] = o['color']
    return result.T if axis else result


def infer(task):
    n = len(task['train'])
    grids = [np.asarray(e['input']) for e in task['train']+task['test']]
    targets = [np.asarray(e['output']) for e in task['train']]
    if all(g.shape==y.shape for g,y in zip(grids,targets)):
        return
    for bg_name in ('zero','mode'):
        bgs = [0 if bg_name=='zero' else int(np.bincount(g.ravel()).argmax()) for g in grids]
        for source in ('palette','mono4','multi4','mono8','multi8'):
            groups = [catalog(g,b,source) for g,b in zip(grids,bgs)]
            if any(not obs or len(obs)>30 for obs in groups):
                continue
            for kind,axis in product(('unit','area','height','width','sprites'),(0,1)):
                # Dimensions do not depend on sorting or alignment.
                def trial(obs,bg):
                    return render(obs,bg,('top',False,kind,axis,0))
                a = trial(groups[0],bgs[0])
                if a is None or a.shape!=targets[0].shape:
                    continue
                for order,reverse,align in product(('top','left','color','area','height','width'),(False,True),(0,1)):
                    rule = (order,reverse,kind,axis,align)
                    if not all(np.array_equal(render(obs,b,rule),y) for obs,b,y in zip(groups,bgs,targets)):
                        continue
                    outs = [render(obs,b,rule) for obs,b in zip(groups[n:],bgs[n:])]
                    if all(o is not None for o in outs):
                        yield int(reverse)+align,['catalog',bg_name,source,rule],[o.tolist() for o in outs]
