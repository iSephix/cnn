"""Learn conditional rendering of object masks with exact Boolean coverage."""
from itertools import product

import numpy as np
from scipy import ndimage

from relations import scene


def measures(grid, split):
    bg,objects=scene(grid,split)
    if not objects or len(objects)>70:
        return bg,{}
    features=[]
    yy,xx=np.indices(grid.shape)
    for i,(color,coords,lo,hi) in enumerate(objects):
        mask=np.zeros(grid.shape,bool)
        mask[tuple(coords.T)]=True
        h,w=hi-lo+1
        near=ndimage.binary_dilation(mask)
        outside=near & ~mask
        touched=tuple(sorted(set(grid[outside])-{bg,color}))
        box=(yy>=lo[0])&(yy<=hi[0])&(xx>=lo[1])&(xx<=hi[1])
        edges=box&((yy==lo[0])|(yy==hi[0])|(xx==lo[1])|(xx==hi[1]))
        region=mask[lo[0]:hi[0]+1,lo[1]:hi[1]+1]
        values={'area':len(coords),'height':h,'width':w,'color':color,'density':len(coords)/(h*w),
                'top':lo[0],'left':lo[1],'bottom':hi[0],'right':hi[1],
                'shape':(region.shape,region.tobytes()),
                'contacts':len(touched),'aspect':int(np.sign(h-w))}
        stamps={'mask':mask,'box':box,'edge':edges,'interior':box&~edges,'holes':ndimage.binary_fill_holes(mask)&~mask,
                'around':outside,'corners':edges&((yy==lo[0])|(yy==hi[0]))&((xx==lo[1])|(xx==hi[1])),
                'diagonal':box&((yy-lo[0]==xx-lo[1])|(yy-lo[0]==hi[1]-xx))}
        features.append((values,stamps))
    conditions={('all',):list(range(len(features)))}
    for feature in features[0][0]:
        values=[a[feature] for a,_ in features]
        for val in set(values):
            if feature in ('color','aspect','contacts'):
                conditions[feature,val]=[i for i,v in enumerate(values) if v==val]
        if feature=='shape':
            counts=[values.count(v) for v in values]
            conditions[feature,'unique']=[i for i,c in enumerate(counts) if c==1]
            conditions[feature,'repeated']=[i for i,c in enumerate(counts) if c>1]
        elif feature != 'color':
            for rank,choose in [('min',min),('max',max)]:
                extreme=choose(values)
                conditions[feature,rank]=[i for i,v in enumerate(values) if v==extreme]
    masks={}
    for key,ids in conditions.items():
        if not ids:
            continue
        for kind in features[0][1]:
            masks[key,kind]=np.logical_or.reduce([features[i][1][kind] for i in ids])
    return bg,masks


def rules(task):
    demos=task['train']
    grids=[np.array(e['input']) for e in demos+task['test']]
    targets=[np.array(e['output']) for e in demos]
    n=len(targets)
    if any(g.shape!=y.shape for g,y in zip(grids,targets)):
        return
    changes=[g!=y for g,y in zip(grids,targets)]
    paints=set.union(*(set(y[m]) for y,m in zip(targets,changes)))
    if not 1<=len(paints)<=3:
        return
    for split in (False,True):
        frames=[measures(g,split) for g in grids]
        keys=set.union(*(set(m) for _,m in frames[:n]))
        keys=sorted(keys,key=lambda k:(len(k[0]),repr(k)))
        empty=[np.zeros(g.shape,bool) for g in grids]
        variants={}
        for paint in sorted(paints):
            changes_color=[m&(y==paint) for m,y in zip(changes,targets)]
            choices=[]
            seen=set()
            for key,only in product(keys,(False,True)):
                rendered=[rules.get(key,z)&(g==bg) if only else rules.get(key,z)
                          for g,(bg,rules),z in zip(grids,frames,empty)]
                if any(np.any(m&~r) or np.any(y[r]!=paint)
                       for m,r,y in zip(changes_color,rendered,targets)):
                    continue
                signature=tuple(r.tobytes() for r in rendered[n:])
                if signature in seen:
                    continue
                seen.add(signature)
                choices.append((len(key[0])+int(only),rendered[n:]))
                if len(choices)>=4:
                    break
            if not choices:
                break
            variants[paint]=choices
        else:
            for programs in product(*variants.values()):
                outputs=[g.copy() for g in grids[n:]]
                for paint,(_,masks) in zip(variants,programs):
                    for out,mask in zip(outputs,masks):
                        out[mask]=paint
                yield sum(cost for cost,_ in programs)+int(split),[out.tolist() for out in outputs]
