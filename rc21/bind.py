"""Bind an object's paint to another object selected by spatial or shape relations."""
from collections import Counter
from itertools import product
import numpy as np
from scipy import ndimage


def objects(g, separate, diag):
    bg=int(np.bincount(g.ravel()).argmax())
    masks=[g==c for c in np.unique(g) if c!=bg] if separate else [g!=bg]
    result=[]
    for mask in masks:
        labels,_=ndimage.label(mask,ndimage.generate_binary_structure(2,diag))
        for i,box in enumerate(ndimage.find_objects(labels),1):
            cells=labels[box]==i
            points=np.argwhere(cells)
            top=np.array([s.start for s in box])
            count=np.bincount(g[box][cells],minlength=10)
            result.append(dict(points=points+top,center=points.mean(axis=0)+top,area=len(points),
                               shape=(cells.shape,cells.tobytes()),color=int(count.argmax())))
    return result


def canvas(grid, parts, selector, donor, relative, metric, background):
    out=grid.copy()
    if not parts or len(parts)>45:
        return
    areas=[p['area'] for p in parts]
    counts=Counter(p['color'] for p in parts)
    for i,obj in enumerate(parts):
        ink=obj['color']
        if isinstance(selector,int) and ink!=selector:
            continue
        if selector=='largest' and obj['area']!=max(areas):
            continue
        if selector=='smallest' and obj['area']!=min(areas):
            continue
        if selector=='common' and counts[ink]!=max(counts.values()):
            continue
        matches=[]
        for j,other in enumerate(parts):
            if i==j or other['color']==ink:
                continue
            if donor=='marker' and other['area']!=1:
                continue
            if donor=='shape' and other['shape']!=obj['shape']:
                continue
            if donor=='larger' and other['area']<=obj['area']:
                continue
            delta=other['center']-obj['center']
            if relative=='row' and abs(delta[0])>=1:
                continue
            if relative=='column' and abs(delta[1])>=1:
                continue
            if relative=='above' and delta[0]>=0:
                continue
            if relative=='left' and delta[1]>=0:
                continue
            distance=np.abs(delta).sum() if metric=='nearest' else -np.abs(delta).sum()
            matches.append((distance,j,other))
        if matches:
            hit=min(matches)[2]
            out[tuple(obj['points'].T)]=hit['color']
            if background:
                out[tuple(hit['points'].T)]=np.bincount(grid.ravel()).argmax()
    return out


def rules(task):
    demos=task['train']
    n=len(demos)
    inputs=[np.array(e['input']) for e in demos+task['test']]
    targets=[np.array(e['output']) for e in demos]
    if any(g.shape!=y.shape for g,y in zip(inputs,targets)):
        return
    if any(not set(y.ravel())<=set(g.ravel()) for g,y in zip(inputs,targets)):
        return
    changed=[g!=y for g,y in zip(inputs,targets)]
    if not any(m.any() for m in changed):
        return
    colors=sorted(set.intersection(*(set(g[m]) for g,m in zip(inputs,changed))))
    for mono,diag in product((True,False),(1,2)):
        parts=[objects(g,mono,diag) for g in inputs]
        if any(not 2<=len(p)<=45 for p in parts):
            continue
        selectors=('all','largest','smallest','common',*(int(c) for c in colors))
        # The renderer only paints existing foreground pixels.
        if any(np.any(m&(g==np.bincount(g.ravel()).argmax())) for g,m in zip(inputs,changed)):
            continue
        for spec in product(selectors,('any','marker','shape','larger'),
                            ('any','row','column','above','left'),('nearest','farthest'),(False,True)):
            if not all(np.array_equal(canvas(g,p,*spec),y) for g,p,y in zip(inputs,parts,targets)):
                continue
            outputs=[canvas(g,p,*spec) for g,p in zip(inputs[n:],parts[n:])]
            if all(out is not None for out in outputs):
                yield 3+int(spec[2]!='any')+int(spec[3]=='farthest')+int(spec[4]),[out.tolist() for out in outputs]
