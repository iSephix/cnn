"""Fit independent translations of input colors under an exact scene reconstruction."""
from itertools import product
import numpy as np
from stencils import shift


def rules(task):
    inputs=[np.array(e['input']) for e in task['train']]
    outputs=[np.array(e['output']) for e in task['train']]
    if any(a.shape!=b.shape for a,b in zip(inputs,outputs)) or all(np.array_equal(a,b) for a,b in zip(inputs,outputs)):
        return
    queries=[np.array(e['input']) for e in task['test']]
    for bg in sorted({0,*[int(np.bincount(x.ravel()).argmax()) for x in inputs]}):
        colors=sorted(set.union(*(set(x.ravel()) for x in inputs))- {bg})
        if any(set(y.ravel())-{bg}-set(colors) for y in outputs):
            continue
        choices={}
        for c in colors:
            masks=[x==c for x in inputs]
            target=[y==c for y in outputs]
            source=next((i for i,m in enumerate(masks) if m.any()),None)
            if source is None:
                continue
            bound=np.argwhere(target[source])
            origin=np.argwhere(masks[source])
            if not len(bound) or len(bound)<len(origin):
                break
            lo=bound.min(axis=0)-origin.min(axis=0)
            hi=bound.max(axis=0)-origin.max(axis=0)
            offsets=set(product(*(sorted({int(a),int(b),0}) for a,b in zip(lo,hi))))
            options=[]
            for copy,(dy,dx) in product((False,True),sorted(offsets)):
                if copy and (dy,dx)==(0,0):
                    continue
                if all(np.array_equal(shift(m,dy,dx)|(m if copy else False),t) for m,t in zip(masks,target)):
                    options.append((copy,int(dy),int(dx)))
            if not options:
                break
            choices[c]=options[0]
        else:
            results=[]
            for g in queries:
                if set(g.ravel())-{bg}-set(choices):
                    break
                out=np.full_like(g,bg)
                for c,(copy,dy,dx) in choices.items():
                    mask=g==c
                    region=shift(mask,dy,dx)|(mask if copy else False)
                    if np.any(out[region]!=bg):
                        break
                    out[region]=c
                else:
                    results.append(out.tolist())
                    continue
                break
            if len(results)==len(queries):
                yield 2+sum((dy,dx)!=(0,0) for _,dy,dx in choices.values()),results
