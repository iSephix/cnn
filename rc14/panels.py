"""Learn a pixel function over aligned panels using their joint color signatures."""
import itertools
import numpy as np


def panels(grid, shape):
    h,w = shape
    for transpose in (False, True):
        g = grid.T if transpose else grid
        height,width = (w,h) if transpose else (h,w)
        if g.shape[0] != height or g.shape[1] <= width:
            continue
        for gap in (0,1,2):
            count, rem = divmod(g.shape[1]+gap,width+gap)
            if rem or not 2 <= count <= 8:
                continue
            pieces = [g[:,i*(width+gap):i*(width+gap)+width] for i in range(count)]
            yield (transpose,gap,count), [p.T if transpose else p for p in pieces]


def signatures(parts, mode):
    if mode=='literal':
        return np.stack(parts,axis=-1)
    if mode=='occupancy':
        return np.stack([p!=0 for p in parts],axis=-1)
    if mode=='local-background':
        return np.stack([p!=np.bincount(p.ravel()).argmax() for p in parts],axis=-1)
    raise ValueError(mode)


def rules(task):
    examples=task['train']
    parses=[dict(panels(np.array(e['input']),np.shape(e['output']))) for e in examples]
    for layout in sorted(set.intersection(*(set(d) for d in parses))):
        transpose,gap,count=layout
        tests=[]
        for q in task['test']:
            g=np.array(q['input'])
            axis=0 if transpose else 1
            width,rem=divmod(g.shape[axis]-(count-1)*gap,count)
            if rem or width<1:
                break
            shape=list(g.shape)
            shape[axis]=width
            tests.append(dict(panels(g,shape))[layout])
        if len(tests)!=len(task['test']):
            continue
        for mode in ('literal','occupancy','local-background'):
            table={}
            valid=True
            for d,e in zip(parses,examples):
                keys=signatures(d[layout],mode).reshape(-1,count)
                for k,c in zip(map(tuple,keys),np.ravel(e['output'])):
                    if table.setdefault(k,c)!=c:
                        valid=False
                        break
                if not valid:
                    break
            if not valid:
                continue
            out=[]
            for parts in tests:
                keys=signatures(parts,mode).reshape(-1,count)
                vals=[table.get(k,-1) for k in map(tuple,keys)]
                if -1 in vals:
                    break
                out.append(np.array(vals).reshape(parts[0].shape))
            if len(out)==len(tests):
                yield 1+int(mode=='literal'),out
