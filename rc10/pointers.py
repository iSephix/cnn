"""Learn decision-tree leaves that point to colors elsewhere in the current scene."""
import numpy as np
from scipy import ndimage as nd
from induce import Frame, fit
from card import compile_tree
from fields import shift


def sources(g):
    h,w = g.shape
    bg = np.bincount(g.ravel()).argmax()
    yield 'self',g
    for m in (g[::-1],g[:,::-1],g[::-1,::-1],g.T if h==w else np.full_like(g,-1)):
        yield 'symmetry',m
    occ = g!=bg
    if occ.any():
        _,idx = nd.distance_transform_edt(~occ,return_indices=True)
        nearest = g[tuple(idx)]
    else:
        nearest = g
    yield 'nearest',nearest
    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)):
        direct = shift(g,dy,dx)
        yield 'neighbor',direct
        yield 'next-neighbor',shift(g,2*dy,2*dx)
        ray = np.full_like(g,-1)
        for i in range(1,max(h,w)):
            v = shift(g,i*dy,i*dx)
            valid = (v>=0)&(v!=bg)&(ray<0)
            ray[valid] = v[valid]
        yield 'ray',ray
    for axis in (0,1):
        def majority(line):
            values = line[line!=bg]
            return np.bincount(values).argmax() if len(values) else -1
        colors = np.apply_along_axis(majority,axis,g)
        yield 'line',np.broadcast_to(colors[None,:] if axis==0 else colors[:,None],g.shape)
    for c in range(10):
        yield 'constant',np.full_like(g,c)


def matrices(g):
    planes = [p for _,p in sources(g)]
    refs = np.column_stack([p.ravel() for p in planes])
    bg = np.bincount(g.ravel()).argmax()
    around = refs[:,6:30:3]
    flags = [refs,refs==refs[:,[0]],refs==bg,around==np.roll(around,1,axis=1)]
    rays = refs[:,8:30:3]
    for j in range(0,8,2):
        flags.append((rays[:,j]==rays[:,j+1])[:,None])
    y,x = np.indices(g.shape)
    flags.extend([p.reshape(-1,1) for p in (y,x,y%2,x%2,(y+x)%2)])
    return refs,np.column_stack(flags)


def assign(refs,ys,indices):
    # Learn a shared ordering of interchangeable sources from the fold's labels.
    candidates = np.concatenate([refs[i]==ys[i][:,None] for i in indices])
    remaining = np.ones(len(candidates),bool)
    order = [0]
    remaining &= ~candidates[:,0]
    while remaining.any():
        counts = candidates[remaining].sum(axis=0)
        source = int(np.argmax(counts))
        if counts[source]==0:
            raise ValueError('Missing constant color expression')
        order.append(source)
        remaining &= ~candidates[:,source]
    return {i: np.asarray(order)[np.argmax(refs[i][:,order]==ys[i][:,None],axis=1)] for i in indices}


def infer(task):
    demos = task['train']
    n = len(demos)
    if n<2 or any(np.shape(e['input'])!=np.shape(e['output']) for e in demos):
        return
    for canonical in (False,True):
        frames = [Frame(e['input'],canonical) for e in demos+task['test']]
        ys = [f.target(e['output']).ravel() for f,e in zip(frames,demos)]
        refs,features = zip(*(matrices(f.g) for f in frames))
        def train(indices):
            labels = assign(refs,ys,indices)
            return fit(features,labels,indices)
        def apply(model,i):
            source = model.predict(features[i])
            v = refs[i][np.arange(len(source)),source]
            return np.where(v<0,frames[i].g.ravel(),v)
        model = train(list(range(n)))
        if any(not np.array_equal(apply(model,i),y) for i,y in enumerate(ys)):
            continue
        errors = []
        for i,y in enumerate(ys):
            fold = train([j for j in range(n) if i!=j])
            errors.append(np.mean(apply(fold,i)!=y))
        outs = [f.decode(apply(model,i).reshape(f.g.shape)).tolist() for i,f in enumerate(frames[n:],n)]
        yield (sum(e>0 for e in errors),sum(errors),model.get_n_leaves()),['pointers',canonical,compile_tree(model)],outs
