"""Induce paint and tile programs over candidate intermediate grids."""
import json
import numpy as np
from views import inspect, transform


def patterns(grids,targets):
    shape = grids[0].shape
    h,w = targets[0].shape
    sy,ry = divmod(h,shape[0])
    sx,rx = divmod(w,shape[1])
    if ry or rx or min(sy,sx)<1 or max(sy,sx)>12:
        return
    if any(y.shape!=(len(g)*sy,g.shape[1]*sx) for g,y in zip(grids,targets)):
        return
    if any(max(len(g)*sy,g.shape[1]*sx)>30 for g in grids):
        return
    for layout in (('blocks',) if sy*sx==1 else ('blocks','panels')):
        tiles = []
        for g,y in zip(grids,targets):
            h,w = g.shape
            a = (y.reshape(h,sy,w,sx).transpose(0,2,1,3) if layout=='blocks'
                 else y.reshape(sy,h,sx,w).transpose(1,3,0,2))
            tiles.append(a.reshape(h*w,-1))
        yield layout,sy,sx,tiles


def fit(keys,tiles,n):
    x = np.concatenate([g.ravel() for g in keys[:n]])
    t = np.concatenate(tiles)
    uniq,first,inv = np.unique(x,return_index=True,return_inverse=True)
    lookup = t[first]
    if not np.array_equal(lookup[inv],t):
        return
    queries = np.unique(np.concatenate([g.ravel() for g in keys[n:]]))
    if not np.isin(queries,uniq).all():
        return
    table = np.zeros((100,t.shape[1]),int)
    table[uniq] = lookup
    return table,uniq


def synth(task):
    if any('output' in e for e in task['test']):
        raise ValueError('Test labels must be stripped')
    demos = task['train']
    n = len(demos)
    grids = [np.asarray(e['input']) for e in demos+task['test']]
    ys = [np.asarray(e['output']) for e in demos]
    catalogs = [dict((name,(p,q)) for name,p,q in inspect(g)) for g in grids]
    seen = set()
    for name in catalogs[0]:
        if any(name not in c for c in catalogs):
            continue
        for k in range(8):
            pairs = [(transform(c[name][0],k),transform(c[name][1],k)) for c in catalogs]
            p,q = zip(*pairs)
            for layout,sy,sx,tiles in patterns(p,ys):
                for joined in (False,True):
                    if joined and all(np.array_equal(a,b) for a,b in pairs):
                        continue
                    keys = [a*10+b for a,b in pairs] if joined else p
                    sig = (layout,sy,sx,tuple((a.shape,a.astype('B').tobytes()) for a in keys))
                    if sig in seen:
                        continue
                    seen.add(sig)
                    match = fit(keys,tiles,n)
                    if match is None:
                        continue
                    table,uniq = match
                    outs = []
                    for key in keys[n:]:
                        h,w = key.shape
                        a = table[key].reshape(h,w,sy,sx)
                        axes = (0,2,1,3) if layout=='blocks' else (2,0,3,1)
                        outs.append(a.transpose(axes).reshape(h*sy,w*sx).tolist())
                    # Count source symbols without independent demonstration support.
                    support = np.sum([np.isin(uniq,np.unique(a)) for a in keys[:n]],axis=0)
                    weak = int(np.sum(support<2))
                    complexity = int(joined)*2 + int(k!=0) + int(name!=('whole',)) + int(layout=='panels')
                    cost = (weak,complexity,len(uniq),sy*sx)
                    model = {int(u):table[u].tolist() for u in uniq}
                    yield cost,['compose',name,k,joined,layout,sy,sx,model],outs
