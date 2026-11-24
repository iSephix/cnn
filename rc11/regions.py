"""Build object-relative coordinate systems, then learn a renderer in those coordinates."""
from pathlib import Path
import sys
import numpy as np
from scipy import ndimage as nd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rc10'))
from induce import Frame, fit
from card import compile_tree


def partition(g,mono,diagonal):
    bg = int(np.bincount(g.ravel()).argmax())
    kernel = nd.generate_binary_structure(2,2 if diagonal else 1)
    idx = np.zeros_like(g)
    offset = 0
    for mask in ([g==c for c in np.unique(g) if c!=bg] if mono else [g!=bg]):
        a,n = nd.label(mask,kernel)
        idx[mask] = a[mask]+offset
        offset += n
    return idx


def measure(g,idx):
    y,x = np.indices(g.shape)
    maps = np.zeros((g.size,32),dtype='int16')
    if not idx.any():
        return maps
    near = nd.distance_transform_edt(idx==0,return_distances=False,return_indices=True)
    owner = idx[tuple(near)]
    ry,rx = near
    signatures = {}
    stats = []
    for i,s in enumerate(nd.find_objects(idx),1):
        patch = idx[s]==i
        vals = g[s][patch]
        counts = np.bincount(vals,minlength=10)
        key = (patch.shape,patch.tobytes())
        signatures[key] = signatures.get(key,0)+1
        stats.append((s,patch,counts,key))
    areas = np.array([int(p.sum()) for _,p,_,_ in stats])
    for i,(s,mask,counts,key) in enumerate(stats,1):
        take = owner==i
        r,c = y[take],x[take]
        a,b = r-s[0].start,c-s[1].start
        h,w = mask.shape
        dy,dx = r-ry[take],c-rx[take]
        colors = np.flatnonzero(counts)
        iy,ix = np.clip(a,0,h-1),np.clip(b,0,w-1)
        inside = (a>=0)&(a<h)&(b>=0)&(b<w)
        patch = g[s]
        planes = (g[take],g[tuple(near)][take],dy,dx,abs(dy)+abs(dx),np.maximum(abs(dy),abs(dx)),
                  a,b,h-1-a,w-1-b,h,w,areas[i-1],h*w-areas[i-1],len(colors),
                  counts.argmax(),colors[counts[colors].argmin()],signatures[key],
                  int((areas>areas[i-1]).sum()),int((areas<areas[i-1]).sum()),
                  inside,mask[iy,ix],patch[h-1-iy,ix],patch[iy,w-1-ix],
                  patch[h-1-iy,w-1-ix],patch[a%h,b%w],(a+b)%2,a%2,b%2,
                  w//2-b,h//2-a,len(stats))
        maps[take.ravel(),:] = np.column_stack(np.broadcast_arrays(*planes))
    return maps


def infer(task):
    demos = task['train']
    n = len(demos)
    if n<2 or any(np.shape(e['input'])!=np.shape(e['output']) for e in demos):
        return
    for canonical in (False,True):
        fs = [Frame(e['input'],canonical) for e in demos+task['test']]
        ys = [f.target(e['output']).ravel() for f,e in zip(fs,demos)]
        seen = set()
        for mono,diagonal in ((True,False),(False,False),(True,True),(False,True)):
            idx = [partition(f.g,mono,diagonal) for f in fs]
            sig = tuple(a.tobytes() for a in idx)
            if sig in seen:
                continue
            seen.add(sig)
            xs = [measure(f.g,a) for f,a in zip(fs,idx)]
            for copy in (True,False):
                labels = [np.where(y==f.g.ravel(),10,y) if copy else y for f,y in zip(fs,ys)]
                tree = fit(xs,labels,range(n))
                if any(not np.array_equal(tree.predict(x),v) for x,v in zip(xs,labels)):
                    continue
                err = []
                for i in range(n):
                    fold = fit(xs,labels,[j for j in range(n) if j!=i])
                    v = fold.predict(xs[i])
                    err.append(float(np.mean(np.where(v==10,fs[i].g.ravel(),v)!=ys[i])))
                pred = []
                for f,x in zip(fs[n:],xs[n:]):
                    v = tree.predict(x).reshape(f.g.shape)
                    pred.append(f.decode(np.where(v==10,f.g,v)).tolist())
                yield (sum(e>0 for e in err),sum(err),tree.get_n_leaves()),['regions',mono,diagonal,canonical,copy,compile_tree(tree)],pred
