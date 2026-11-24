"""Factor demonstration outputs into a learned motif dictionary and latent grid.

A spatial encoder predicts motif indices. A separately inferred decoder expands
those indices into output cells at the requested canvas size.
"""
from itertools import product
import numpy as np
from regions import Frame,fit,compile_tree
from resample import projected,dimensions


def factor(targets,h,w,panels):
    dictionary = {}
    labels = []
    for y in targets:
        H,W = y.shape
        rows,cols = H//h,W//w
        if panels:
            blocks = y.reshape(h,rows,w,cols).transpose(1,3,0,2)
        else:
            blocks = y.reshape(rows,h,cols,w).transpose(0,2,1,3)
        flat = blocks.reshape(-1,h*w)
        code = []
        for tile in flat:
            key = tile.astype('B').tobytes()
            code.append(dictionary.setdefault(key,len(dictionary)))
        labels.append(np.array(code).reshape(rows,cols))
    tiles = np.stack([np.frombuffer(k,dtype='B').reshape(h,w) for k in dictionary])
    return tiles,labels


def decode(code,tiles,panels):
    rows,cols = code.shape
    _,h,w = tiles.shape
    axes = (2,0,3,1) if panels else (0,2,1,3)
    return tiles[code].transpose(axes).reshape(rows*h,cols*w)


def decompositions(ys):
    height = int(np.gcd.reduce([y.shape[0] for y in ys]))
    width = int(np.gcd.reduce([y.shape[1] for y in ys]))
    total = sum(y.size for y in ys)
    opts = []
    for h,w in product(range(1,min(8,height)+1),range(1,min(8,width)+1)):
        if height%h or width%w or not 2<=h*w or total//(h*w)<2*len(ys):
            continue
        for panels in (False,True):
            tiles,codes = factor(ys,h,w,panels)
            k = len(tiles)
            if k>16 or k*h*w>total*0.6:
                continue
            # Repeated solid blocks are included. More complex tiles must also
            # pay for the number of stored cells in their dictionary.
            description = k*h*w + (total//(h*w))*np.log2(max(k,2))/3.32
            opts.append((description,h,w,panels,tiles,codes))
    return sorted(opts,key=lambda a:a[0])[:6]


def infer(task):
    demos = task['train']
    n = len(demos)
    if n<2:
        return
    for canonical in (False,True):
        fs = [Frame(e['input'],canonical) for e in demos+task['test']]
        grids = [f.g for f in fs]
        ys = [f.target(e['output']) for f,e in zip(fs,demos)]
        cache = {}
        for desc,h,w,panels,tiles,codes in decompositions(ys):
            dims = [list(dimensions(grids,codes,a)) for a in (0,1)]
            labels = [a.ravel() for a in codes]
            for (yr,hs),(xr,ws) in product(*dims):
                if max(hs)*h>30 or max(ws)*w>30:
                    continue
                key = tuple(hs),tuple(ws)
                if key not in cache:
                    cache[key] = [projected(g,hh,ww) for g,hh,ww in zip(grids,hs,ws)]
                for parts in ((1,),(0,2,3),(0,1,2,3)):
                    xs = [np.column_stack([group[j] for j in parts]) for group in cache[key]]
                    tree = fit(xs,labels,range(n))
                    if any(not np.array_equal(tree.predict(x),v) for x,v in zip(xs,labels)):
                        continue
                    err = []
                    for i in range(n):
                        fold = fit(xs,labels,[j for j in range(n) if i!=j])
                        v = fold.predict(xs[i]).reshape(codes[i].shape)
                        err.append(float(np.mean(decode(v,tiles,panels)!=ys[i])))
                    outs = []
                    for x,f,hh,ww in zip(xs[n:],fs[n:],hs[n:],ws[n:]):
                        v = tree.predict(x).reshape(hh,ww)
                        outs.append(f.decode(decode(v,tiles,panels)).tolist())
                    yield (sum(v>0 for v in err),sum(err),tree.get_n_leaves(),desc),['latent',canonical,yr,xr,parts,panels,tiles.tolist(),compile_tree(tree)],outs
