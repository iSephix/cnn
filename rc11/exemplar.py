"""Use a multicolor example inside the input as a program for completing other objects."""
import numpy as np
from scipy import ndimage as nd
from views import transform


def objects(g,bg,diag):
    labels,_ = nd.label(g!=bg,nd.generate_binary_structure(2,2 if diag else 1))
    out = []
    for i,s in enumerate(nd.find_objects(labels),1):
        m = labels[s]==i
        p = np.where(m,g[s],bg)
        out.append((s,p,m))
    return out


def alternatives(g,bg,diag):
    obs = objects(g,bg,diag)
    if not 2<=len(obs)<=100:
        return {}
    def stats(o):
        vals = o[1][o[2]]
        return len(set(vals)),len(vals)
    rules = {}
    for select in ('palette','size'):
        order = (lambda o:stats(o)) if select=='palette' else (lambda o:stats(o)[::-1])
        s0,source,m0 = max(obs,key=order)
        if m0.sum()<2:
            continue
        for symmetry in (False,True):
            views = []
            seen = set()
            for k in range(8 if symmetry else 1):
                a = transform(source,k)
                key = a.shape,a.tobytes()
                if key not in seen:
                    seen.add(key)
                    views.append(a)
            for recolor in ('literal','uniform','accent'):
                overlays = []
                for box,p,m in obs:
                    if box==s0 or m.sum()>m0.sum():
                        continue
                    y,x = np.where(m)
                    first = p[y[0],x[0]]
                    pos_y,pos_x = y+box[0].start,x+box[1].start
                    options = []
                    for ti,a in enumerate(views):
                        match = a==first if recolor=='literal' else a!=bg
                        for ay,ax in zip(*np.where(match)):
                            r,c = int(pos_y[0]-ay),int(pos_x[0]-ax)
                            h,w = a.shape
                            if r<0 or c<0 or r+h>len(g) or c+w>g.shape[1]:
                                continue
                            anchor = a[ay,ax]
                            stamp = (a if recolor=='literal' else np.where(a==bg,bg,first)
                                     if recolor=='uniform' else np.where(a==anchor,first,a))
                            underlying = g[r:r+h,c:c+w]
                            iy,ix = pos_y-r,pos_x-c
                            if (iy<0).any() or (iy>=h).any() or (ix<0).any() or (ix>=w).any():
                                continue
                            if not np.array_equal(stamp[iy,ix],p[y,x]):
                                continue
                            ink = stamp!=bg
                            if ((underlying!=bg)&ink&(underlying!=stamp)).any():
                                continue
                            overlap = int(((underlying==stamp)&ink).sum())
                            options.append(((-overlap,ti,r,c),r,c,stamp))
                    if options:
                        options.sort(key=lambda o:o[0])
                        overlays.append((options[0],len(options)==1))
                for unique in (False,True):
                    result = g.copy()
                    for (_,r,c,a),unambiguous in overlays:
                        if unique and not unambiguous:
                            continue
                        block = result[r:r+a.shape[0],c:c+a.shape[1]]
                        ink = a!=bg
                        block[ink] = a[ink]
                    if not np.array_equal(result,g):
                        rules[select,symmetry,recolor,unique] = result
    return rules


def infer(task):
    demos = task['train']
    if any(np.shape(e['input'])!=np.shape(e['output']) for e in demos):
        return
    grids = [np.asarray(e['input']) for e in demos+task['test']]
    targets = [np.asarray(e['output']) for e in demos]
    for diag in (False,True):
        for mode in ('mode','zero'):
            choices = None
            for g,y in zip(grids,targets):
                bg = int(np.bincount(g.ravel()).argmax()) if mode=='mode' else 0
                opts = alternatives(g,bg,diag)
                # A rule with no applicable completion leaves its input unchanged.
                if np.array_equal(g,y):
                    if choices is not None:
                        choices = [r for r in choices if r not in opts]
                    continue
                valid = [r for r,out in opts.items() if np.array_equal(out,y)]
                choices = valid if choices is None else [r for r in choices if r in valid]
                if not choices:
                    break
            if not choices:
                continue
            all_opts = [alternatives(g,int(np.bincount(g.ravel()).argmax()) if mode=='mode' else 0,diag) for g in grids]
            for rule in choices:
                result = [o.get(rule,g) for o,g in zip(all_opts,grids)]
                if all(np.array_equal(a,b) for a,b in zip(result,targets)):
                    yield (0,int(rule[1]),int(rule[2]!='literal')),['exemplar',diag,mode,rule],[v.tolist() for v in result[len(demos):]]
