"""Learn a canvas size from scene measurements and a paint program in output coordinates."""
from itertools import product
import numpy as np
from scipy import ndimage as nd
from induce import fit, Frame
from card import compile_tree


def measure(g,axis):
    bg = np.bincount(g.ravel()).argmax()
    yy,xx = np.where(g!=bg)
    features = {'axis':g.shape[axis],'other':g.shape[1-axis],
                'colors':np.unique(g).size, 'ink-colors':np.unique(g).size-1,
                'ink':len(yy), 'objects':nd.label(g!=bg)[1]}
    if len(yy):
        for i,v in enumerate((yy,xx)):
            features[('extent',i)] = v.max()-v.min()+1
            features[('occupied',i)] = len(np.unique(v))
    return features


def dimensions(grids, targets, axis):
    n = len(targets)
    expected = [y.shape[axis] for y in targets]
    measurements = [measure(g,axis) for g in grids]
    options = []
    for key in measurements[0]:
        if any(key not in m for m in measurements):
            continue
        vals = np.array([m[key] for m in measurements])
        options.append((key, vals))
        if key in ('axis','other'):
            for k in (2,3,4,5):
                options.extend([((key,'*',k),vals*k),((key,'//',k),vals//k)])
            options.extend([((key,'squared'),vals**2),((key,'minus-two'),vals-2)])
    options.append((('fixed',expected[0]),np.full(len(grids),expected[0])))
    seen = set()
    for name,vals in options:
        if vals.min()<1 or vals.max()>30 or list(vals[:n])!=expected or tuple(vals) in seen:
            continue
        seen.add(tuple(vals))
        yield name,vals
        if len(seen)>=3:
            return


def projected(g,h,w):
    H,W = g.shape
    yy,xx = np.indices((h,w))
    cy,cx = yy*H//h,xx*W//w
    ty,tx = yy%H,xx%W
    bg = np.bincount(g.ravel()).argmax()
    def stack(vals):
        return np.column_stack([np.broadcast_to(v,(h,w)).ravel() for v in vals])
    geom = stack([yy,xx,h-1-yy,w-1-xx,yy%2,xx%2,(yy+xx)%2,yy==xx,yy+xx==w-1,
                  np.minimum.reduce((yy,xx,h-1-yy,w-1-xx)),ty,tx,
                  yy%max(1,h//H),xx%max(1,w//W)])
    sample = stack([g[cy,cx],g[ty,tx],g[cy,tx],g[ty,cx],g[H-1-cy,cx],g[cy,W-1-cx],
                    g[tx%H,ty%W],g[np.minimum(yy,H-1),np.minimum(xx,W-1)]])
    cells = []
    for i in range(h):
        for j in range(w):
            # A field can aggregate down or broadcast up in each axis independently.
            p = g[i*H//h:max(i*H//h+1,(i+1)*H//h),j*W//w:max(j*W//w+1,(j+1)*W//w)]
            counts = np.bincount(p.ravel(),minlength=10)
            used = np.flatnonzero(counts)
            ink = used[used!=bg]
            cells.append([counts.argmax(),used[counts[used].argmin()],len(used),
                          ink[counts[ink].argmax()] if len(ink) else bg,*counts])
    global_counts = np.bincount(g.ravel(),minlength=10)
    colors = np.flatnonzero(global_counts)
    colors = colors[np.argsort(-global_counts[colors],kind='stable')]
    context = [np.full_like(yy,v) for v in global_counts]
    for pos in (yy,xx):
        context.extend([colors[pos%len(colors)],global_counts[colors[pos%len(colors)]]])
    for axis,idx in ((0,cx),(1,cy)):
        for c in range(10):
            context.append((g==c).sum(axis=axis)[idx])
    return geom,sample,np.array(cells),stack(context)


def infer(task):
    demos = task['train']
    n = len(demos)
    if n<2 or all(np.shape(e['input'])==np.shape(e['output']) for e in demos):
        return
    for canonical in (False,True):
        frames = [Frame(e['input'],canonical) for e in demos+task['test']]
        targets = [f.target(e['output']) for f,e in zip(frames,demos)]
        grids = [f.g for f in frames]
        ys = [y.ravel() for y in targets]
        for (yr,hs),(xr,ws) in product(*(dimensions(grids,targets,a) for a in (0,1))):
            features = [projected(g,h,w) for g,h,w in zip(grids,hs,ws)]
            for subset in ((1,), (2,), (0,1), (0,2,3), (0,1,2,3)):
                xs = [np.column_stack([group[i] for i in subset]) for group in features]
                tree = fit(xs,ys,range(n))
                if any(not np.array_equal(tree.predict(x),y) for x,y in zip(xs,ys)):
                    continue
                errors = [np.mean(fit(xs,ys,[j for j in range(n) if i!=j]).predict(xs[i])!=y)
                          for i,y in enumerate(ys)]
                cost = (sum(e>0 for e in errors),sum(errors),tree.get_n_leaves(),len(subset))
                outs = [f.decode(tree.predict(x).reshape(h,w)).tolist()
                        for f,x,h,w in zip(frames[n:],xs[n:],hs[n:],ws[n:])]
                yield cost,['resample',yr,xr,subset,canonical,compile_tree(tree)],outs
