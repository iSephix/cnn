"""Choose moving objects, simulate contact with obstacles, then render their paths."""
from itertools import product
import numpy as np
from regions import partition


def scene(g,mono,diag):
    idx = partition(g,mono,diag)
    groups = []
    for i in range(1,int(idx.max())+1):
        y,x = np.where(idx==i)
        c = int(np.bincount(g[y,x]).argmax())
        groups.append((y,x,c))
    return idx,groups


def selections(groups):
    if not groups:
        return {}
    areas = np.array([len(y) for y,_,_ in groups])
    colors = np.array([c for _,_,c in groups])
    return {'all':np.arange(len(groups)), 'smallest':np.flatnonzero(areas==areas.min()),
            'largest':np.flatnonzero(areas==areas.max()),
            **{int(c):np.flatnonzero(colors==c) for c in np.unique(colors)}}


def displacement(g,idx,groups,sel,direction):
    dy,dx = direction
    ids = sel+1
    # Move selected bodies independently; only unselected cells act as obstacles.
    blocked = (idx!=0)&~np.isin(idx,ids)
    result = []
    for j in sel:
        y,x,c = groups[j]
        edge = []
        if dy:
            edge.append(len(g)-1-y.max() if dy>0 else y.min())
        if dx:
            edge.append(g.shape[1]-1-x.max() if dx>0 else x.min())
        limit = int(min(edge))
        travel = limit
        for k in range(1,limit+1):
            if blocked[y+dy*k,x+dx*k].any():
                travel = k-1
                break
        result.append((y,x,travel,travel<limit))
    return result


def render(g,paths,direction,bg,style,stop):
    dy,dx = direction
    out = g.copy()
    active = [(y,x,n) for y,x,n,contact in paths if n and (contact or stop=='edge-or-object')]
    if style=='move':
        for y,x,_ in active:
            out[y,x] = bg
    for y,x,n in active:
        colors = g[y,x]
        for step in (range(1,n+1) if style=='trail' else (n,)):
            out[y+dy*step,x+dx*step] = colors
    return out


def infer(task):
    demos = task['train']
    if any(np.shape(e['input'])!=np.shape(e['output']) or
           not np.isin(e['output'],np.unique(e['input'])).all() for e in demos):
        return
    grids = [np.asarray(e['input']) for e in demos+task['test']]
    n = len(demos)
    ys = [np.asarray(e['output']) for e in demos]
    if all(np.array_equal(g,y) for g,y in zip(grids,ys)):
        return
    bgs = [int(np.bincount(g.ravel()).argmax()) for g in grids]
    seen = set()
    for mono,diag in ((True,False),(False,False),(True,True),(False,True)):
        scenes = [scene(g,mono,diag) for g in grids]
        key = tuple(idx.tobytes() for idx,_ in scenes)
        if key in seen:
            continue
        seen.add(key)
        selectors = [selections(group) for _,group in scenes]
        for selector in selectors[0]:
            if any(selector not in opts for opts in selectors):
                continue
            for dy,dx in product((-1,0,1),repeat=2):
                if not(dy or dx):
                    continue
                modes = list(product(('move','copy','trail'),('edge-or-object','object')))
                path_args = zip(grids,scenes,selectors,bgs)
                for i,(g,(idx,group),opts,bg) in enumerate(path_args):
                    paths = displacement(g,idx,group,opts[selector],(dy,dx))
                    if i==n:
                        break
                    modes = [(style,stop) for style,stop in modes
                             if np.array_equal(render(g,paths,(dy,dx),bg,style,stop),ys[i])]
                    if not modes:
                        break
                for style,stop in modes:
                    outs = []
                    for g,(idx,group),opts,bg in zip(grids[n:],scenes[n:],selectors[n:],bgs[n:]):
                        paths = displacement(g,idx,group,opts[selector],(dy,dx))
                        outs.append(render(g,paths,(dy,dx),bg,style,stop).tolist())
                    yield (0,int(selector!='all'),int(stop=='object')),['motion',mono,diag,selector,dy,dx,style,stop],outs
