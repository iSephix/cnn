"""Fit latent column ordering, panel mosaics, and arithmetic motion through frames."""
from itertools import product

import numpy as np
from scipy import ndimage

from lattice import lattices


def background(g):
    return int(np.bincount(g.ravel()).argmax())


def slices(grid,axis,color):
    view=grid if axis==0 else grid.T
    excluded=np.all(view==color,axis=1)
    positions=np.flatnonzero(~excluded)
    if not len(positions) or not excluded.any():
        return []
    return np.split(positions,np.flatnonzero(np.diff(positions)>1)+1)


def separators(g,axis):
    view=g if axis==0 else g.T
    return {int(p[0]) for p in view if np.all(p==p[0]) and p[0]!=background(g)}


def tight(g,bg):
    points=np.argwhere(g!=bg)
    if not len(points):
        return
    lo,hi=points.min(axis=0),points.max(axis=0)+1
    return g[lo[0]:hi[0],lo[1]:hi[1]]


def mosaic(tiles,bg,size_mode,align):
    h,w=len(tiles),len(tiles[0])
    heights=[max(t.shape[0] for t in row) for row in tiles]
    widths=[max(tiles[i][j].shape[1] for i in range(h)) for j in range(w)]
    if size_mode=='fixed':
        heights=[max(heights)]*h
        widths=[max(widths)]*w
    if max(sum(heights),sum(widths))>30:
        return
    board=np.full((sum(heights),sum(widths)),bg)
    top=0
    for i,row in enumerate(tiles):
        left=0
        for j,t in enumerate(row):
            r=top+(heights[i]-t.shape[0])*align//2
            c=left+(widths[j]-t.shape[1])*align//2
            board[r:r+t.shape[0],c:c+t.shape[1]]=t
            left+=widths[j]
        top+=heights[i]
    return board


def separator_assignments(options):
    common=set.intersection(*options)
    for color in sorted(common):
        yield [color]*len(options)
    if all(len(o)==1 for o in options):
        yield [next(iter(o)) for o in options]


def mosaics(grids,targets,n):
    colors=[separators(g,0)&separators(g,1) for g in grids]
    for assigned in separator_assignments(colors):
        panels=[]
        for g,separator in zip(grids,assigned):
            rows=slices(g,0,separator)
            cols=slices(g,1,separator)
            if not 2<=len(rows)<=6 or not 2<=len(cols)<=6:
                break
            bg=background(g)
            tiles=[[tight(g[np.ix_(i,j)],bg) for j in cols] for i in rows]
            if any(t is None for row in tiles for t in row):
                break
            panels.append((tiles,bg))
        else:
            for spec in product(('fixed','variable'),range(3)):
                if all(np.array_equal(mosaic(t,b,*spec),y) for (t,b),y in zip(panels,targets)):
                    out=[mosaic(t,b,*spec) for t,b in panels[n:]]
                    if all(g is not None for g in out):
                        yield 4+int(spec[0]=='variable')+int(spec[1]!=0),out


def sequence(g,axis,separator):
    ranges=slices(g,axis,separator)
    if not 3<=len(ranges)<=8:
        return []
    parts=[g[r,:] if axis==0 else g[:,r] for r in ranges]
    return parts if all(q.shape==parts[0].shape for q in parts) else []


def extrapolate(frames,background,padding):
    if not frames:
        return
    h,w=frames[0].shape
    colors=set.union(*(set(f.ravel()) for f in frames))-{background}
    if not colors:
        return
    out=np.full((h,w),background)
    for color in colors:
        coords=[np.argwhere(g==color) for g in frames]
        if any(not len(v) for v in coords):
            return
        lowers=np.stack([a.min(axis=0) for a in coords])
        uppers=np.stack([a.max(axis=0)+1 for a in coords])
        diffs=[a[1:]-a[:-1] for a in (lowers,uppers)]
        if any(np.any(v!=v[0]) for v in diffs):
            return
        low=lowers[-1]+diffs[0][-1]
        high=uppers[-1]+diffs[1][-1]
        if np.any(low<0) or np.any(high>[h,w]) or np.any(low>=high):
            return
        old_y,old_x=lowers[-1]
        end_y,end_x=uppers[-1]
        pattern=frames[-1][old_y:end_y,old_x:end_x]==color
        r,c=np.indices(tuple(high-low))
        if padding=='scale':
            r=r*pattern.shape[0]//len(r)
            c=c*pattern.shape[1]//c.shape[1]
        else:
            r%=pattern.shape[0]
            c%=pattern.shape[1]
        y,x=low
        target=out[y:high[0],x:high[1]]
        active=pattern[r,c]
        if np.any(active&(target!=background)):
            return
        target[active]=color
    return out


def spread(frames):
    """Find a monotonic, color-constrained spread that reproduces all input transitions."""
    if not frames:
        return
    inks=set.intersection(*(set(g.ravel()) for g in frames))
    for ink in sorted(inks):
        edit=[(a!=b) for a,b in zip(frames,frames[1:])]
        if not all(m.any() and np.all(b[m]==ink) for b,m in zip(frames[1:],edit)):
            continue
        substrates=set.union(*(set(a[m]) for a,m in zip(frames,edit)))
        for diagonal,repeats in product((1,2),(1,2,3)):
            kernel=ndimage.generate_binary_structure(2,diagonal)
            def step(grid):
                mask=grid==ink
                enabled=np.isin(grid,list(substrates))
                for _ in range(repeats):
                    mask|=ndimage.binary_dilation(mask,kernel)&enabled
                return np.where(mask,ink,grid)
            if all(np.array_equal(step(a),b) for a,b in zip(frames,frames[1:])):
                yield step(frames[-1])


def next_frame(frames,bg,mode):
    if mode.startswith('propagate'):
        implicit=next(spread(frames),None)
        if implicit is not None:
            return implicit
    return extrapolate(frames,bg,mode.split()[-1])


def temporal(grids,targets,n):
    for axis in (0,1):
        for assigned in separator_assignments([separators(g,axis) for g in grids]):
            frames=[sequence(g,axis,c) for g,c in zip(grids,assigned)]
            if any(not f for f in frames):
                continue
            if any(f[0].shape!=y.shape for f,y in zip(frames,targets)):
                continue
            for mode in ('tile','scale','propagate tile','propagate scale'):
                for mode_bg in ('global','corner'):
                    colors=[background(g) if mode_bg=='global' else int(f[0][0,0])
                            for g,f in zip(grids,frames)]
                    if not all(np.array_equal(next_frame(f,b,mode),y) for f,b,y in zip(frames,colors,targets)):
                        continue
                    out=[next_frame(f,b,mode) for f,b in zip(frames[n:],colors[n:])]
                    if all(g is not None for g in out):
                        yield 5+int(mode!='tile')+int(mode_bg!='global'),out


def sorted_latent(grid,axis,color,reverse):
    scan=grid if axis==0 else grid.T
    values=(scan==color).sum(axis=1)
    order=np.argsort(-values if reverse else values,kind='stable')
    return scan[order] if axis==0 else scan[order].T


def reorder(grids,targets,n):
    if any(g.shape!=y.shape for g,y in zip(grids,targets)):
        return
    for _,axes,_ in lattices(grids,targets):
        latent=[g[a] for g,a in zip(grids,axes)]
        expected=[g[a] for g,a in zip(targets,axes)]
        palette=sorted(set.intersection(*(set(x.ravel()) for x in latent[:n])))
        for axis,marker,reverse in product((0,1),[None,*palette],(False,True)):
            colors=[background(g) if marker is None else marker for g in latent]
            if not all(np.array_equal(sorted_latent(g,axis,c,reverse),y) for g,c,y in zip(latent,colors,expected)):
                continue
            out=[]
            for g,a,small,c in zip(grids[n:],axes[n:],latent[n:],colors[n:]):
                answer=g.copy()
                answer[a]=sorted_latent(small,axis,c,reverse)
                out.append(answer)
            yield 4+int(marker is not None),out


def recipes(task):
    n=len(task['train'])
    grids=[np.array(e['input']) for e in task['train']+task['test']]
    expected=[np.array(e['output']) for e in task['train']]
    for fn in (mosaics,temporal,reorder):
        for cost,outputs in fn(grids,expected,n):
            yield fn.__name__,cost,[g.tolist() for g in outputs]
