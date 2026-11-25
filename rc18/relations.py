"""Infer how pairs of colored objects interact and render the shared relation."""
from itertools import product
import numpy as np
from scipy import ndimage


def scene(grid, split):
    bg = int(np.bincount(grid.ravel()).argmax())
    objects = []
    for color in np.unique(grid):
        if color == bg:
            continue
        mask = grid == color
        if split:
            labels, _ = ndimage.label(mask)
            masks = (labels == i for i in range(1, labels.max() + 1))
        else:
            masks = (mask,)
        for part in masks:
            coords = np.argwhere(part)
            lo, hi = coords.min(axis=0), coords.max(axis=0)
            objects.append((int(color), coords, lo, hi))
    return bg, objects


def link_mask(grid, scene_data, mode, link, direction):
    _, objects = scene_data
    if len(objects) > 45:
        return None
    mask = np.zeros(grid.shape, dtype=bool)
    for i, (a, _, alo, ahi) in enumerate(objects):
        for b, _, blo, bhi in objects[i+1:]:
            if (link == 'same' and a != b) or (link == 'different' and a == b):
                continue
            if isinstance(link, tuple) and (a,b) != link:
                continue
            if mode == 'box':
                lo, hi = np.minimum(alo,blo), np.maximum(ahi,bhi)
                mask[lo[0]:hi[0]+1,lo[1]:hi[1]+1] = True
                continue
            if mode == 'centers':
                start, stop = (alo+ahi)//2, (blo+bhi)//2
                dy, dx = stop-start
                if dy and dx and abs(dy)!=abs(dx):
                    continue
                if (direction == 'axial' and dy and dx) or (direction == 'diagonal' and not (dy and dx)):
                    continue
                steps = np.arange(max(abs(dy),abs(dx))+1)
                mask[start[0]+steps*np.sign(dy),start[1]+steps*np.sign(dx)] = True
                continue
            for axis in (0,1):
                if direction not in ('any',axis):
                    continue
                perpendicular = 1-axis
                lo, hi = max(alo[perpendicular],blo[perpendicular]), min(ahi[perpendicular],bhi[perpendicular])
                start, stop = min(ahi[axis],bhi[axis])+1, max(alo[axis],blo[axis])
                if lo<=hi and start<stop:
                    view=mask if axis==0 else mask.T
                    view[start:stop,lo:hi+1]=True
    return mask


def rules(task):
    demos = task['train']
    raw = [np.array(e['input']) for e in demos+task['test']]
    targets = [np.array(e['output']) for e in demos]
    if any(a.shape!=b.shape for a,b in zip(raw,targets)):
        return
    edits = [x!=y for x,y in zip(raw,targets)]
    colors = set.union(*(set(y[m]) for y,m in zip(targets,edits)))
    if len(colors)!=1 or not any(m.any() for m in edits):
        return
    color = min(colors)
    for split in (False,True):
        scenes = [scene(g,split) for g in raw]
        palettes = [set(o[0] for o in s[1]) for s in scenes[:len(demos)]]
        shared = sorted(set.intersection(*palettes))
        links = ('same','different','all',*(p for p in product(shared,repeat=2) if p[0]<=p[1]))
        for mode, directions in (('centers',('any','axial','diagonal')),('gap',('any',0,1)),('box',('any',))):
            for link in links:
                for direction in directions:
                    eligible = {False,True}
                    for grid,sc,target,changed in zip(raw,scenes,targets,edits):
                        mask=link_mask(grid,sc,mode,link,direction)
                        if mask is None:
                            eligible.clear()
                            break
                        for background_only in tuple(eligible):
                            region=mask & (grid==sc[0]) if background_only else mask
                            if np.any(changed&~region) or np.any(target[region]!=color):
                                eligible.remove(background_only)
                        if not eligible:
                            break
                    for background_only in sorted(eligible):
                        outputs=[]
                        for grid,sc in zip(raw[len(demos):],scenes[len(demos):]):
                            mask=link_mask(grid,sc,mode,link,direction)
                            if mask is None:
                                break
                            if background_only:
                                mask &= grid==sc[0]
                            out=grid.copy()
                            out[mask]=color
                            outputs.append(out.tolist())
                        if len(outputs)==len(task['test']):
                            yield 2+int(split)+int(isinstance(link,tuple)),outputs
