"""Normalize empty space and match objects to bounded negative-space shapes."""
from itertools import product

import numpy as np
from scipy import ndimage

from concepts import background, components, orient


def normalize_space(grid, bg_mode, gap_row, gap_col, margin):
    bg = background(grid, bg_mode)
    output = grid
    for axis, gap in enumerate((gap_row, gap_col)):
        view = output if axis == 0 else output.T
        present = np.any(view != bg, axis=1)
        positions = np.flatnonzero(present)
        if not len(positions):
            return
        groups = np.split(positions, np.flatnonzero(np.diff(positions) > 1)+1)
        parts = []
        if margin:
            parts.append(np.full((margin, view.shape[1]), bg))
        for index, rows in enumerate(groups):
            if index and gap:
                parts.append(np.full((gap, view.shape[1]), bg))
            parts.append(view[rows])
        if margin:
            parts.append(np.full((margin, view.shape[1]), bg))
        view = np.concatenate(parts)
        output = view if axis == 0 else view.T
    return output if max(output.shape) <= 30 else None


def cavity_shapes(mask, side):
    oriented = np.rot90(mask, side)
    padded = np.pad(oriented, 1)
    padded[-1,:] = True
    padded[:,0] = padded[:,-1] = True
    fill = ndimage.binary_fill_holes(padded)[1:-1,1:-1] & ~oriented
    labels, _ = ndimage.label(fill)
    return [np.rot90(labels[box] == index, -side)
            for index, box in enumerate(ndimage.find_objects(labels), 1)]


def fits_cavity(mask, shape, symmetric, insertion):
    for code in range(8 if symmetric else 1):
        value = orient(mask, code)
        if insertion:
            if value.shape[0] < shape.shape[0]:
                continue
            value = value[:shape.shape[0]]
            columns = np.flatnonzero(value.any(axis=0))
            if not len(columns):
                continue
            value = value[:,columns[0]:columns[-1]+1]
        if np.array_equal(value, shape):
            return True
    return False


def cavity_paint(grid, bg_mode, select, side, symmetric, insertion, ink):
    bg = background(grid, bg_mode)
    pieces = components(grid, bg, separated=True, diagonal=1)
    if not 2 <= len(pieces) <= 40:
        return
    score = {'area': lambda obj: obj['area'], 'width': lambda obj: obj['mask'].shape[1],
             'height': lambda obj: obj['mask'].shape[0]}[select]
    maximum = max(map(score, pieces))
    hosts = [obj for obj in pieces if score(obj) == maximum]
    if len(hosts) != 1:
        return
    host = hosts[0]
    shapes = cavity_shapes(host['mask'], side)
    if not shapes:
        return
    output = grid.copy()
    changed = False
    for obj in pieces:
        if obj is host:
            continue
        match = any(fits_cavity(obj['mask'], shape, symmetric, insertion) for shape in shapes)
        if match:
            view = output[obj['box']]
            view[obj['mask']] = ink
            changed = True
    return output if changed else None


def object_grid(grid, bg_mode, columns, axis, gap, align):
    bg = background(grid, bg_mode)
    pieces = components(grid, bg, separated=False, diagonal=2)
    if not 2 <= len(pieces) <= 20 or len(pieces) % columns:
        return
    pieces = sorted(pieces, key=lambda obj: (obj['top'],obj['left']) if axis == 0
                    else (obj['left'],obj['top']))
    rows = []
    for start in range(0,len(pieces),columns):
        group = sorted(pieces[start:start+columns], key=lambda obj: obj['left'] if axis == 0 else obj['top'])
        rows.append([obj['patch'] if axis == 0 else obj['patch'].T for obj in group])
    height = max(p.shape[0] for row in rows for p in row)
    width = max(p.shape[1] for row in rows for p in row)
    shape = len(rows)*height+gap*(len(rows)-1), columns*width+gap*(columns-1)
    if max(shape) > 30:
        return
    output = np.full(shape,bg)
    for r,row in enumerate(rows):
        for c,patch in enumerate(row):
            top = r*(height+gap)+(height-patch.shape[0])*align//2
            left = c*(width+gap)+(width-patch.shape[1])*align//2
            output[top:top+patch.shape[0],left:left+patch.shape[1]] = patch
    return output if axis == 0 else output.T


def fit(task):
    if not task['train'] or not task['test'] or any('output' in q for q in task['test']):
        raise ValueError('Expected demonstrations and unlabeled queries')
    grids = [np.asarray(e['input']) for e in task['train']+task['test']]
    targets = [np.asarray(e['output']) for e in task['train']]
    n = len(targets)
    seen = set()
    for spec in product(('zero','dominant'), (2,3,4), (0,1), range(3), range(3)):
        if not all(np.array_equal(object_grid(g,*spec),y) for g,y in zip(grids,targets)):
            continue
        outputs = [object_grid(g,*spec) for g in grids[n:]]
        if any(out is None for out in outputs):
            continue
        key = tuple((out.shape,out.tobytes()) for out in outputs)
        if key not in seen:
            seen.add(key)
            yield dict(family='object grid',program=list(spec),cost=5,grids=[out.tolist() for out in outputs])
    for spec in product(('zero','dominant'), range(3), range(3), range(2)):
        if not all(np.array_equal(normalize_space(g,*spec), y) for g,y in zip(grids,targets)):
            continue
        outputs = [normalize_space(g,*spec) for g in grids[n:]]
        if any(out is None for out in outputs):
            continue
        key = tuple((out.shape,out.tobytes()) for out in outputs)
        if key not in seen:
            seen.add(key)
            yield dict(family='space',program=list(spec),cost=4,grids=[out.tolist() for out in outputs])
    if any(g.shape != y.shape for g,y in zip(grids,targets)):
        return
    paints = set.union(*(set(map(int,y[g != y])) for g,y in zip(grids,targets)))
    if len(paints) != 1:
        return
    ink = next(iter(paints))
    for spec in product(('zero','dominant'), ('area','width','height'), range(4), (False,True), (False,True)):
        if not all(np.array_equal(cavity_paint(g,*spec,ink),y) for g,y in zip(grids,targets)):
            continue
        outputs = [cavity_paint(g,*spec,ink) for g in grids[n:]]
        if any(out is None for out in outputs):
            continue
        key = tuple((out.shape,out.tobytes()) for out in outputs)
        if key not in seen:
            seen.add(key)
            yield dict(family='cavity',program=[*spec,ink],cost=6,grids=[out.tolist() for out in outputs])
