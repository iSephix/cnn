"""Solve scene correspondence and conserved-piece placement as constraints."""
from itertools import product

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from concepts import background, components, orient


def legend(grid, bg_mode, code, metric):
    bg = background(grid, bg_mode)
    groups = components(grid, bg)
    palettes = [len(np.unique(obj['patch'][obj['mask']])) for obj in groups]
    if not palettes or max(palettes) < 3:
        return
    sources = [obj for obj, count in zip(groups, palettes) if count == max(palettes)]
    if len(sources) != 1:
        return
    source = sources[0]
    patch = orient(source['patch'], code)
    ink = int(np.bincount(patch.ravel()).argmax())
    markers = components(patch, ink, separated=True)
    markers = [obj for obj in markers if int(obj['patch'][obj['mask']][0]) != bg]
    targets = [obj for obj in groups if obj is not source and
               len(np.unique(obj['patch'][obj['mask']])) == 1]
    if not 2 <= len(markers) == len(targets) <= 8:
        return
    def positions(objects):
        coords = np.array([[obj['top']+(obj['mask'].shape[0]-1)/2,
                            obj['left']+(obj['mask'].shape[1]-1)/2] for obj in objects])
        low, high = coords.min(axis=0), coords.max(axis=0)
        return (coords-low)/np.maximum(high-low, 1)
    a, b = positions(markers), positions(targets)
    distance = np.abs(a[:,None,:]-b[None,:,:])
    costs = distance.sum(axis=2) if metric == 'absolute' else (distance**2).sum(axis=2)
    rows, cols = linear_sum_assignment(costs)
    output = grid.copy()
    for i, j in zip(rows, cols):
        obj = targets[j]
        color = int(markers[i]['patch'][markers[i]['mask']][0])
        view = output[obj['box']]
        view[obj['mask']] = color
    return output


def projected_pins(grid, bg_mode, reach):
    bg = background(grid, bg_mode)
    pieces = components(grid, bg, separated=True)
    hosts = [obj for obj in pieces if obj['area'] >= 12 and
             obj['area']/obj['mask'].size >= .8 and min(obj['mask'].shape) >= 3]
    if not hosts or len(hosts) > 6:
        return
    output = grid.copy()
    changed = False
    for host in hosts:
        box = host['box']
        top, left, bottom, right = box[0].start, box[1].start, box[0].stop-1, box[1].stop-1
        host_color = int(host['patch'][host['mask']][0])
        for color in np.unique(grid):
            if color in (bg, host_color):
                continue
            points = np.argwhere(grid == color)
            row_pins = [p for p in points if top <= p[0] <= bottom and
                        0 < min(abs(p[1]-left), abs(p[1]-right)) <= reach and
                        (p[1] < left or p[1] > right)]
            col_pins = [p for p in points if left <= p[1] <= right and
                        0 < min(abs(p[0]-top), abs(p[0]-bottom)) <= reach and
                        (p[0] < top or p[0] > bottom)]
            if len(row_pins) != 1 or len(col_pins) != 1:
                continue
            row_pin, col_pin = row_pins[0], col_pins[0]
            row_start = top if col_pin[0] < top else bottom
            col_start = left if row_pin[1] < left else right
            r0, r1 = sorted((row_start, int(row_pin[0])))
            c0, c1 = sorted((col_start, int(col_pin[1])))
            window = output[r0:r1+1, c0:c1+1]
            old = grid[r0:r1+1, c0:c1+1]
            if np.any((old != host_color) & (old != color)):
                continue
            window[:] = color
            output[tuple(row_pin)] = output[tuple(col_pin)] = bg
            changed = True
    return output if changed else None


def exact_cover(grid, bg_mode, excluded, rotate, budget=40000):
    bg = background(grid, bg_mode)
    hosts, donors = [], []
    empty = np.zeros(grid.shape, bool)
    for obj in components(grid, bg, separated=True, diagonal=1):
        holes = ndimage.binary_fill_holes(obj['mask']) & ~obj['mask']
        if holes.any():
            hosts.append(obj)
            empty[obj['box']] |= holes
        elif obj['mask'].all() and int(obj['patch'][obj['mask']][0]) not in excluded:
            donors.append(obj)
    if not hosts or not 1 <= len(donors) <= 14 or sum(obj['area'] for obj in donors) != int(empty.sum()):
        return []
    board = np.full(grid.shape, bg)
    for host in hosts:
        view = board[host['box']]
        view[host['mask']] = host['patch'][host['mask']]
    solutions = {}
    nodes = 0
    def solve(space, available):
        nonlocal nodes
        nodes += 1
        if nodes > budget or len(solutions) >= 16:
            return
        positions = np.argwhere(space)
        if not len(positions):
            if not available:
                solutions.setdefault(board.tobytes(), board.copy())
            return
        row, col = positions[0]
        tried = set()
        for index in available:
            obj = donors[index]
            color = int(obj['patch'][obj['mask']][0])
            h, w = obj['mask'].shape
            for height, width in sorted({(h,w), (w,h)}) if rotate else [(h,w)]:
                signature = color, height, width
                if signature in tried:
                    continue
                tried.add(signature)
                # The first empty cell must be the top-left of a filled rectangle.
                if row+height > space.shape[0] or col+width > space.shape[1]:
                    continue
                region = np.s_[row:row+height, col:col+width]
                if not space[region].all():
                    continue
                board[region] = color
                space[region] = False
                solve(space, [i for i in available if i != index])
                space[region] = True
                board[region] = bg
    solve(empty.copy(), list(range(len(donors))))
    return list(solutions.values())


def fit(task):
    if not task['train'] or not task['test'] or any('output' in q for q in task['test']):
        raise ValueError('Expected demonstrations and unlabeled queries')
    grids = [np.asarray(e['input']) for e in task['train']+task['test']]
    targets = [np.asarray(e['output']) for e in task['train']]
    n = len(targets)
    if any(g.shape != y.shape for g, y in zip(grids, targets)):
        return
    if all(np.array_equal(g, y) for g, y in zip(grids, targets)):
        return
    seen = set()
    families = [('legend', legend, (bg, code, metric))
                for bg, code, metric in product(('zero','dominant'), range(8), ('absolute','square'))]
    families += [('pins', projected_pins, (bg, reach))
                 for bg, reach in product(('zero','dominant'), (1,2,3))]
    for name, function, spec in families:
        if not all(np.array_equal(function(g,*spec), y) for g,y in zip(grids, targets)):
            continue
        outputs = [function(g,*spec) for g in grids[n:]]
        if any(out is None for out in outputs):
            continue
        key = tuple(g.tobytes() for g in outputs)
        if key not in seen:
            seen.add(key)
            yield dict(family=name, program=list(spec), cost=6, grids=[g.tolist() for g in outputs])
    removed = set.union(*(set(map(int,g.ravel()))-set(map(int,y.ravel())) for g,y in zip(grids,targets)))
    for bg, exclusions, rotate in product(('zero','dominant'), ((), tuple(sorted(removed))), (False,True)):
        solutions = [exact_cover(g,bg,exclusions,rotate) for g in grids]
        if any(not rows for rows in solutions):
            continue
        for rank in range(16):
            if not all(len(rows)>rank and np.array_equal(rows[rank], y) for rows,y in zip(solutions,targets)):
                continue
            if any(len(rows)<=rank for rows in solutions[n:]):
                continue
            outputs = [rows[rank] for rows in solutions[n:]]
            key = tuple(g.tobytes() for g in outputs)
            if key not in seen:
                seen.add(key)
                yield dict(family='exact cover', program=[bg,list(exclusions),rotate,rank],
                           cost=7+rank, grids=[g.tolist() for g in outputs])
