"""Infer correspondence between panels and reconstruct noisy periodic fields."""
from collections import Counter
from itertools import product

import numpy as np
from scipy import ndimage

from concepts import background, orient


def compress(mask):
    reduced = mask[np.r_[True, np.any(mask[1:] != mask[:-1], axis=1)]]
    return reduced[:, np.r_[True, np.any(reduced[:,1:] != reduced[:,:-1], axis=0)]]


def panel_transfer(grid, bg_mode, wall, matching, turn):
    bg = background(grid, bg_mode)
    if wall == bg:
        return
    labels, _ = ndimage.label(grid != wall)
    panels = []
    for index, box in enumerate(ndimage.find_objects(labels), 1):
        if not np.all(labels[box] == index):
            continue
        patch = grid[box]
        colors = set(map(int, patch.ravel()))-{bg}
        if len(colors) == 1 and patch.size >= 4:
            panels.append((box, patch, next(iter(colors))))
    if not 3 <= len(panels) <= 60:
        return
    frequency = Counter(ink for _, _, ink in panels)
    usual = frequency.most_common(1)[0][0]
    if frequency[usual] < 2:
        return
    donors = [(patch != bg, color) for _, patch, color in panels if color != usual]
    if not donors:
        return
    output = grid.copy()
    changed = False
    for box, patch, color in panels:
        if color != usual:
            continue
        possibilities = []
        for mask, ink in donors:
            for code in range(8) if turn == 'match' else [turn]:
                pattern = orient(mask, code)
                target = patch != bg
                if matching == 'compressed':
                    equal = np.array_equal(compress(pattern), compress(target))
                else:
                    h, w = target.shape
                    dh, dw = pattern.shape
                    if dh % h == dw % w == 0:
                        resized = pattern[::dh//h, ::dw//w]
                        regular = np.array_equal(pattern, resized.repeat(dh//h, 0).repeat(dw//w, 1))
                        equal = regular and (not np.any(resized & ~target) if matching == 'subset'
                                             else np.array_equal(resized, target))
                    elif h % dh == w % dw == 0:
                        resized = pattern.repeat(h//dh, 0).repeat(w//dw, 1)
                        equal = not np.any(resized & ~target) if matching == 'subset' else np.array_equal(target, resized)
                    else:
                        equal = False
                if equal:
                    painted = patch.copy()
                    painted[resized if matching == 'subset' else target] = ink
                    possibilities.append(painted)
        if possibilities and any(not np.array_equal(possibilities[0], value) for value in possibilities):
            return
        if possibilities:
            view = output[box]
            view[:] = possibilities[0]
            changed = True
    return output if changed else None


def bounds(mask):
    points = np.argwhere(mask)
    if not len(points):
        return
    low, high = points.min(axis=0), points.max(axis=0)+1
    return np.array([low[0], low[1], high[0], high[1]])


def anchor(grid, bg, selector):
    if selector[0] == 'color':
        return selector[1] if np.any(grid == selector[1]) else None
    scores = []
    for color in np.unique(grid):
        if color == bg:
            continue
        view = grid == color
        if selector[1] == 1:
            view = view.T
        longest = 0
        for row in view:
            changes = np.diff(np.r_[False, row, False].astype(int))
            starts = np.flatnonzero(changes == 1)
            ends = np.flatnonzero(changes == -1)
            longest = max(longest, max(ends-starts, default=0))
        scores.append((-longest, int(color)))
    scores.sort()
    rank = selector[2]
    return scores[rank][1] if len(scores) > rank else None


def periodic_reconstruction(grid, bg_mode, selector, padding, periods):
    bg = background(grid, bg_mode)
    color = anchor(grid, bg, selector)
    if color is None or color == bg:
        return
    box = bounds(grid == color)
    box = box+padding
    r, c, end_r, end_c = box
    if min(r, c) < 0 or end_r > grid.shape[0] or end_c > grid.shape[1] or r >= end_r or c >= end_c:
        return
    region = grid[r:end_r, c:end_c]
    ph, pw = periods
    output = np.full(grid.shape, bg)
    field = output[r:end_r, c:end_c]
    for i, j in product(range(ph), range(pw)):
        values = region[i::ph, j::pw]
        if values.size:
            counts = np.bincount(values.ravel(), minlength=10)
            if np.count_nonzero(counts == counts.max()) > 1:
                return
            field[i::ph, j::pw] = counts.argmax()
    return output


def fit(task):
    if not task['train'] or not task['test'] or any('output' in q for q in task['test']):
        raise ValueError('Expected demonstrations and unlabeled queries')
    grids = [np.asarray(e['input']) for e in task['train']+task['test']]
    targets = [np.asarray(e['output']) for e in task['train']]
    n = len(targets)
    if any(g.shape != y.shape for g, y in zip(grids, targets)):
        return
    colors = sorted(set.intersection(*(set(map(int, g.ravel())) for g in grids[:n])))
    seen = set()
    for bg, wall, matching, turn in product(('zero', 'dominant'), colors,
                                           ('scaled', 'compressed', 'subset'), ('match', *range(8))):
        spec = bg, wall, matching, turn
        if not all(np.array_equal(panel_transfer(g, *spec), y) for g, y in zip(grids, targets)):
            continue
        outputs = [panel_transfer(g, *spec) for g in grids[n:]]
        if all(out is not None for out in outputs):
            key = tuple(out.tobytes() for out in outputs)
            if key not in seen:
                seen.add(key)
                yield dict(family='panel', program=list(spec), cost=6, grids=[g.tolist() for g in outputs])
    selectors = [('color', c) for c in colors]+[('streak', axis, rank) for axis, rank in product((0,1), range(3))]
    for bg_mode in ('zero', 'dominant'):
        bgs = [background(g, bg_mode) for g in grids]
        target_boxes = [bounds(y != bg) for y, bg in zip(targets, bgs)]
        if any(box is None for box in target_boxes):
            continue
        fields = [y[box[0]:box[2], box[1]:box[3]] for y, box in zip(targets, target_boxes)]
        periods = []
        for axis in (0, 1):
            valid = []
            for p in range(1, 13):
                if all(g.shape[axis] > p and
                       np.array_equal(np.take(g, range(p, g.shape[axis]), axis=axis),
                                      np.take(g, range(g.shape[axis]-p), axis=axis)) for g in fields):
                    valid.append(p)
            periods.append(valid)
        if not all(periods):
            continue
        for selector in selectors:
            selected = [anchor(g, bg, selector) for g, bg in zip(grids[:n], bgs)]
            if any(c is None or c == bg for c, bg in zip(selected, bgs)):
                continue
            pads = [box-bounds(g == color) for box, g, color in zip(target_boxes, grids, selected)]
            if any(np.max(np.abs(pad)) > 8 for pad in pads) or any(not np.array_equal(pad, pads[0]) for pad in pads):
                continue
            padding = pads[0]
            for period in product(*periods):
                if not all(np.array_equal(periodic_reconstruction(g, bg_mode, selector, padding, period), y)
                           for g, y in zip(grids, targets)):
                    continue
                outputs = [periodic_reconstruction(g, bg_mode, selector, padding, period) for g in grids[n:]]
                if any(out is None for out in outputs):
                    continue
                key = tuple(out.tobytes() for out in outputs)
                if key not in seen:
                    seen.add(key)
                    yield dict(family='periodic', program=[bg_mode, list(selector), padding.tolist(), list(period)],
                               cost=7, grids=[g.tolist() for g in outputs])
