"""Object-relative executable concepts fitted to demonstrations."""
from itertools import product
from functools import lru_cache

import numpy as np
from scipy import ndimage


def background(grid, mode):
    return 0 if mode == 'zero' else int(np.bincount(grid.ravel()).argmax())


def components(grid, bg, separated=False, diagonal=2):
    return cached_components(grid.astype(np.int8).tobytes(), grid.shape, bg, separated, diagonal)


@lru_cache(maxsize=256)
def cached_components(pixels, shape, bg, separated, diagonal):
    grid = np.frombuffer(pixels, dtype=np.int8).reshape(shape)
    result = []
    masks = [grid == c for c in np.unique(grid) if c != bg] if separated else [grid != bg]
    for mask in masks:
        labels, _ = ndimage.label(mask, ndimage.generate_binary_structure(2, diagonal))
        for index, box in enumerate(ndimage.find_objects(labels), 1):
            local = labels[box] == index
            patch = np.where(local, grid[box], bg)
            result.append(dict(box=box, mask=local, patch=patch, area=int(local.sum()),
                               top=box[0].start, left=box[1].start))
    return result


def orient(grid, code):
    return np.rot90(grid if code < 4 else grid[:, ::-1], code % 4)


def transfer(grid, bg_mode, pick, orientation):
    bg = background(grid, bg_mode)
    pieces = components(grid, bg)
    donors = [obj for obj in pieces if len(np.unique(obj['patch'][obj['mask']])) > 1]
    targets = [obj for obj in pieces if len(np.unique(obj['patch'][obj['mask']])) == 1]
    if not donors or not targets or len(pieces) > 35:
        return
    output = grid.copy()
    changed = False
    for target in targets:
        possible = []
        for donor in donors:
            for code in range(8) if orientation == 'match' else [orientation]:
                pattern = orient(donor['patch'], code)
                support = orient(donor['mask'], code)
                h, w = target['mask'].shape
                dh, dw = support.shape
                if h % dh or w % dw:
                    continue
                sy, sx = h//dh, w//dw
                if min(sy, sx) < 1 or max(sy, sx) > 10:
                    continue
                resized = support.repeat(sy, 0).repeat(sx, 1)
                if not np.array_equal(resized, target['mask']):
                    continue
                image = pattern.repeat(sy, 0).repeat(sx, 1)
                distance = abs(donor['top']-target['top'])+abs(donor['left']-target['left'])
                value = donor['area'] if pick == 'smallest' else -donor['area'] if pick == 'largest' else distance
                possible.append((value, code, image))
        if possible:
            possible.sort(key=lambda row: (row[0], row[1]))
            best = possible[0][2]
            if orientation == 'match' and any(not np.array_equal(best, image)
                                             for score, _, image in possible if score == possible[0][0]):
                return
            view = output[target['box']]
            view[target['mask']] = best[target['mask']]
            changed = True
    return output if changed else None


def broadcast(grid, bg_mode, host, margin, diagonal, kernel):
    bg = background(grid, bg_mode)
    if host == bg or host not in grid:
        return
    output = grid.copy()
    changed = False
    for obj in components(grid, bg, diagonal=diagonal):
        patch, mask = obj['patch'], obj['mask']
        seeds = set(map(int, patch[mask]))-{host}
        if len(seeds) != 1 or not np.any(patch == host):
            continue
        ink = next(iter(seeds))
        core = ndimage.binary_erosion(mask, structure=np.ones((kernel, kernel)), iterations=margin,
                                     border_value=0) if margin else mask
        view = output[obj['box']]
        view[core & (patch == host)] = ink
        changed |= bool(np.any(core & (patch == host)))
    return output if changed else None


def staircase(grid, bg_mode, axis, order, reverse, gap_y, gap_x, canvas, turn):
    bg = background(grid, bg_mode)
    pieces = components(grid, bg, separated=True)
    if not 2 <= len(pieces) <= 15:
        return
    keys = dict(position=lambda obj: (obj['top'], obj['left']),
                column=lambda obj: (obj['left'], obj['top']),
                area=lambda obj: obj['area'],
                color=lambda obj: int(obj['patch'][obj['mask']][0]))
    ordered = sorted(pieces, key=keys[order], reverse=reverse)
    patches = [obj['patch'] if axis == 0 else obj['patch'].T for obj in ordered]
    origins = []
    row = col = 0
    for patch in patches:
        origins.append((row, col))
        row += patch.shape[0]+gap_y
        col += patch.shape[1]+gap_x
        if min(row, col) < 0:
            return
    height = max(r+p.shape[0] for (r, _), p in zip(origins, patches))
    width = max(c+p.shape[1] for (_, c), p in zip(origins, patches))
    if canvas == 'original':
        shape = grid.shape if axis == 0 else grid.T.shape
        if height > shape[0] or width > shape[1]:
            return
        height, width = shape
    if max(height, width) > 30:
        return
    output = np.full((height, width), bg)
    for (r, c), patch in zip(origins, patches):
        view = output[r:r+patch.shape[0], c:c+patch.shape[1]]
        np.copyto(view, patch, where=patch != bg)
    if axis:
        output = output.T
    return orient(output, turn)


def fit(task):
    if not task['train'] or not task['test'] or any('output' in q for q in task['test']):
        raise ValueError('Expected demonstrations and unlabeled queries')
    inputs = [np.asarray(e['input']) for e in task['train']+task['test']]
    targets = [np.asarray(e['output']) for e in task['train']]
    n = len(targets)
    if all(np.array_equal(x, y) for x, y in zip(inputs, targets)):
        return
    same_shape = all(x.shape == y.shape for x, y in zip(inputs, targets))
    common = set.intersection(*(set(map(int, g.ravel())) for g in inputs[:n]))
    specs = []
    for bg in ('zero', 'dominant'):
        if same_shape:
            specs.extend(('transfer', (bg, pick, code)) for pick, code in
                         product(('smallest', 'largest', 'nearest'), ('match', *range(8))))
            specs.extend(('broadcast', (bg, host, margin, diag, kernel)) for host, margin, diag, kernel in
                         product(sorted(common), (0, 1, 2), (1, 2), (2, 3)))
        specs.extend(('staircase', (bg, axis, order, reverse, gy, gx, canvas, turn))
                     for axis, order, reverse, gy, gx, canvas, turn in product(
                         (0, 1), ('position', 'column', 'area', 'color'), (False, True),
                         (-1, 0, 1), (-1, 0, 1), ('original', 'tight'), (0, 2, 4, 6)))
    functions = dict(transfer=transfer, broadcast=broadcast, staircase=staircase)
    seen = set()
    for name, spec in specs:
        function = functions[name]
        if not all(np.array_equal(function(g, *spec), y) for g, y in zip(inputs, targets)):
            continue
        predictions = [function(g, *spec) for g in inputs[n:]]
        if any(out is None for out in predictions):
            continue
        key = tuple((g.shape, g.tobytes()) for g in predictions)
        if key not in seen:
            seen.add(key)
            yield dict(family=name, program=list(spec), cost=4+len(spec),
                       grids=[g.tolist() for g in predictions])
