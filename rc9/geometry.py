"""Infer paint operations from spatial relationships between a cell and a color."""
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_cdt
from combine import compatible
from memory import Frame


def relations(g, c):
    mark = g == c
    blank = np.zeros_like(mark)
    y, x = np.indices(g.shape)
    row, col = np.where(mark)
    for name, axis, values in [('y', y, row), ('x', x, col), ('sum', y+x, row+col),
                                ('diff', y-x, row-col)]:
        yield name + '-line', np.isin(axis, values)
    for name, sy, sx in [('row',0,1), ('col',1,0), ('diag',1,1), ('anti',1,-1)]:
        line = y * sx - x * sy
        coord = y * sy + x * sx
        between, before, after = (blank.copy() for _ in range(3))
        for v in np.unique(line[mark]):
            where = line == v
            points = coord[where & mark]
            between |= where & (coord > points.min()) & (coord < points.max())
            before |= where & (coord < points.min())
            after |= where & (coord > points.max())
        yield name + '-between', between
        yield name + '-before', before
        yield name + '-after', after
    if row.size:
        inside = (y >= row.min()) & (y <= row.max()) & (x >= col.min()) & (x <= col.max())
        border = inside & ((y == row.min()) | (y == row.max()) | (x == col.min()) | (x == col.max()))
    else:
        inside = border = blank
    yield 'box', inside
    yield 'border', border
    yield 'hole', binary_fill_holes(mark) & ~mark
    for metric in ('taxicab', 'chessboard'):
        d = distance_transform_cdt(~mark, metric=metric) if mark.any() else np.full(g.shape,-1)
        for radius in (1, 2):
            yield (metric, radius), d == radius


def infer(task):
    demos = task['train']
    if any(np.shape(e['input']) != np.shape(e['output']) for e in demos):
        return
    for canonical in (False, True):
        frames = [Frame(e['input'], canonical) for e in demos + task['test']]
        y = np.concatenate([f.target(e['output']).ravel() for f,e in zip(frames,demos)])
        g = np.concatenate([f.g.ravel() for f in frames])
        n = len(y)
        if np.array_equal(y, g[:n]):
            continue
        masks, specs, seen = [], [], set()
        for c in sorted(set(g[:n].tolist())):
            for rows in zip(*(relations(f.g,c) for f in frames)):
                mask = np.concatenate([m.ravel() for _,m in rows])
                signature = np.packbits(mask ^ mask[0]).tobytes()
                if signature in seen or mask[:n].all() or not mask[:n].any():
                    continue
                seen.add(signature)
                masks.append(mask)
                specs.append((c,rows[0][0]))
        if not masks:
            continue
        masks = np.array(masks)
        conflict, singles = compatible(g[:n], y, masks)
        if not conflict:
            continue
        singles = set(singles)
        for i, a in enumerate(masks):
            keys = g + 10*a
            if i in singles:
                partners = [None]
            else:
                _, candidates = compatible(keys[:n], y, masks)
                partners = [j for j in candidates if j > i and j not in singles]
            for j in partners:
                joint = keys if j is None else keys + 20*masks[j]
                lut = np.full(40,-1)
                lut[joint[:n]] = y
                guessed = lut[joint[n:]]
                out = np.where(guessed < 0, g[n:], guessed)
                result, start = [], 0
                for f in frames[len(demos):]:
                    result.append(f.decode(out[start:start+f.g.size].reshape(f.g.shape)).tolist())
                    start += f.g.size
                terms = [specs[i]] if j is None else [specs[i],specs[j]]
                edits = int(((lut >= 0) & (lut != np.arange(40)%10)).sum())
                yield (len(terms), edits, int((guessed<0).sum()), canonical), ['geometry', terms, canonical], result
