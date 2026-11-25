"""A shared executable representation for grid programs and learned color maps."""
from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

_spec = spec_from_file_location('rc15_views', Path(__file__).resolve().parents[1]/'rc15/compose.py')
_views = module_from_spec(_spec)
_spec.loader.exec_module(_views)


def valid(grid):
    return (grid.ndim == 2 and 0 < min(grid.shape) and max(grid.shape) <= 30
            and grid.dtype.kind in 'biu' and grid.min() >= 0 and grid.max() <= 9)


def variants(grid):
    """Yield named operations whose parameters can be shared across examples."""
    for name, value in _views.views(grid):
        if name != ('scene',):
            yield ('view', *name), value
    for code in range(1, 8):
        yield ('orient', code), _views.orient(grid, code)
    bg = int(np.bincount(grid.ravel()).argmax())
    points = np.argwhere(grid != bg)
    if len(points):
        low, high = points.min(axis=0), points.max(axis=0)+1
        yield ('crop',), grid[low[0]:high[0], low[1]:high[1]]
    for axis in (0, 1):
        keep = np.any(grid != bg, axis=axis)
        if keep.any():
            yield ('remove empty', axis), grid[:, keep] if axis == 0 else grid[keep]
        yield ('mirror append', axis), np.concatenate((grid, np.flip(grid, axis)), axis=axis)
        if grid.shape[axis] % 2 == 0:
            first, second = np.split(grid, 2, axis=axis)
            for kind in ('overlay', 'intersection', 'xor'):
                if kind == 'overlay':
                    result = np.where(first != bg, first, second)
                elif kind == 'intersection':
                    result = np.where((first != bg) & (second != bg), first, bg)
                else:
                    result = np.where((first != bg) ^ (second != bg),
                                      np.where(first != bg, first, second), bg)
                yield ('combine halves', axis, kind), result
    for factor in (2, 3):
        yield ('tile', factor), np.tile(grid, (factor, factor))
        yield ('scale', factor), grid.repeat(factor, 0).repeat(factor, 1)
        if all(size % factor == 0 for size in grid.shape):
            small = grid[::factor, ::factor]
            if np.array_equal(small.repeat(factor, 0).repeat(factor, 1), grid):
                yield ('unscale', factor), small


def apply(grid, operation):
    operation = tuple(operation)
    if operation[0] == 'orient':
        return _views.orient(grid, operation[1])
    if operation[0] in ('tile', 'scale'):
        return restore(grid, operation)
    return next((value for name, value in variants(grid)
                 if name == operation and valid(value)), None)


def execute(grid, program, cache=None):
    value = np.asarray(grid, dtype=np.int16)
    cache = {} if cache is None else cache
    prefix = ()
    for operation in program:
        prefix += (tuple(operation),)
        if prefix not in cache:
            cache[prefix] = apply(value, operation)
        value = cache[prefix]
        if value is None:
            return None
    return value


def simplify(program):
    result = []
    probe = np.arange(6).reshape(2, 3)
    for operation in program:
        operation = list(operation)
        if result and result[-1][0] == operation[0] == 'orient':
            previous = result.pop()
            combined = _views.orient(_views.orient(probe, previous[1]), operation[1])
            code = next(k for k in range(8) if np.array_equal(_views.orient(probe, k), combined))
            if code:
                result.append(['orient', code])
        else:
            result.append(operation)
    return result


def signature(grids):
    return tuple((g.shape, g.tobytes()) for g in grids)


def canonical(grids):
    """A shared color-renaming invariant across all demonstration outputs."""
    mapping = {}
    result = []
    for grid in grids:
        pixels = []
        for color in grid.ravel():
            color = int(color)
            if color not in mapping:
                mapping[color] = len(mapping)
            pixels.append(mapping[color])
        result.append((grid.shape, bytes(pixels)))
    return tuple(result)


def recolor(inputs, targets):
    table = np.arange(10, dtype=np.int16)
    known = {}
    for source, target in zip(inputs, targets):
        if source.shape != target.shape:
            return None
        for color in np.unique(source):
            values = np.unique(target[source == color])
            if len(values) != 1:
                return None
            color, value = int(color), int(values[0])
            if color in known and known[color] != value:
                return None
            known[color] = value
    for color, value in known.items():
        table[color] = value
    return table


def restore(grid, operation):
    if operation is None:
        return grid
    kind, value = operation
    if kind == 'orient':
        return _views.orient(grid, value)
    if kind == 'tile':
        result = np.tile(grid, (value, value))
    else:
        result = grid.repeat(value, 0).repeat(value, 1)
    return result if valid(result) else None


def goals(targets):
    """Reverse known rendering operations to constrain the forward search."""
    yield None, targets
    for code in range(1, 8):
        # Inverse of the exact orientation, including reflected rotations.
        inverses = [_views.orient(target, code) for target in targets]
        inverse_code = next(k for k in range(8)
                            if np.array_equal(_views.orient(_views.orient(np.arange(6).reshape(2, 3), code), k),
                                              np.arange(6).reshape(2, 3)))
        yield ('orient', inverse_code), inverses
    for factor in (2, 3):
        if any(any(size % factor for size in g.shape) for g in targets):
            continue
        for kind in ('scale', 'tile'):
            small = [g[::factor, ::factor] if kind == 'scale'
                     else g[:g.shape[0]//factor, :g.shape[1]//factor] for g in targets]
            if all(np.array_equal(restore(g, (kind, factor)), y) for g, y in zip(small, targets)):
                yield (kind, factor), small


def library(programs, groups=None):
    by_task = {}
    for group, program in zip(groups if groups is not None else range(len(programs)), programs):
        program = simplify(program)
        fragments = {tuple(tuple(op) for op in program[a:b])
                     for a in range(len(program)) for b in range(a+2, len(program)+1)}
        by_task.setdefault(group, set()).update(fragments)
    counts = Counter(fragment for fragments in by_task.values() for fragment in sorted(fragments, key=repr))
    return [dict(program=[list(op) for op in fragment], support=count)
            for fragment, count in counts.most_common() if count >= 2]
