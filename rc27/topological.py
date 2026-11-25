"""Acquire paint rules in an object's intrinsic topology with scene-relative ink."""
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

from concepts import background

sys.path.append(str(Path(__file__).resolve().parents[1]/'rc25'))
from structured import topology


def frame(grid, mode):
    bg = background(grid, mode)
    mask = grid != bg
    fields = topology(mask.astype(int), 1)
    for radius in (2, 3):
        count = ndimage.convolve(mask.astype(int), np.ones((radius*2+1, radius*2+1), int),
                                 mode='constant', cval=0)
        fields['mass '+str(radius)] = count
    return mask, fields


def learn(frames, labels, keys):
    lookup = {}
    for (mask, fields), actions in zip(frames, labels):
        columns = [fields[key][mask] for key in keys]
        for code, action in zip(zip(*columns), actions[mask]):
            code = tuple(map(int, code))
            value = int(action)
            if code in lookup and lookup[code] != value:
                return
            lookup[code] = value
    return lookup


def render(grid, data, keys, lookup):
    mask, fields = data
    codes = zip(*(fields[key][mask] for key in keys))
    actions = np.array([lookup.get(tuple(map(int, code)), 10) for code in codes])
    output = grid.copy()
    original = grid[mask]
    output[mask] = np.where(actions == 10, original, actions)
    return output


def fit(task):
    n = len(task['train'])
    grids = [np.asarray(e['input']) for e in task['train']+task['test']]
    targets = [np.asarray(e['output']) for e in task['train']]
    if not n or not task['test'] or any('output' in q for q in task['test']):
        raise ValueError('Expected demonstrations and unlabeled queries')
    if any(g.shape != y.shape for g, y in zip(grids, targets)):
        return
    labels = [np.where(g == y, 10, y) for g, y in zip(grids, targets)]
    seen = {}
    for mode in ('zero', 'dominant'):
        frames = [frame(g, mode) for g in grids]
        if any(np.any((g != y) & ~data[0]) for g, y, data in zip(grids, targets, frames)):
            continue
        fields = list(frames[0][1])
        for keys in [*((key,) for key in fields), *combinations(fields, 2)]:
            lookup = learn(frames[:n], labels, keys)
            if lookup is None or len(lookup) > 60:
                continue
            if not any(action != 10 for action in lookup.values()):
                continue
            folds = 0
            for index in range(n):
                others = [j for j in range(n) if j != index]
                table = learn([frames[j] for j in others], [labels[j] for j in others], keys)
                if table is not None and np.array_equal(render(grids[index], frames[index], keys, table), targets[index]):
                    folds += 1
            outputs = [render(g, data, keys, lookup) for g, data in zip(grids[n:], frames[n:])]
            key = tuple(g.tobytes() for g in outputs)
            strength = (folds, -len(keys))
            if key not in seen or strength > seen[key]:
                seen[key] = strength
                yield dict(family='topological', program=[mode, list(keys),
                               [[list(code), action] for code, action in sorted(lookup.items())]],
                           cost=3+len(keys), folds=folds, demonstrations=n,
                           grids=[g.tolist() for g in outputs])
