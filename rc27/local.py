"""Learn object-relative paint actions and test them on held-out demonstrations."""
from itertools import product

import numpy as np
from scipy import ndimage
from sklearn.tree import DecisionTreeClassifier

from concepts import background, components


def shifted(values, row, column, outside):
    output = np.full(values.shape, outside, dtype=values.dtype)
    h, w = values.shape
    if abs(row) < h and abs(column) < w:
        output[max(0,row):min(h,h+row), max(0,column):min(w,w+column)] = values[
            max(0,-row):min(h,h-row), max(0,-column):min(w,w-column)]
    return output


def measurements(grid, bg_mode):
    bg = background(grid, bg_mode)
    foreground = grid != bg
    features = [grid, foreground.astype(int)]
    for radius in (1, 2):
        for row, column in product((-radius, 0, radius), repeat=2):
            if row == column == 0:
                continue
            neighbor = shifted(grid, row, column, -1)
            features.extend((neighbor == grid, neighbor == bg))
    for metric in ('taxicab', 'chessboard'):
        # Pad so the implicit outside background participates in the metric.
        distance = ndimage.distance_transform_cdt(np.pad(foreground, 1), metric=metric)[1:-1,1:-1]
        features.append(distance)
    holes = ndimage.binary_fill_holes(foreground) & ~foreground
    features.extend((holes, ndimage.binary_dilation(holes),
                     ndimage.binary_dilation(~(holes | foreground))))
    for separated in (True, False):
        fields = [np.zeros(grid.shape, dtype=int) for _ in range(8)]
        for obj in components(grid, bg, separated=separated, diagonal=1):
            box, mask = obj['box'], obj['mask']
            h, w = mask.shape
            r, c = np.indices(mask.shape)
            attributes = [r, c, h-1-r, w-1-c, r*12//h, c*12//w,
                          np.full(mask.shape, obj['area']), np.full(mask.shape, h*w-obj['area'])]
            for field, values in zip(fields, attributes):
                view = field[box]
                view[mask] = values[mask]
        features.extend(fields)
    return np.column_stack([np.asarray(field, dtype=np.int32).ravel() for field in features])


def train(features, labels, leaves):
    tree = DecisionTreeClassifier(max_depth=8, max_leaf_nodes=leaves, random_state=0)
    tree.fit(features, labels)
    return tree if np.array_equal(tree.predict(features), labels) else None


def fit(task):
    if not task['train'] or not task['test'] or any('output' in q for q in task['test']):
        raise ValueError('Expected demonstrations and unlabeled queries')
    n = len(task['train'])
    images = [np.asarray(e['input'], dtype=np.int16) for e in task['train']+task['test']]
    targets = [np.asarray(e['output'], dtype=np.int16) for e in task['train']]
    if n < 2 or any(g.shape != y.shape for g, y in zip(images, targets)):
        return
    labels = [np.where(g == y, 10, y).ravel() for g, y in zip(images, targets)]
    if all(np.all(y == 10) for y in labels):
        return
    seen = {}
    for bg_mode in ('zero', 'dominant'):
        features = [measurements(g, bg_mode) for g in images]
        data = np.concatenate(features[:n])
        truth = np.concatenate(labels)
        for leaves in (6, 12, 20):
            tree = train(data, truth, leaves)
            if tree is None:
                continue
            folds = 0
            for index in range(n):
                others = [j for j in range(n) if j != index]
                heldout = train(np.concatenate([features[j] for j in others]),
                                np.concatenate([labels[j] for j in others]), leaves)
                folds += int(heldout is not None and
                             np.array_equal(heldout.predict(features[index]), labels[index]))
            outputs = []
            for grid, frame in zip(images[n:], features[n:]):
                actions = tree.predict(frame).reshape(grid.shape)
                outputs.append(np.where(actions == 10, grid, actions))
            key = tuple(g.tobytes() for g in outputs)
            strength = (folds, -int(tree.get_n_leaves()))
            if key not in seen or strength > seen[key]:
                seen[key] = strength
                yield dict(family='local', program=[bg_mode, leaves], cost=int(tree.get_n_leaves()),
                           folds=folds, demonstrations=n,
                           tree=dict(children_left=tree.tree_.children_left.tolist(),
                                     children_right=tree.tree_.children_right.tolist(),
                                     features=tree.tree_.feature.tolist(),
                                     thresholds=tree.tree_.threshold.tolist(),
                                     actions=tree.classes_[tree.tree_.value[:,0,:].argmax(axis=1)].tolist()),
                           grids=[g.tolist() for g in outputs])
