"""Fit a categorical paint rule conditioned on two stored numerical regions."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rc8'))
from memory import Frame
from fragments import library, matches


def regions(bank):
    parts = {}
    for canonical, terms, idx in library(bank):
        # A threshold and its complement induce the same partition.
        for j, t, _ in terms:
            parts.setdefault((canonical, ((j, t, True),)), idx)
        if len(terms) > 1:
            parts.setdefault((canonical, terms), idx)
    return {c: sorted([(terms, idx) for (flag, terms), idx in parts.items() if flag == c],
                      key=lambda p: (len(p[0]), p[1], p[0])) for c in (False, True)}


def partitions(frames, parts):
    masks, specs, seen = [], [], set()
    for mask, terms, idx in matches(frames, parts):
        signature = np.packbits(mask ^ mask[0]).tobytes()
        if signature in seen or mask.all() or not mask.any():
            continue
        masks.append(mask)
        specs.append((idx, terms))
        seen.add(signature)
    return np.array(masks, dtype=bool), specs


def compatible(keys, target, masks):
    """Find all second masks that separate conflicting labels within each key."""
    possible = np.arange(len(masks))
    conflict = False
    for key in np.unique(keys):
        positions = np.flatnonzero(keys == key)
        colors = np.unique(target[positions])
        if len(colors) == 1:
            continue
        conflict = True
        if len(colors) > 2:
            return conflict, []
        positive = target[positions] == colors[0]
        i, j = positions[positive][0], positions[~positive][0]
        possible = possible[masks[possible, i] != masks[possible, j]]
        if not len(possible):
            return conflict, []
        selected = masks[np.ix_(possible, positions)]
        ok = (selected == positive).all(axis=1) | (selected != positive).all(axis=1)
        possible = possible[ok]
        if not len(possible):
            break
    return conflict, possible


def solve(task, parts, bank):
    demos = task['train']
    if any(np.shape(e['input']) != np.shape(e['output']) for e in demos):
        return
    for k in (0, 4, 1, 2, 3, 5, 6, 7):
        for canonical in (False, True):
            frames = [Frame(e['input'], canonical, k) for e in demos + task['test']]
            y = np.concatenate([f.target(e['output']).ravel() for f, e in zip(frames, demos)])
            g = np.concatenate([f.g.ravel() for f in frames])
            n = len(y)
            masks, specs = partitions(frames, parts[canonical])
            if not len(masks):
                continue
            _, single = compatible(g[:n], y, masks)
            single = set(single)
            # Score both the single-region control and strictly joint rules.
            for i, first in enumerate(masks):
                keys = g + 10 * first
                if i in single:
                    partners = [None]
                else:
                    conflict, partners = compatible(keys[:n], y, masks)
                    if not conflict:
                        continue
                    partners = [j for j in partners if j > i and j not in single
                                and set(bank[specs[i][0]]['origins']).isdisjoint(bank[specs[j][0]]['origins'])]
                for j in partners:
                    pair = j is not None
                    joint = keys + 20 * masks[j] if pair else keys
                    lut = np.full(40, -1)
                    lut[joint[:n]] = y
                    guessed = lut[joint[n:]]
                    # Use the unchanged input for unseen combinations.
                    out = np.where(guessed < 0, g[n:], guessed)
                    result, pos = [], 0
                    for f in frames[len(demos):]:
                        result.append(f.decode(out[pos:pos + f.g.size].reshape(f.g.shape)).tolist())
                        pos += f.g.size
                    members = [specs[i], specs[j]] if pair else [specs[i]]
                    # Prefer simple, well-supported explanations.
                    cost = (sum(len(terms) for _, terms in members), int((guessed < 0).sum()), k, canonical)
                    yield cost, ['joint' if pair else 'single', members, k, canonical], result
