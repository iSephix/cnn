"""Infer a grid of panels, then summarize each panel or select one by a statistic."""
import itertools
import numpy as np
from memory import Frame


def axis_parts(g, axis, rule):
    lines = g if axis == 0 else g.T
    n = len(lines)
    kind, arg = rule
    if kind == 'lines':
        keep = ~np.all(lines == arg, axis=1)
        ids = np.flatnonzero(keep)
        return [v for v in np.split(ids, np.flatnonzero(np.diff(ids) != 1)+1) if len(v)]
    count = arg if kind == 'count' else n // arg
    if count < 1 or n % count or (kind == 'size' and n % arg):
        return []
    return np.split(np.arange(n), count)


def descriptions(g, target, reduce):
    def options(axis):
        n = g.shape[axis]
        m = target.shape[axis]
        yield from [('lines',int(c)) for c in np.unique(g)]
        count = m if reduce else n//m
        if count and n%count == 0 and (reduce or n% m == 0):
            yield 'count', count
            yield 'size', n//count
    return itertools.product(options(0), options(1))


def boards(grids, rules):
    result = []
    for g in grids:
        rows, cols = (axis_parts(g,a,rules[a]) for a in (0,1))
        if not rows or not cols or len(rows)*len(cols)<2:
            return
        cells = [g[np.ix_(r,c)] for r in rows for c in cols]
        result.append(((len(rows),len(cols)), cells))
    return result


def stats(tile):
    counts = np.bincount(tile.ravel(), minlength=10)
    present = np.flatnonzero(counts)
    return [int(counts.argmax()), int(present[counts[present].argmin()]), len(present),
            int(tile[0,0]), int(tile[tile.shape[0]//2,tile.shape[1]//2]),
            int(counts.max()), *map(int,counts)]


def summaries(boards, targets):
    stats_by_board = [np.array([stats(tile) for tile in tiles]) for _, tiles in boards]
    target = np.concatenate([v.ravel() for v in targets])
    for i in range(stats_by_board[0].shape[1]):
        values = [a[:,i] for a in stats_by_board]
        for style in ('value','present','maximum','minimum'):
            if style == 'present':
                features = [v>0 for v in values]
            elif style == 'maximum':
                features = [v==v.max() for v in values]
            elif style == 'minimum':
                features = [v==v.min() for v in values]
            else:
                features = values
            keys = np.concatenate(features[:len(targets)]).astype(int)
            lut = dict(zip(keys,target))
            if any(lut[k] != y for k,y in zip(keys,target)):
                continue
            result = []
            for (shape,_),v in zip(boards[len(targets):],features[len(targets):]):
                if any(k not in lut and not (i in (0,1,3,4) and style=='value') for k in v):
                    break
                result.append(np.array([lut.get(k,k) for k in v]).reshape(shape))
            else:
                yield len(lut), ['reduce', i, style], result


def selections(boards, targets):
    metrics = []
    for _,tiles in boards:
        encoded = [(t.shape,t.tobytes()) for t in tiles]
        metrics.append(np.array([stats(t) + [encoded.count(key)] for t,key in zip(tiles,encoded)]))
    for i in range(metrics[0].shape[1]):
        for maximum in (False,True):
            picks = []
            for (_,tiles),m in zip(boards,metrics):
                col = m[:,i]
                best = np.flatnonzero(col==(col.max() if maximum else col.min()))
                if len(best)!=1:
                    break
                picks.append(tiles[best[0]])
            else:
                if all(np.array_equal(a,b) for a,b in zip(picks,targets)):
                    yield 1, ['select', i, maximum], picks[len(targets):]


def infer(task):
    demos = task['train']
    for canonical in (False, True):
        frames = [Frame(e['input'],canonical) for e in demos+task['test']]
        targets = [f.target(e['output']) for f,e in zip(frames,demos)]
        grids = [f.g for f in frames]
        if all(g.shape==t.shape for g,t in zip(grids,targets)):
            continue
        for reduce in (True,False):
            seen = set()
            for rules in descriptions(grids[0],targets[0],reduce):
                # Equivalent partitions induce the same panel programs.
                partition = tuple(tuple(tuple(v) for v in axis_parts(g,a,r))
                                  for g in grids for a,r in enumerate(rules))
                if partition in seen:
                    continue
                seen.add(partition)
                bs = boards(grids,rules)
                if not bs:
                    continue
                if reduce:
                    if any(s!=t.shape for (s,_),t in zip(bs,targets)):
                        continue
                    fits = summaries(bs,targets)
                else:
                    if any(not any(t.shape==tile.shape for tile in tiles) for (_,tiles),t in zip(bs,targets)):
                        continue
                    fits = selections(bs,targets)
                for cost,tag,outs in fits:
                    result = [f.decode(v).tolist() for f,v in zip(frames[len(demos):],outs)]
                    yield (cost, canonical), ['tiles', rules, tag, canonical], result
