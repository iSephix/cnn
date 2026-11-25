"""Complete a damaged field using a repeating lattice inferred from visible cells."""
import numpy as np


def recover(grid, ink):
    hole = grid == ink
    if not hole.any() or hole.all():
        return {}
    h, w = grid.shape
    y, x = np.indices(grid.shape)
    known = ~hole
    # A basis (a, shift), (0, b) allows staggered tiles and diagonal stripes.
    models = []
    for a in range(1, h + 1):
        for b in range(1, w + 1):
            if a * b > known.sum() or (a == h and b == w):
                continue
            models.append((a*b, a, b))
    answers = {}
    for _, a, b in sorted(models):
        for shift in range(b if a < h else 1):
            slot = (y % a) * b + (x - (y // a) * shift) % b
            table = np.full(a*b, -1)
            table[slot[known]] = grid[known]
            if np.any(table[slot[known]] != grid[known]) or np.any(table[slot[hole]] < 0):
                continue
            key = 'plain' if shift == 0 else 'skew'
            answers.setdefault(key, table[slot])
        if len(answers) == 2:
            break
    # Infer mirror axes from overlapping visible cells, including off-center axes.
    for axis in (0,1):
        view = grid if axis == 0 else grid.T
        rows = np.arange(view.shape[0])
        best = None
        for center in range(1,2*len(rows)-2):
            reflected = center-rows
            valid = (reflected>=0) & (reflected<len(rows))
            if np.any(view[~valid]==ink):
                continue
            source, donor = view[valid], view[reflected[valid]]
            shared = (source!=ink) & (donor!=ink)
            if np.any(source[shared]!=donor[shared]) or np.any((source==ink)&(donor==ink)):
                continue
            support = int(shared.sum())
            if support and (best is None or support>best[0]):
                result = view.copy()
                result[valid] = np.where(source==ink,donor,source)
                best = support, result
        if best is not None:
            answers['mirror'+str(axis)] = best[1] if axis==0 else best[1].T
    return answers


def forms(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    present = np.flatnonzero(counts)
    return {**{('ink', int(c)): c for c in present},
            'rare': present[counts[present].argmin()], 'common': counts.argmax()}


def rules(task):
    examples = task['train']
    cache = {}
    def proposals(grid, selector):
        ink = forms(grid).get(selector)
        if ink is None:
            return {}
        key = grid.shape, grid.tobytes(), int(ink)
        if key not in cache:
            cache[key] = recover(grid, ink)
        y, x = np.where(grid == ink)
        out = {}
        for model, full in cache[key].items():
            out[model, 'grid'] = full
            out[model, 'patch'] = full[y.min():y.max()+1, x.min():x.max()+1]
        return out
    first = np.array(examples[0]['input'])
    for selector in forms(first):
        possible = None
        for e in examples:
            g, target = np.array(e['input']), np.array(e['output'])
            ink = forms(g).get(selector)
            if ink is None:
                possible = set()
                break
            hole = g == ink
            y, x = np.where(hole)
            if g.shape == target.shape:
                if np.any(g[~hole] != target[~hole]) or np.array_equal(g, target):
                    possible = set()
                    break
            elif target.shape != (np.ptp(y)+1, np.ptp(x)+1):
                possible = set()
                break
            fits = {k for k,v in proposals(g,selector).items() if np.array_equal(v,target)}
            possible = fits if possible is None else possible & fits
            if not possible:
                break
        for k in sorted(possible or ()):
            out = [proposals(np.array(q['input']),selector).get(k) for q in task['test']]
            if all(v is not None for v in out):
                yield 1 + int(k[0]=='skew'), out
