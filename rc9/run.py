"""Extend RC8 guesses and emit paired and single-region ablations."""
from pathlib import Path
import json
import sys
from combine import regions, solve
from memory import read
from geometry import infer as geometry
from tiles import infer as tiles


def write(path, obj):
    Path(path).write_text(json.dumps(obj, separators=(',', ':')) + '\n')


def run(folder, prior, out, stage="cards"):
    base = read(prior)
    traces = read(prior + '.trace.json')
    bank = read(Path(__file__).resolve().parents[1] / 'rc8/bank.json')
    parts = regions(bank)
    modes = ('single', 'joint', 'both') if stage == 'cards' else (stage,)
    predictions = {m: {} for m in modes}
    logs = {m: {} for m in modes}
    for path in sorted(Path(folder).glob('*.json')):
        task = read(path)
        if any('output' in q for q in task['test']):
            raise ValueError('Test labels present')
        fits = []
        if any('identity' in tags for tags in traces[path.stem]):
            # Keep only the cheapest explanation for each distinct prediction per family.
            unique = {}
            candidates = solve(task, parts, bank) if stage == "cards" else {"geometry": geometry, "tiles": tiles}[stage](task)
            for cost, tag, grids in candidates:
                key = tag[0], json.dumps(grids)
                if key not in unique or cost < unique[key][0]:
                    unique[key] = cost, tag, grids
            fits = sorted(unique.values(), key=lambda f: f[0])
        for mode in modes:
            output, trace = [], []
            for i, q in enumerate(task['test']):
                choices = [(base[path.stem][i][f'attempt_{j}'], tag)
                           for j, tag in enumerate(traces[path.stem][i], 1) if tag != 'identity']
                choices.extend((grids[i], tag) for _, tag, grids in fits if mode in ('both', tag[0]))
                choices.append((q['input'], 'identity'))
                a = choices[0]
                b = next((p for p in choices if p[0] != a[0]), a)
                output.append(dict(attempt_1=a[0], attempt_2=b[0]))
                trace.append([a[1], b[1]])
            predictions[mode][path.stem] = output
            logs[mode][path.stem] = trace
    for m in modes:
        write(out + '-' + m + '.json', predictions[m])
        write(out + '-' + m + '.json.trace.json', logs[m])


if __name__ == '__main__':
    run(*sys.argv[1:])
