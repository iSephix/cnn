"""Read labels only after predictions are saved. Attribute every gained or lost task."""
import json,sys
from pathlib import Path

def load(p):
    return json.loads(Path(p).read_text())

def audit(split, results, n):
    root=Path(results)
    pred=load(root/f'arc{n}.json')
    sources=load(root/f'arc{n}.json.trace.json')
    baseline=load(f'rc2/results/arc{n}-reuse-512-score.json')
    old={r['id']:all(r['hits']) for r in baseline['rows']}
    gains={}
    losses=[]
    for p in sorted(Path(split).glob('*.json')):
        task=load(p)
        rules=[]
        for i,e in enumerate(task['test']):
            hit=next((k for k in (1,2) if pred[p.stem][i][f'attempt_{k}']==e['output']),None)
            if hit is None:
                break
            rules.append(sources[p.stem][i][hit-1])
        solved=len(rules)==len(task['test'])
        if solved and not old[p.stem]:
            gains[p.stem]=rules
        if old[p.stem] and not solved:
            losses.append(p.stem)
    return {'gained':gains,'lost':losses}

if __name__=='__main__':
    print(json.dumps(audit(*sys.argv[1:]),indent=2))
