import json,sys
from pathlib import Path

def read(p):
    return json.loads(Path(p).read_text())

for n in (1,2):
    root=Path(sys.argv[1])
    prior={r['id']:all(r['hits']) for r in read(f'rc4/results/arc{n}-score.json')['rows']}
    rows=read(root/f'arc{n}-score.json')['rows']
    trace=read(root/f'arc{n}.json.trace.json')
    gains={r['id']:[trace[r['id']][j][h-1] for j,h in enumerate(r['hits'])]
           for r in rows if all(r['hits']) and not prior[r['id']]}
    losses=[r['id'] for r in rows if prior[r['id']] and not all(r['hits'])]
    (root/f'arc{n}-change.json').write_text(json.dumps({'gains':gains,'losses':losses},indent=2)+'\n')
    print(n,gains,losses)
