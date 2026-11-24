"""Identify every changed task and the exact attempt that solved it."""
import json,sys
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())

def audit(oldfile,newfile,traces):
    old={r['id']:all(r['hits']) for r in read(oldfile)['rows']}
    new={r['id']:r['hits'] for r in read(newfile)['rows']}
    if old.keys()!=new.keys():raise ValueError('Different task sets')
    wins={k for k,h in new.items() if all(h)}
    prior={k for k,hit in old.items() if hit}
    trace=read(traces)
    return {'gains':{k:[trace[k][i][h-1] for i,h in enumerate(new[k])] for k in sorted(wins-prior)},
            'losses':sorted(prior-wins)}

if __name__=='__main__':print(json.dumps(audit(*sys.argv[1:]),indent=2))
