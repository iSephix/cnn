"""Retrospective oracle on gained tasks. Never called by inference."""
from pathlib import Path
import json,sys
import numpy as np
from memory import Frame,fit_binding,read

def check(task,bank):
    demos=task['train']
    queries=task['test']
    gates=sorted({tuple(p) for b in bank for p in b['card']['predicates']})
    if any(np.shape(e['output'])!=np.shape(e['input']) for e in demos+queries):return None
    simple=single=False
    for k in range(8):
        for canonical in (False,True):
            fs=[Frame(e['input'],canonical,k) for e in demos+queries]
            x=np.concatenate([f.features for f in fs])
            g=np.concatenate([f.g.ravel() for f in fs])
            y=np.concatenate([f.target(e['output']).ravel() for f,e in zip(fs,demos+queries)])
            n=sum(f.g.size for f in fs[:len(demos)])
            for j,t in [(None,0),*gates]:
                keys=g if j is None else g+10*(x[:,j]>t)
                lut=fit_binding([keys[:n]],[y[:n]])
                if lut is None:continue
                out=lut[keys[n:]]
                if np.array_equal(np.where(out<0,g[n:],out),y[n:]):
                    if j is None:simple=True
                    else:single=True
    return dict(color_only=simple,single_gate=single)

if __name__=='__main__':
    folder,changes=sys.argv[1:]
    bank=read('rc8/bank.json')
    print(json.dumps({k:check(read(Path(folder)/(k+'.json')),bank) for k in read(changes)['gains']},indent=2))
