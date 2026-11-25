"""Use native synthesis as a mask solver and compose independently learned planes."""
from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from panels import search


def solve(task, binary):
    demos=task['train']
    n=len(demos)
    outputs=[np.array(e['output']) for e in demos]
    counts=np.bincount(np.concatenate([g.ravel() for g in outputs]),minlength=10)
    bg=int(counts.argmax())
    inks=[int(c) for c in np.flatnonzero(counts) if c!=bg]
    result=[{} for _ in task['test']]
    if len(inks)!=2 or not any(np.shape(e['input'])!=y.shape for e,y in zip(demos,outputs)):
        return [dict(status='inapplicable',candidates=[]) for _ in result]
    with TemporaryDirectory() as directory:
        path=Path(directory)/'plane.json'
        layers=[]
        for ink in inks:
            projected=dict(train=[dict(input=e['input'],output=(y==ink).astype(int).tolist())
                                  for e,y in zip(demos,outputs)],test=task['test'])
            path.write_text(json.dumps(projected))
            candidates=search(path,binary,2,1,5)
            planes=[]
            for row in candidates:
                planes.append([(score,np.array(grid,bool)) for score,_,grid in row['candidates']
                               if score>n-1 and all(c in (0,1) for line in grid for c in line)][:4])
            if not all(planes):
                break
            layers.append(planes)
        if len(layers)==len(inks):
            for row,a,b in zip(result,*layers):
                for (sa,ma),(sb,mb) in product(a,b):
                    if ma.shape!=mb.shape or np.any(ma&mb):
                        continue
                    g=np.where(ma,inks[0],np.where(mb,inks[1],bg)).tolist()
                    key=json.dumps(g)
                    score=(sa+sb)/2-.03
                    if key not in row or score>row[key][0]:
                        row[key]=[score,n,g]
    return [dict(status='ok',candidates=sorted(r.values(),key=lambda v:-v[0])) for r in result]
