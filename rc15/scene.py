"""Run RC14 object classification and pattern completion inside selected views."""
import sys
from pathlib import Path

import numpy as np
from compose import views

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rc14'))
import objects
import repair
import panels


def rules(task):
    records=task['train']+task['test']
    scenes=[dict(views(np.array(q['input']))) for q in records]
    n=len(task['train'])
    seen=set()
    for key in scenes[0]:
        if key[0] not in ('object','panel','rank panel','inside') or any(key not in s for s in scenes):
            continue
        grids=[s[key] for s in scenes]
        signature=tuple((g.shape,g.tobytes()) for g in grids)
        if signature in seen:
            continue
        seen.add(signature)
        paired=[dict(input=g.tolist(),output=e['output']) for g,e in zip(grids,task['train'])]
        latent=dict(train=paired,test=[dict(input=g.tolist()) for g in grids[n:]])
        same=all(g.shape==np.shape(e['output']) for g,e in zip(grids,paired))
        families=(objects.rules,repair.rules) if same else (panels.rules,)
        for family in families:
            for cost,outputs in family(latent):
                yield 5+len(key)+cost,outputs
