"""Retrospective coverage of RC3-only families. Never called by prediction."""
import json,sys
from pathlib import Path
import numpy as np
from learn import objects,pixels,Scene

def measure(folder):
    totals={'tasks':0,'fitted_and_executable':0,'oracle_all_queries_exact':0}
    for file in sorted(Path(folder).glob('*.json')):
        t=json.loads(file.read_text())
        xs=[Scene(e['input']) for e in t['train']]
        ys=[np.array(e['output']) for e in t['train']]
        qs=[Scene(e['input']) for e in t['test']]
        fits=False
        hit=False
        for learner in (objects,pixels):
            for _,_,outs in learner(xs,ys,qs):
                fits=True
                if all(a.tolist()==e['output'] for a,e in zip(outs,t['test'])):
                    hit=True
                    break
            if hit:
                break
        totals['tasks']+=1
        totals['fitted_and_executable']+=fits
        totals['oracle_all_queries_exact']+=hit
    return totals

if __name__=='__main__':
    print(json.dumps(measure(sys.argv[1])))
