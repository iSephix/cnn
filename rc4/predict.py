import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rc3'))
import numpy as np
from scene import Scene
from placement import programs
from pattern import layouts,stamps,sweeps
from panels import panels


def solve(task,prior,origins):
    if any('output' in e for e in task['test']):
        raise ValueError('Strip query answers before inference')
    xs=[Scene(e['input']) for e in task['train']]
    ys=[np.array(e['output']) for e in task['train']]
    qs=[Scene(e['input']) for e in task['test']]
    best={}
    for family in (programs,sweeps,stamps,layouts,panels):
        for cost,rule,outs in family(xs,ys,qs):
            signature=tuple((a.shape,a.tobytes()) for a in outs)
            if signature not in best or best[signature][0]>cost:
                best[signature]=(cost,rule,outs)
    ranked=sorted(best.values(),key=lambda x:x[0])
    pred=[]
    trace=[]
    for i,q in enumerate(qs):
        pairs=[]
        def offer(g,r):
            if len(pairs)<2 and all(g!=p for p,_ in pairs):
                pairs.append((g,r))
        for j,source in enumerate(origins[i],1):
            if source!='identity':
                offer(prior[i][f'attempt_{j}'],source)
        for _,rule,outs in ranked:
            if len(pairs)>=2:
                break
            offer(outs[i].tolist(),rule)
        offer(q.grid.tolist(),'identity')
        if len(pairs)==1:
            pairs.append(pairs[0])
        pred.append({f'attempt_{k}':g for k,(g,_) in enumerate(pairs,1)})
        trace.append([r for _,r in pairs])
    return pred,trace,len(best)


def main(folder,base,out):
    load=lambda p:json.loads(Path(p).read_text())
    prev=load(base)
    origins=load(base+'.trace.json')
    preds,trace,counts={},{},{}
    for p in sorted(Path(folder).glob('*.json')):
        preds[p.stem],trace[p.stem],counts[p.stem]=solve(load(p),prev[p.stem],origins[p.stem])
    for name,data in [(out,preds),(out+'.trace.json',trace),(out+'.counts.json',counts)]:
        Path(name).write_text(json.dumps(data,separators=(',',':'))+'\n')

if __name__=='__main__':
    main(*sys.argv[1:])
