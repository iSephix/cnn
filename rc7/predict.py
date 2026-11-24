import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rc3'))
from scene import Scene
from rules import strokes,trees
from repair import repairs
from patches import patches

def infer(xs,ys,qs):
    train=[Scene(x) for x in xs]
    query=[Scene(x) for x in qs]
    for learner in (repairs,strokes,trees,patches):
        for cost,rule,answers in learner(train,ys,query):
            yield cost,rule,[a.tolist() for a in answers]


def run(folder,prev,out):
    read=lambda p:json.loads(Path(p).read_text())
    baseline=read(prev)
    traces=read(prev+'.trace.json')
    pred={}
    for path in sorted(Path(folder).glob('*.json')):
        t=read(path)
        if any('output' in e for e in t['test']):
            raise ValueError('Query labels in input')
        inputs=[np.array(e['input']) for e in t['train']]
        labels=[np.array(e['output']) for e in t['train']]
        queries=[np.array(e['input']) for e in t['test']]
        prior=baseline[path.stem]
        candidates=sorted(infer(inputs,labels,queries),key=lambda r:r[0])
        result=[]
        for i,p in enumerate(prior):
            options=[(p[f'attempt_{j}'],r) for j,r in enumerate(traces[path.stem][i],1) if r!='identity']
            options.extend((a[i],r) for _,r,a in candidates)
            options.append((queries[i].tolist(),'identity'))
            choices=[]
            for g,r in options:
                if not choices or g!=choices[0][0]:
                    choices.append((g,r))
                if len(choices)==2:
                    break
            if len(choices)==1:
                choices*=2
            result.append(dict(attempt_1=choices[0][0],attempt_2=choices[1][0]))
            traces[path.stem][i]=[r for _,r in choices]
        pred[path.stem]=result
    for name,data in [(out,pred),(out+'.trace.json',traces)]:
        Path(name).write_text(json.dumps(data,separators=(',',':'))+'\n')

if __name__=='__main__':
    run(*sys.argv[1:])
