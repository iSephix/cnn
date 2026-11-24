"""Calibrate competing grid predictions on training tasks and emit two distinct guesses."""
import argparse,json,math
from pathlib import Path
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from critic import describe,evidence


def read(path):
    return json.loads(Path(path).read_text())


def rows(task,prior,tags,natives,pool,i):
    choices={}
    def add(g,source,features):
        key=json.dumps(g,separators=(',',':'))
        f=choices.setdefault(key,{})
        f[source]=1
        f.update({source+'/'+k:v for k,v in features.items()})
    for j,tag in enumerate(tags,1):
        name=tag[0] if isinstance(tag,list) else tag
        add(prior[f'attempt_{j}'],'prior'+str(j),{str(name):1})
    for mode,records in enumerate(natives):
        for j,(score,n,g) in enumerate(records[i]['candidates']):
            fit=math.ceil(score)
            add(g,'search'+str(mode),{'rank':1/(1+j),'exact':int(fit==n),'fraction':fit/n,
                                      'simple':max(0,1-(fit-score))})
    for family,records in pool.items():
        for j,(cost,count,grids) in enumerate(records):
            add(grids[i],family,{'rank':1/(1+j),'support':math.log1p(count),
                                 'strong':int(cost[0]==0)})
    q=task['test'][i]['input']
    examples=np.stack([describe(e['input'],e['output']) for e in task['train']])
    for k,f in choices.items():
        g=json.loads(k)
        f.update(evidence(q,g,examples))
        f['native_votes']=sum(s.startswith('search') and '/' not in s for s in f)
        f['native_exact_votes']=sum(v for s,v in f.items() if s.startswith('search') and s.endswith('/exact'))
        f['program_votes']=sum(s in ('regions','latent','compose','rewrite','motion','exemplar') for s in f)
        demos=task['train']
        shapes=lambda a:(len(a),len(a[0]))
        f['shape_violation']=int(all(shapes(e['input'])==shapes(e['output']) for e in demos) and shapes(q)!=shapes(g))
        palette=lambda a:{c for row in a for c in row}
        new=set().union(*(palette(e['output'])-palette(e['input']) for e in demos))
        f['unexpected_colors']=len(palette(g)-palette(q)-new)
        f['sources']=math.log1p(sum('/' not in s for s in f))
        f['identity']=int(g==q)
        f['flat']=int(len(set(v for row in g for v in row))==1)
    return choices


def run(a):
    prior=read(a.prior)
    traces=read(str(a.prior)+'.trace.json')
    if not a.labels:
        model=read(a.model)
    X,Y=[],[]
    pred={}
    for path in sorted(a.blind.glob('*.json')):
        t=read(path)
        if any('output' in q for q in t['test']):
            raise ValueError('Test labels present')
        natives=[read(d/path.name) for d in a.native]
        pool=read(a.pool/path.name)
        out=[]
        answers=read(a.labels/path.name)['test'] if a.labels else None
        for i,q in enumerate(t['test']):
            groups=rows(t,prior[path.stem][i],traces[path.stem][i],natives,pool,i)
            if a.labels:
                target=answers[i]['output']
                for k,f in groups.items():
                    X.append(f)
                    Y.append(int(json.loads(k)==target))
            else:
                def score(k):
                    return sum(model.get(s,0)*v for s,v in groups[k].items())
                best=sorted(groups,key=score,reverse=True)
                out.append(dict(attempt_1=json.loads(best[0]),attempt_2=json.loads(best[min(1,len(best)-1)])))
        pred[path.stem]=out
    if a.labels:
        vec=DictVectorizer()
        clf=LogisticRegression(C=.5,max_iter=500).fit(vec.fit_transform(X),Y)
        a.model.write_text(json.dumps(dict(zip(vec.get_feature_names_out(),clf.coef_[0])),indent=2)+'\n')
    else:
        a.out.write_text(json.dumps(pred,separators=(',',':'))+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('blind','prior','pool','model'):
        p.add_argument(name,type=Path)
    p.add_argument('native',nargs='+',type=Path)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--labels',type=Path)
    group.add_argument('--out',type=Path)
    run(p.parse_args())
