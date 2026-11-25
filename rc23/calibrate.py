"""Calibrate evidence from the verified detector bank against the existing solver."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from select_attempts import read, select

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rc12'))
from critic import describe


def key(grid):
    return tuple(map(tuple,grid))


def features(demos,query,baseline,bank):
    first=key(baseline['attempt_1'])
    second=key(baseline['attempt_2'])
    entries={first:None,second:None}
    ranks={}
    for index,c in enumerate(bank):
        grid=key(c['grid'])
        entries[grid]=c
        ranks[grid]=index
    examples=np.array([describe(e['input'],e['output']) for e in demos])
    constants=np.ptp(examples,axis=0)<1e-9
    palette=lambda rows:set(c for row in rows for c in row)
    query_shape=len(query),len(query[0])
    same_shape=all((len(e['input']),len(e['input'][0]))==(len(e['output']),len(e['output'][0])) for e in demos)
    allowed=palette(query).union(*(palette(e['output'])-palette(e['input']) for e in demos))
    result={}
    for grid,evidence in entries.items():
        f={'first':int(grid==first),'second':int(grid==second),'bank':int(evidence is not None)}
        delta=np.abs(examples-describe(query,grid))
        f['stable_ok']=np.count_nonzero(constants&(delta[0]<1e-9))/max(1,int(constants.sum()))
        f['closest']=float(delta.mean(axis=1).min())
        f['unexpected']=len(palette(grid)-allowed)
        f['wrong-size']=int(same_shape and (len(grid),len(grid[0]))!=query_shape)
        f['alternatives']=math.log1p(len(bank))
        f['unchanged']=int(grid==key(query))
        if evidence:
            modes=Counter(d.rsplit('.',1)[0] for d in evidence['detectors'])
            f['identity_votes']=math.log1p(evidence['identity_votes'])
            f['bank_rank']=1/(1+ranks[grid])
            f['bank_top']=int(ranks[grid]==0)
            f['orientations']=math.log1p(len(evidence['orientations']))
            f['detectors']=math.log1p(len(evidence['detectors']))
            f['families']=math.log1p(len(modes))
            for member in modes:
                f[f'module/{member}']=1
            for name in evidence['detectors']:
                f[f'detector/{name}']=1
        result[grid]=f
    return result


def queries(args):
    baseline=read(args.baseline)
    for path in sorted(args.blind.glob('*.json')):
        t=read(path)
        if any('output' in q for q in t['test']):
            raise ValueError('Expected unlabeled queries')
        candidates=read(args.bank/path.name)
        observed=baseline[path.stem]
        if not len(candidates)==len(observed)==len(t['test']):
            raise ValueError(f'Mismatched queries for {path.stem}')
        labels=read(args.labels/path.name)['test'] if args.labels else None
        for i,(q,b,c) in enumerate(zip(t['test'],observed,candidates)):
            target=labels[i]['output'] if labels else None
            vec=features(t['train'],q['input'],b,c)
            yield path.stem,vec,target,b,c


def fit(args):
    xs=[]
    ys=[]
    ids=[]
    groups=[]
    controls={mode:{} for mode in ('baseline','reserve','identity')}
    for task,choices,target,baseline,bank in queries(args):
        groups.append((task,slice(len(xs),len(xs)+len(choices))))
        for g,f in choices.items():
            xs.append(f)
            ys.append(key(target)==g)
            ids.append(task)
        for mode,claims in controls.items():
            p=baseline if mode=='baseline' else select([baseline],[bank],mode)[0]
            claims[task]=claims.get(task,True) and target in p.values()
    vectorizer=DictVectorizer()
    x=vectorizer.fit_transform(xs)
    y=np.array(ys)
    scores=np.empty(len(y))
    for train,test in GroupKFold(5).split(x,y,ids):
        learner=LogisticRegression(C=.3,max_iter=400).fit(x[train],y[train])
        scores[test]=learner.decision_function(x[test])
    validation={}
    for task,batch in groups:
        top=np.argsort(-scores[batch],kind='stable')[:2]
        validation[task]=validation.get(task,True) and bool(y[batch][top].any())
    final=LogisticRegression(C=.3,max_iter=400).fit(x,y)
    values=dict(zip(vectorizer.get_feature_names_out(),map(float,final.coef_[0])))
    args.model.write_text(json.dumps(values,separators=(',',':'))+'\n')
    print(json.dumps(dict(tasks=len(validation),learned=sum(validation.values()),
                          **{name:sum(v.values()) for name,v in controls.items()})))


def run(args):
    model=read(args.model)
    output={}
    for task,choices,_,_,_ in queries(args):
        score=lambda g:sum(model.get(name,0)*value for name,value in choices[g].items())
        top=sorted(choices,key=score,reverse=True)
        output.setdefault(task,[]).append(dict(attempt_1=top[0],attempt_2=top[min(1,len(top)-1)]))
    args.out.write_text(json.dumps(output,separators=(',',':'))+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for field in ('blind','baseline','bank','model'):
        p.add_argument(field,type=Path)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--labels',type=Path)
    mode.add_argument('--out',type=Path)
    a=p.parse_args()
    fit(a) if a.labels else run(a)
