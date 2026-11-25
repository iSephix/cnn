"""Estimate candidate reliability using held-out demonstration evidence."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rc23'))
from calibrate import features, key, read


def queries(args):
    baseline=read(args.baseline)
    for path in sorted(args.blind.glob('*.json')):
        t=read(path)
        if any('output' in q for q in t['test']):
            raise ValueError('Expected unlabeled queries')
        truth=read(args.labels/path.name)['test'] if args.labels else [None]*len(t['test'])
        bank=read(args.verified/path.name)
        palette=read(args.palette/path.name)
        rows=baseline[path.stem]
        if not len(bank)==len(palette)==len(rows)==len(t['test'])==len(truth):
            raise ValueError(f'Query count mismatch for {path.stem}')
        for q,verified,recoded,prior,label in zip(t['test'],bank,palette,rows,truth):
            claims={key(c['grid']):dict(c) for c in verified}
            for item in recoded:
                value=claims.setdefault(key(item['grid']),dict(grid=item['grid'],detectors={},
                    orientations=[],identity_votes=0,full_fit=False,folds=[],fold_detectors={}))
                value.update(item)
            candidates=list(claims.values())
            vector=features(t['train'],q['input'],prior,candidates)
            n=len(t['train'])
            for grid,attrs in vector.items():
                c=claims.get(grid,{})
                folds=len(c.get('folds',[]))
                attrs.update({'fold_coverage':folds/n,'all_folds':int(folds==n),
                              'fold_families':math.log1p(len(c.get('fold_detectors',{}))),
                              'full_fit':int(c.get('full_fit',False)),
                              'palette_fits':c.get('palette_fits',0),
                              'palette_families':math.log1p(len(c.get('palette_detectors',{})))})
            yield path.stem,vector,label


def fit(args):
    x=[]
    y=[]
    task_ids=[]
    batches=[]
    for task,candidates,expected in queries(args):
        batches.append((task,slice(len(x),len(x)+len(candidates))))
        for grid,attributes in candidates.items():
            x.append(attributes)
            y.append(grid==key(expected['output']))
            task_ids.append(task)
    vectorizer=DictVectorizer()
    data=vectorizer.fit_transform(x)
    y=np.array(y)
    if args.metrics:
        scores=np.empty(len(y))
        for training,held_out in GroupKFold(5).split(data,y,task_ids):
            fold=LogisticRegression(C=.3,max_iter=400).fit(data[training],y[training])
            scores[held_out]=fold.decision_function(data[held_out])
        outcomes={}
        for task,batch in batches:
            attempts=np.argsort(-scores[batch],kind='stable')[:2]
            outcomes[task]=outcomes.get(task,True) and bool(y[batch][attempts].any())
        args.metrics.write_text(json.dumps(dict(tasks=len(outcomes),two_attempts=sum(outcomes.values())),
                                           indent=2)+'\n')
    model=LogisticRegression(C=.3,max_iter=400).fit(data,y)
    weights=dict(zip(vectorizer.get_feature_names_out(),map(float,model.coef_[0])))
    args.model.write_text(json.dumps(weights,separators=(',',':'))+'\n')


def predict(args):
    weights=read(args.model)
    data={}
    for task,candidates,_ in queries(args):
        def score(grid):
            return sum(weights.get(k,0)*v for k,v in candidates[grid].items())
        ordered=sorted(candidates,key=score,reverse=True)
        data.setdefault(task,[]).append(dict(attempt_1=ordered[0],attempt_2=ordered[min(1,len(ordered)-1)]))
    args.out.write_text(json.dumps(data,separators=(',',':'))+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('blind','baseline','verified','palette','model'):
        parser.add_argument(name,type=Path)
    kind=parser.add_mutually_exclusive_group(required=True)
    kind.add_argument('--labels',type=Path)
    kind.add_argument('--out',type=Path)
    parser.add_argument('--metrics',type=Path)
    a=parser.parse_args()
    fit(a) if a.labels else predict(a)
