"""Rank the full portfolio using shared transformation evidence and candidate disagreement."""
import argparse
import base64
from functools import lru_cache
import json
from pathlib import Path
import zlib
import sys

import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GroupKFold

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rc12'))
from rank import read, rows


@lru_cache(maxsize=18000)
def scene(grid):
    g=np.array(grid)
    colors,counts=np.unique(g,return_counts=True)
    bg=colors[counts.argmax()]
    foreground=g!=bg
    regions,n=ndimage.label(foreground)
    sizes=np.bincount(regions[foreground],minlength=n+1)[1:]
    y,x=np.where(foreground)
    h,w=g.shape
    return np.array([h,w,len(colors),n,foreground.sum(),int(sizes.max()) if n else 0,
                     np.ptp(y)+1 if n else 0,np.ptp(x)+1 if n else 0,
                     len({tuple(r) for r in g}),len({tuple(c) for c in g.T}),
                     int(np.array_equal(g,g[::-1])),int(np.array_equal(g,g[:,::-1])),
                     int(np.array_equal(g,g[::-1,::-1])),int(np.array_equal(g,g.T)),
                     sum(ndimage.label(g==c)[1] for c in colors if c!=bg),
                     np.count_nonzero(ndimage.binary_fill_holes(foreground)&~foreground)],dtype=float)


def parameters(x,y):
    a,b=scene(tuple(map(tuple,x))),scene(tuple(map(tuple,y)))
    before,after=np.array(x),np.array(y)
    shape=before.shape==after.shape
    colors=[set(v.ravel()) for v in (before,after)]
    return np.r_[b,b-a,(b+1)/(a+1),float(shape),
                 len(colors[0]-colors[1]),len(colors[1]-colors[0]),
                 np.mean(before!=after) if shape else 1.]


def augment(choices,task,index,legacy):
    examples=np.array([parameters(d['input'],d['output']) for d in task['train']])
    constant=np.ptp(examples,axis=0)<1e-9
    keys=list(choices)
    scores=[sum(legacy.get(k,0)*v for k,v in f.items()) for f in choices.values()]
    best=max(scores)
    shape_counts={}
    for key in keys:
        g=json.loads(key)
        size=(len(g),len(g[0]))
        shape_counts[size]=shape_counts.get(size,0)+1
    for key,score in zip(keys,scores):
        f=choices[key]
        g=json.loads(key)
        values=parameters(task['test'][index]['input'],g)
        diffs=np.abs(examples-values)
        stable=np.flatnonzero(constant)
        f.update({f'invariant/{j}':float(diffs[0,j]<1e-9) for j in stable})
        f.update({f'range/{j}':float(v<low-1e-9 or v>high+1e-9)
                  for j,(v,low,high) in enumerate(zip(values,examples.min(axis=0),examples.max(axis=0)))})
        f['legacy/score']=score
        f['legacy/gap']=best-score
        f['legacy/rank']=sum(other>score+1e-9 for other in scores)
        f['competition/size']=len(choices)
        f['competition/shape_votes']=shape_counts[len(g),len(g[0])]
        f['invariant/fraction']=float(np.mean(diffs[0,stable]<1e-9)) if len(stable) else 0.
    return choices


def examples(args):
    prior=read(args.prior)
    traces=read(str(args.prior)+'.trace.json')
    legacy=read(args.legacy)
    for path in sorted(args.blind.glob('*.json')):
        task=read(path)
        if any('output' in q for q in task['test']):
            raise ValueError('Expected unlabeled queries')
        sources=[read(d/path.name) for d in args.native]
        pool=read(args.pool/path.name)
        labels=read(args.labels/path.name)['test'] if args.labels else None
        for i in range(len(task['test'])):
            choices=rows(task,prior[path.stem][i],traces[path.stem][i],sources,pool,i)
            choices=augment(choices,task,i,legacy)
            yield path.stem,choices,labels[i]['output'] if labels else None


def make_model():
    return HistGradientBoostingClassifier(max_iter=125,learning_rate=.065,max_leaf_nodes=13,
                                          min_samples_leaf=30,l2_regularization=3,random_state=19)


def choose(scores):
    idx=np.argsort(-scores,kind='stable')
    return idx[0],idx[min(1,len(idx)-1)]


def train(args):
    rows_list=[]
    labels=[]
    task_ids=[]
    groups=[]
    for task,choices,target in examples(args):
        end=len(rows_list)
        rows_list.extend(choices.values())
        labels.extend(int(json.loads(k)==target) for k in choices)
        task_ids.extend([task]*len(choices))
        groups.append((task,slice(end,len(rows_list))))
    vectorizer=DictVectorizer(sparse=False,dtype=np.float32)
    x=vectorizer.fit_transform(rows_list)
    y=np.array(labels)
    legacy=x[:,vectorizer.vocabulary_['legacy/score']]
    oof=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(x,y,task_ids):
        oof[te]=make_model().fit(x[tr],y[tr]).decision_function(x[te])
    outcomes={}
    for weight in (0,.25,.5,.75,1):
        totals={}
        for task,sub in groups:
            hit=any(y[sub][i] for i in choose((1-weight)*legacy[sub]+weight*oof[sub]))
            totals[task]=totals.get(task,True) and hit
        outcomes[weight]=sum(totals.values())
    weight=max(outcomes,key=outcomes.get)
    model=make_model().fit(x,y)
    trees=[]
    for (tree,) in model._predictors:
        trees.append([[float(node['value']),int(node['feature_idx']),
                       float(node['num_threshold']),int(node['left']),int(node['right']),
                       bool(node['is_leaf'])] for node in tree.nodes])
    data=dict(features=list(vectorizer.get_feature_names_out()),trees=trees,
              bias=float(model._baseline_prediction.item()),weight=weight)
    expected=model.decision_function(x[:500])
    if not np.allclose(expected,forest_scores(x[:500],data)):
        raise ValueError('Exported model differs from the fitted ranker')
    payload=zlib.compress(json.dumps(data,separators=(',',':')).encode(),9)
    args.model.write_text(base64.b85encode(payload).decode()+'\n')
    print(json.dumps(dict(tasks=len(set(task_ids)),selected_weight=weight,weights=outcomes)))


def forest_scores(x,data):
    result=np.full(len(x),data['bias'])
    for tree in data['trees']:
        for i,row in enumerate(x):
            node=tree[0]
            while not node[-1]:
                _,feature,threshold,left,right,_=node
                node=tree[left if row[feature]<=threshold else right]
            result[i]+=node[0]
    return result


def predict(args):
    data=json.loads(zlib.decompress(base64.b85decode(args.model.read_text().strip())))
    features=data['features']
    weight=data['weight']
    legacy_index=features.index('legacy/score')
    out={}
    for task,choices,_ in examples(args):
        x=np.array([[row.get(name,0) for name in features] for row in choices.values()],
                   dtype=np.float32)
        scores=(1-weight)*x[:,legacy_index]+weight*forest_scores(x,data)
        keys=list(choices)
        first,second=choose(scores)
        out.setdefault(task,[]).append(dict(attempt_1=json.loads(keys[first]),attempt_2=json.loads(keys[second])))
    args.out.write_text(json.dumps(out,separators=(',',':'))+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('blind','prior','pool','legacy','model'):
        p.add_argument(name,type=Path)
    p.add_argument('native',nargs='+',type=Path)
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--labels',type=Path)
    modes.add_argument('--out',type=Path)
    a=p.parse_args()
    train(a) if a.labels else predict(a)
