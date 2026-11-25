"""Induce a decision tree whose leaves point to input-derived color sources."""
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rc10'))
from induce import Frame
from pointers import sources


def measurements(g):
    h,w=g.shape
    refs=np.column_stack([v.ravel() for _,v in sources(g)])
    bg=np.bincount(g.ravel()).argmax()
    y,x=np.indices(g.shape)
    occ=g!=bg
    labels,n=ndimage.label(occ)
    count=np.bincount(labels.ravel(),minlength=n+1)
    count[0]=0
    features=np.column_stack([
        refs,refs==refs[:,[0]],refs==bg,
        y.ravel(),x.ravel(),(y%2).ravel(),(x%2).ravel(),
        np.minimum(y,h-1-y).ravel(),np.minimum(x,w-1-x).ravel(),
        np.broadcast_to(occ.sum(axis=0),g.shape).ravel(),
        np.broadcast_to(occ.sum(axis=1)[:,None],g.shape).ravel(),
        count[labels].ravel(),
        ndimage.distance_transform_cdt(~occ,metric='taxicab').ravel()])
    return refs,features


def build(features, fits, order, max_depth):
    """Minimize wrong colors rather than errors against a fixed donor assignment."""
    def tree(indices,depth):
        match=fits[indices]
        hits=match.sum(axis=0)
        best=int(order[np.argmax(hits[order])])
        if hits[best]==len(indices):
            return best
        if not depth or len(indices)<2:
            return
        choice=None
        gain=0
        subset=features[indices]
        for column in range(features.shape[1]):
            values=subset[:,column]
            idx=np.argsort(values,kind='stable')
            boundaries=np.flatnonzero(values[idx][:-1]!=values[idx][1:])
            if not len(boundaries):
                continue
            upper=np.cumsum(match[idx],axis=0)[boundaries]
            score=upper.max(axis=1)+(hits-upper).max(axis=1)-hits[best]
            j=int(np.argmax(score))
            if score[j]>gain:
                gain=int(score[j])
                choice=column,int(values[idx[boundaries[j]]])
        if choice is None:
            return
        col,threshold=choice
        mask=subset[:,col]<=threshold
        left=tree(indices[mask],depth-1)
        right=tree(indices[~mask],depth-1)
        if left is not None and right is not None:
            return col,threshold,left,right
    return tree(np.arange(len(features)),max_depth)


def evaluate(rule, refs, features):
    if isinstance(rule,int):
        return refs[:,rule]
    col,threshold,left,right=rule
    take=features[:,col]<=threshold
    out=np.empty(len(take),dtype=refs.dtype)
    out[take]=evaluate(left,refs[take],features[take])
    out[~take]=evaluate(right,refs[~take],features[~take])
    return out


def leaves(tree):
    if isinstance(tree,int):
        return 1
    return leaves(tree[2])+leaves(tree[3])


def rules(task):
    demos=task['train']
    n=len(demos)
    if n<2 or any(np.shape(e['input'])!=np.shape(e['output']) for e in demos):
        return
    if all(e['input']==e['output'] for e in demos):
        return
    for canonical in (False,True):
        frames=[Frame(e['input'],canonical) for e in demos+task['test']]
        data=[measurements(frame.g) for frame in frames]
        labels=[frame.target(e['output']).ravel() for frame,e in zip(frames,demos)]
        valid=[refs==y[:,None] for y,(refs,_) in zip(labels,data)]
        fits=np.concatenate(valid)
        features=np.concatenate([x for _,x in data[:n]])
        order=np.arange(fits.shape[1])
        rule=build(features,fits,order,4)
        if rule is None:
            continue
        cost=leaves(rule)
        # Independent demonstration folds measure whether routing is transferable.
        failures=0
        for i,(refs,x) in enumerate(data[:n]):
            indices=[j for j in range(n) if j!=i]
            fold=build(np.concatenate([data[j][1] for j in indices]),
                       np.concatenate([valid[j] for j in indices]),order,4)
            if fold is None or not np.array_equal(evaluate(fold,refs,x),labels[i]):
                failures+=1
        predictions=[]
        for frame,(refs,x) in zip(frames[n:],data[n:]):
            value=evaluate(rule,refs,x)
            if (value<0).any():
                break
            predictions.append(frame.decode(value.reshape(frame.g.shape)).tolist())
        if len(predictions)==len(task['test']):
            yield cost+8*failures+int(canonical),predictions
