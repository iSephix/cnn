"""Induce exact output dimensions from reusable measurements of the input."""
from itertools import product
import numpy as np
from scipy import ndimage


def measure(g):
    height,width=g.shape
    counts=np.bincount(g.ravel(),minlength=10)
    background=int(counts.argmax())
    result={'height':height,'width':width,'colors':int(np.count_nonzero(counts)),
            'unit':1,'active colors':int(np.count_nonzero(counts)-1)}
    for ink in [-1,*np.flatnonzero(counts)]:
        mask=g!=background if ink==-1 else g==ink
        ys,xs=np.where(mask)
        tag='foreground' if ink==-1 else str(ink)
        labels,n=ndimage.label(mask)
        sizes=np.bincount(labels[mask],minlength=n+1)[1:]
        result.update({tag+' height':int(np.ptp(ys))+1 if ys.size else 0,
                       tag+' width':int(np.ptp(xs))+1 if xs.size else 0,
                       tag+' rows':int(np.unique(ys).size),tag+' cols':int(np.unique(xs).size),
                       tag+' objects':n,tag+' area':int(mask.sum()),
                       tag+' largest':int(sizes.max()) if n else 0})
    result['row bands']=1+int(np.any(g[1:]!=g[:-1],axis=1).sum())
    result['col bands']=1+int(np.any(g[:,1:]!=g[:,:-1],axis=0).sum())
    return result


def linear_values(train, query, labels):
    """Return the simplest exact dimension expression for each query value."""
    result={}
    def save(value, cost):
        if 1<=value<=30:
            result[value]=min(cost,result.get(value,cost))
    if len(set(labels))==1:
        save(labels[0],3)
    common=sorted(set.intersection(*(set(e) for e in train),set(query)))
    for feature in common:
        values=[g[feature] for g in train]
        base=1 if feature in ('height','width') else 2 if feature.startswith('foreground') else 3
        for mul,div in ((1,1),(2,1),(3,1),(4,1),(1,2),(1,3)):
            linear=[v*mul//div for v in values]
            shift=labels[0]-linear[0]
            if abs(shift)>5 or any(v+shift!=t for v,t in zip(linear,labels)):
                continue
            save(query[feature]*mul//div+shift,base+(mul!=1)+(div!=1)+abs(shift))
    return result


def rank_sizes(inputs, targets, query):
    xs=[measure(g) for g in inputs]
    q=measure(query)
    heights=linear_values(xs,q,[g.shape[0] for g in targets])
    widths=linear_values(xs,q,[g.shape[1] for g in targets])
    paired=[(a+b,h,w) for (h,a),(w,b) in product(heights.items(),widths.items())]
    return [(h,w) for _,h,w in sorted(paired)]
