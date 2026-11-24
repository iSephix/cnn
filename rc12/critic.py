"""Compare a proposed transformation's geometry and color statistics with the demonstrations."""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=12000)
def features(grid):
    a=np.asarray(grid)
    h,w=a.shape
    hist=np.bincount(a.ravel(),minlength=10)/a.size
    sorted_hist=np.sort(hist)[::-1]
    rows=np.mean(a[1:]==a[:-1]) if h>1 else 1.
    cols=np.mean(a[:,1:]==a[:,:-1]) if w>1 else 1.
    return np.r_[np.log(h),np.log(w),np.count_nonzero(hist)/10,
                 sorted_hist[:3],np.mean(a==a[::-1]),np.mean(a==a[:,::-1]),rows,cols,hist]


def describe(x,y):
    fx=features(tuple(map(tuple,x)))
    fy=features(tuple(map(tuple,y)))
    same=len(x)==len(y) and len(x[0])==len(y[0])
    edit=np.mean(np.asarray(x)!=np.asarray(y)) if same else 1.
    return np.r_[fy,fy-fx,edit]


def evidence(query,pred,examples):
    proposed=describe(query,pred)
    diff=np.abs(examples-proposed)
    return {f'consistency/{i}':float(value) for i,value in enumerate(diff.min(axis=0))}
