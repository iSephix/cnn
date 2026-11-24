"""Search over short ordered grid programs. All supervision comes from demonstrations."""
from collections import deque
from functools import lru_cache
import numpy as np


def pack(g):
    return (g.shape,g.astype(np.uint8,copy=False).tobytes())


def unpack(state):
    shape,buf=state
    return np.frombuffer(buf,np.uint8).reshape(shape)


OPS=['crop','compress','inner','unique-rows','unique-cols','mirror-x','mirror-y','transpose',
     'complete-x','complete-y','complete-diagonal','mirror-stack','mirror-side',
     'repeat2','enlarge2','period','fill-period','box','outline']


@lru_cache(maxsize=24000)
def step(state,op):
    a=unpack(state)
    h,w=a.shape
    if op=='crop':
        ys,xs=np.where(a!=0)
        if not len(ys): return None
        b=a[ys.min():ys.max()+1,xs.min():xs.max()+1]
    elif op=='compress':
        b=a[np.r_[True,np.any(a[1:]!=a[:-1],axis=1)]]
        b=b[:,np.r_[True,np.any(b[:,1:]!=b[:,:-1],axis=0)]]
    elif op=='inner':
        if min(h,w)<3:return None
        b=a[1:-1,1:-1]
    elif op.startswith('unique'):
        axis=int(op.endswith('cols'))
        indices=np.unique(a,axis=axis,return_index=True)[1]
        b=np.take(a,np.sort(indices),axis=axis)
    elif op=='transpose': b=a.T
    elif op=='mirror-x': b=a[:,::-1]
    elif op=='mirror-y': b=a[::-1]
    elif op.startswith('complete'):
        other={'complete-x':a[:,::-1],'complete-y':a[::-1],'complete-diagonal':a.T}[op]
        if other.shape!=a.shape:return None
        b=np.where(a,a,other)
    elif op=='mirror-stack': b=np.vstack((a,a[::-1]))
    elif op=='mirror-side': b=np.hstack((a,a[:,::-1]))
    elif op=='repeat2': b=np.tile(a,(2,2))
    elif op=='enlarge2': b=a.repeat(2,0).repeat(2,1)
    elif op in ('period','fill-period'):
        # Smallest repeating rectangle independently along both axes.
        def compatible(x,y):
            if op=='period':return np.array_equal(x,y)
            return not np.any((x!=y)&(x!=0)&(y!=0))
        def stride(x):
            for d in range(1,len(x)):
                if all(compatible(x[k:],x[:-k]) for k in range(d,len(x),d)):return d
            return len(x)
        dy,dx=stride(a),stride(a.T)
        if op=='period': b=a[:dy,:dx]
        else:
            b=a.copy()
            for y,x in zip(*np.where(a==0)):
                values=np.unique(a[y%dy::dy,x%dx::dx])
                ink=values[values!=0]
                if len(ink)==1:b[y,x]=ink[0]
    else:
        b=a.copy()
        for c in np.unique(a):
            if c==0:continue
            y,x=np.where(a==c)
            r,s=y.min(),y.max()
            t,u=x.min(),x.max()
            if op=='box':b[r:s+1,t:u+1]=c
            else:
                b[[r,s],t:u+1]=c
                b[r:s+1,[t,u]]=c
        b=np.where(a,a,b)
    return pack(b) if max(b.shape)<=30 else None


def infer(xs,ys,qs,ordered=True,budget=700):
    initial=tuple(map(pack,xs+qs))
    queue=deque([(initial,())])
    seen=set()
    targets=[pack(y) for y in ys]
    trials=0
    while queue:
        states,path=queue.popleft()
        for i,op in enumerate(OPS):
            if not ordered and path and i<path[-1]:continue
            trials+=1
            if trials>budget:return
            demo=tuple(step(s,op) for s in states[:len(xs)])
            if None in demo:continue
            tests=tuple(step(s,op) for s in states[len(xs):])
            if None in tests:continue
            group=demo+tests
            seq=path+(i,)
            if demo==tuple(targets):
                yield len(seq),[OPS[j] for j in seq],[unpack(s).tolist() for s in tests]
            key=(group,None if ordered else i)
            if len(seq)<3 and group!=states and key not in seen:
                seen.add(key)
                queue.append((group,seq))
