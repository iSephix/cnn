"""Use a learned edit mask as the seed for a spatial write."""
from collections import deque
import numpy as np
from memory import Frame,codes


def project(mask,op):
    if op=='rows':return np.broadcast_to(mask.any(1)[:,None],mask.shape)
    if op=='cols':return np.broadcast_to(mask.any(0)[None,:],mask.shape)
    if op=='cross':return mask.any(1)[:,None]|mask.any(0)[None,:]
    if op=='box':return (mask.any(1).cumsum()>0)[:,None] & (mask.any(1)[::-1].cumsum()[::-1]>0)[:,None] & (mask.any(0).cumsum()>0) & (mask.any(0)[::-1].cumsum()[::-1]>0)
    if op in ('horizontal','vertical'):
        a=mask.T if op=='vertical' else mask
        filled=(np.maximum.accumulate(a,axis=1)&np.maximum.accumulate(a[:,::-1],axis=1)[:,::-1])
        return filled.T if op=='vertical' else filled
    if op=='surround':
        p=np.pad(mask,1)
        return (p[:-2,1:-1]|p[2:,1:-1]|p[1:-1,:-2]|p[1:-1,2:])&~mask
    # All background cells disconnected from the outside.
    h,w=mask.shape
    visited=mask.copy()
    todo=deque()
    for r,c in zip(*np.where(~mask)):
        if r in (0,h-1) or c in (0,w-1):
            todo.append((r,c))
            visited[r,c]=True
    while todo:
        r,c=todo.popleft()
        for y,x in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if 0<=y<h and 0<=x<w and not visited[y,x]:
                visited[y,x]=True
                todo.append((y,x))
    return ~visited


def expand(task,bank):
    demos=task['train']
    if any(np.shape(e['input'])!=np.shape(e['output']) for e in demos):return
    for canonical in (False,True):
        frames=[Frame(e['input'],canonical) for e in demos+task['test']]
        n=len(demos)
        targets=[f.target(e['output']) for f,e in zip(frames,demos)]
        ink=np.unique(np.concatenate([y[y!=f.g] for f,y in zip(frames,targets)]))
        if len(ink)!=1:continue
        color=int(ink[0])
        selectors=[('color',c) for c in np.unique(frames[0].g)]
        selectors += [('card',i) for i,b in enumerate(bank) if b['canonical']==canonical]
        seen=set()
        for kind,i in selectors:
            def mask(f):
                if kind=='color':return f.g==i
                b=bank[i]
                z=codes(b,f)
                return z!=f.g if b['kind']=='paint' else z!=0
            masks=[mask(f) for f in frames]
            key=tuple(m.tobytes() for m in masks)
            if key in seen:continue
            seen.add(key)
            for op in ('rows','cols','cross','box','horizontal','vertical','surround','enclosed'):
                if all(np.array_equal(np.where(project(m,op),color,f.g),y)
                       for m,f,y in zip(masks,frames,targets)):
                    outs=[f.decode(np.where(project(m,op),color,f.g)).tolist() for f,m in zip(frames[n:],masks[n:])]
                    yield 4,['spatial',kind,int(i),op,color,canonical],outs
