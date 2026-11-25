"""Learn a shared stencil relative to each occurrence of an anchor color."""
import numpy as np


def translated(points, delta, shape):
    y,x=points[0]+delta[0],points[1]+delta[1]
    keep=(y>=0)&(y<shape[0])&(x>=0)&(x<shape[1])
    return y[keep],x[keep]


def rules(task):
    demos=task['train']
    xs=[np.asarray(e['input']) for e in demos]
    ys=[np.asarray(e['output']) for e in demos]
    if any(x.shape!=y.shape for x,y in zip(xs,ys)):
        return
    changes=[set(zip(*np.where(x!=y))) for x,y in zip(xs,ys)]
    if not all(changes):
        return
    inks=set(int(ys[0][p]) for p in changes[0])
    if len(inks)!=1:
        return
    ink=inks.pop()
    if any(any(y[p]!=ink for p in changed) for y,changed in zip(ys,changes)):
        return
    for anchor in sorted(set.intersection(*(set(np.unique(x)) for x in xs))):
        points=[np.where(x==anchor) for x in xs]
        if max(len(y) for y,_ in points)>45:
            continue
        offsets={(r-i,c-j) for r,c in changes[0] for i,j in zip(*points[0]) if max(abs(r-i),abs(c-j))<=12}
        for background_only in (True,False):
            covers={}
            for offset in sorted(offsets,key=lambda d:(abs(d[0])+abs(d[1]),d)):
                covered=set()
                for i,(x,y,pts) in enumerate(zip(xs,ys,points)):
                    r,c=translated(pts,offset,x.shape)
                    if background_only:
                        valid=x[r,c]==np.bincount(x.ravel()).argmax()
                        r,c=r[valid],c[valid]
                    if np.any(y[r,c]!=ink):
                        break
                    covered.update((i,a,b) for a,b in zip(r,c) if x[a,b]!=ink)
                else:
                    if covered:
                        covers[offset]=covered
            needed={(i,r,c) for i,changed in enumerate(changes) for r,c in changed}
            stencil=[]
            while needed and covers:
                offset=max(covers,key=lambda d:len(needed & covers[d]))
                hits=needed & covers.pop(offset)
                if not hits:
                    break
                needed-=hits
                stencil.append(offset)
            if needed:
                continue
            outputs=[]
            for q in task['test']:
                x=np.array(q['input'])
                out=x.copy()
                pts=np.where(x==anchor)
                bg=np.bincount(x.ravel()).argmax()
                for offset in stencil:
                    r,c=translated(pts,offset,x.shape)
                    if background_only:
                        keep=x[r,c]==bg
                        r,c=r[keep],c[keep]
                    out[r,c]=ink
                outputs.append(out)
            yield 3+len(stencil),outputs
