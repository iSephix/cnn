"""Search shared scene selections followed by geometric and learned tile programs."""
from collections import Counter
import numpy as np
from scipy import ndimage


def views(g):
    bg = np.bincount(g.ravel()).argmax()
    yield ('scene',),g
    for c in np.unique(g):
        yy,xx = np.where(g==c)
        crop = g[yy.min():yy.max()+1,xx.min():xx.max()+1]
        yield ('color box',int(c)),crop
        if min(crop.shape)>2:
            yield ('inside',int(c)),crop[1:-1,1:-1]
    compressed = g[np.r_[True,np.any(g[1:]!=g[:-1],axis=1)]]
    yield ('compress',),compressed[:,np.r_[True,np.any(compressed[:,1:]!=compressed[:,:-1],axis=0)]]
    for axis in (0,1):
        a = g if axis==0 else g.T
        for count in (2,3,4):
            for gap in (0,1):
                length,rem = divmod(len(a)-(count-1)*gap,count)
                if rem or length<1:
                    continue
                parts = [a[i*(length+gap):i*(length+gap)+length] for i in range(count)]
                for i,p in enumerate(parts):
                    yield ('panel',axis,count,gap,i),p if axis==0 else p.T
                for metric,score in [('colors',lambda p:np.unique(p).size),
                                     ('ink',lambda p:np.count_nonzero(p!=bg))]:
                    for order,select in [('least',min),('most',max)]:
                        p=select(parts,key=score)
                        yield ('rank panel',axis,count,gap,metric,order),p if axis==0 else p.T
    for connectivity in (1,2):
        labels,_ = ndimage.label(g!=bg,ndimage.generate_binary_structure(2,connectivity))
        objs=[]
        for i,box in enumerate(ndimage.find_objects(labels),1):
            mask=labels[box]==i
            patch=np.where(mask,g[box],bg)
            shapes=[np.rot90(face,k) for face in (mask,mask[:,::-1]) for k in range(4)]
            key=min((m.shape,m.tobytes()) for m in shapes)
            objs.append((patch,int(mask.sum()),key))
        if not objs:
            continue
        freq=Counter(o[2] for o in objs)
        for metric,score in [('area',lambda o:o[1]),('palette',lambda o:np.unique(o[0]).size),
                             ('copies',lambda o:freq[o[2]])]:
            for order,select in [('least',min),('most',max)]:
                yield ('object',connectivity,metric,order),select(objs,key=score)[0]


def orient(g,k):
    return np.rot90(g if k<4 else g[:,::-1],k%4)


def tile_rule(inputs,targets):
    h,w=inputs[0].shape
    oh,ow=targets[0].shape
    a,b=oh//h,ow//w
    if not a or not b or any(y.shape!=(x.shape[0]*a,x.shape[1]*b) for x,y in zip(inputs,targets)):
        return
    table={}
    for x,y in zip(inputs,targets):
        blocks=y.reshape(x.shape[0],a,x.shape[1],b).transpose(0,2,1,3)
        for color in np.unique(x):
            cells=blocks[x==color]
            tile=table.setdefault(int(color),cells[0])
            if np.any(cells!=tile):
                return
    return table


def generate(task):
    if not task['train'] or not task['test'] or any('output' in q for q in task['test']):
        raise ValueError('Expected demonstrations and unlabeled queries')
    scenes=[dict(views(np.array(e['input']))) for e in task['train']+task['test']]
    ys=[np.array(e['output']) for e in task['train']]
    n=len(ys)
    results=[{} for _ in task['test']]
    seen=set()
    for key in scenes[0]:
        if any(key not in s for s in scenes):
            continue
        grids=[s[key] for s in scenes]
        signature=tuple((g.shape,g.tobytes()) for g in grids)
        if signature in seen:
            continue
        seen.add(signature)
        for k in range(8):
            xs=[orient(g,k) for g in grids]
            table=tile_rule(xs[:n],ys)
            if table is None:
                continue
            a,b=next(iter(table.values())).shape
            if any(x.shape[0]*a>30 or x.shape[1]*b>30 or any(c not in table for c in np.unique(x)) for x in xs[n:]):
                continue
            cost=len(key)+int(k!=0)+sum(np.any(v!=c) for c,v in table.items())
            for row,x in zip(results,xs[n:]):
                lut=np.zeros((10,a,b),dtype=int)
                for c,tile in table.items():
                    lut[c]=tile
                out=lut[x].transpose(0,2,1,3).reshape(x.shape[0]*a,x.shape[1]*b)
                sig=out.shape,out.tobytes()
                if sig not in row or cost<row[sig][0]:
                    row[sig]=cost,out.tolist()
    return [dict(status='ok',candidates=[[n-min(.99,.01*c),n,g] for c,g in sorted(row.values(),key=lambda v:v[0])]) for row in results]
