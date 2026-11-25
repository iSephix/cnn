"""Infer programs over marker edges, empty rectangles, and translated bands."""
from itertools import combinations, product

import numpy as np
from scipy import ndimage

from repair import translate


def packed(masks):
    out=0
    for mask in reversed(masks):
        data=np.packbits(mask.ravel(),bitorder='little').tobytes()
        out=(out<<mask.size)|int.from_bytes(data,'little')
    return out


def l_path(grid,start,end,horizontal_first):
    r,c=start
    y,x=end
    mask=np.zeros(grid.shape,bool)
    if horizontal_first:
        mask[r,min(c,x):max(c,x)+1]=True
        mask[min(r,y):max(r,y)+1,x]=True
    else:
        mask[min(r,y):max(r,y)+1,c]=True
        mask[y,min(c,x):max(c,x)+1]=True
    return mask


def marker_paths(grids,ys,n):
    delta=[g!=y for g,y in zip(grids,ys)]
    colors=set.union(*(set(y[m]) for y,m in zip(ys,delta)))
    if len(colors)!=1 or not all(m.any() for m in delta):
        return
    ink=int(colors.pop())
    backgrounds=[int(np.bincount(g.ravel()).argmax()) for g in grids]
    marks=[]
    for g,bg in zip(grids,backgrounds):
        if np.count_nonzero(g!=bg)>12:
            return
        row={}
        for c in np.unique(g):
            if c==bg:
                continue
            points=np.argwhere(g==c)
            if len(points)==1:
                row[int(c)]=points[0]
        marks.append(row)
    common=sorted(set.intersection(*(set(row) for row in marks)))
    if len(common)<2:
        return
    for anchor in common:
        partial=[]
        required=packed(delta)
        for start,horizontal in product(common,(True,False)):
            if start==anchor:
                continue
            paths=[l_path(g,a[start],a[anchor],horizontal)&(g==bg) for g,a,bg in
                   zip(grids,marks,backgrounds)]
            if any(np.any(y[path]!=ink) for y,path in zip(ys,paths)):
                continue
            coverage=packed(paths[:n])&required
            if coverage:
                partial.append((coverage,paths[n:]))
        for depth in range(1,min(4,len(partial))+1):
            for chosen in combinations(partial,depth):
                union=0
                for coverage,_ in chosen:
                    union|=coverage
                if union==required:
                    masks=[np.logical_or.reduce(items) for items in zip(*(paths for _,paths in chosen))]
                    yield 3+depth,[np.where(m,ink,g) for g,m in zip(grids[n:],masks)]


def free_rectangles(mask,minside,cap,style):
    h,w=mask.shape
    if min(h,w)<minside:
        return np.zeros_like(mask)
    area=0
    boxes=[]
    sums=np.pad(mask.cumsum(axis=0).cumsum(axis=1),((1,0),(1,0)))
    for a,b in product(range(minside,min(h,cap)+1),range(minside,min(w,cap)+1)):
        if a*b<area:
            continue
        mass=sums[a:,b:]-sums[:-a,b:]-sums[a:,:-b]+sums[:-a,:-b]
        y,x=np.where(mass==a*b)
        if not len(x):
            continue
        if a*b>area:
            area=a*b
            boxes=[]
        boxes.extend((int(row),int(col),a,b) for row,col in zip(y,x))
    output=np.zeros_like(mask)
    for y,x,a,b in sorted(boxes,reverse=style=='reverse'):
        window=output[y:y+a,x:x+b]
        if style=='union' or not window.any():
            window[:]=True
    return output


def rectangles(grids,ys,n):
    diff=[a!=b for a,b in zip(grids,ys)]
    colors=set.union(*(set(y[m]) for y,m in zip(ys,diff)))
    bgs=[np.bincount(g.ravel()).argmax() for g in grids]
    if len(colors)!=1 or any(np.any(m&(g!=bg)) for g,m,bg in zip(grids,diff,bgs)):
        return
    ink=int(colors.pop())
    for minside,cap,style in product((2,3),(5,10,30),('union','greedy','reverse')):
        if any(not np.array_equal(free_rectangles(g==bg,minside,cap,style),m)
               for g,bg,m in zip(grids,bgs,diff)):
            continue
        outputs=[np.where(free_rectangles(g==bg,minside,cap,style),ink,g)
                 for g,bg in zip(grids[n:],bgs[n:])]
        yield 3+int(style!='union'),outputs


def band_groups(grid,kind,bg):
    present=(grid!=bg).any(axis=1)
    indices=np.flatnonzero(present)
    if not len(indices):
        return []
    if kind=='row':
        return [[i] for i in indices]
    if kind=='segment':
        return np.split(indices,np.flatnonzero(np.diff(indices)!=1)+1)
    codes=[tuple(int(v) for v in np.unique(grid[i]) if v!=bg) for i in indices]
    cut=[j for j in range(1,len(indices)) if indices[j]!=indices[j-1]+1 or codes[j]!=codes[j-1]]
    return np.split(indices,cut)


def shift_rows(rows,offset,bg,wrap):
    values=np.roll(rows,offset,axis=1)
    if not wrap and offset:
        if offset>0:
            values[:,:offset]=bg
        else:
            values[:,offset:]=bg
    return values


def band_shift(oriented,groups,bg,offsets,wrap):
    out=oriented.copy()
    for index,indices in enumerate(groups):
        offset=offsets[index%len(offsets)]
        out[indices]=shift_rows(oriented[indices],offset,bg,wrap)
    return out


def bands(grids,ys,n):
    if any(sorted(g.ravel())!=sorted(y.ravel()) for g,y in zip(grids,ys)):
        return
    for axis,kind,reverse,wrap in product((0,1),('row','segment','palette'),(False,True),(False,True)):
        oriented=[g if axis==0 else g.T for g in grids]
        expected=[y if axis==0 else y.T for y in ys]
        bgs=[int(np.bincount(g.ravel()).argmax()) for g in oriented]
        groups=[band_groups(g,kind,bg) for g,bg in zip(oriented,bgs)]
        if reverse:
            groups=[g[::-1] for g in groups]
        fit=[]
        supported=True
        for grid,target,bg,bands in zip(oriented,expected,bgs,groups):
            active=(grid!=bg).any(axis=1)
            if not np.array_equal(grid[~active],target[~active]):
                supported=False
                break
            fit.append([{offset for offset in range(-3,4)
                         if np.array_equal(shift_rows(grid[indices],offset,bg,wrap),target[indices])}
                        for indices in bands])
        if not supported:
            continue
        for period in (2,3,4,5):
            offsets=[]
            for phase in range(period):
                evidence=[x for f in fit for x in f[phase::period]]
                shared=set.intersection(*evidence) if evidence else set()
                if not shared:
                    break
                offsets.append(min(shared,key=lambda v:(abs(v),v)))
            if len(offsets)!=period or not any(offsets):
                continue
            values=[band_shift(g,s,bg,offsets,wrap) for g,s,bg in zip(oriented[n:],groups[n:],bgs[n:])]
            yield 3+int(wrap)+period+sum(abs(v) for v in offsets),[g if axis==0 else g.T for g in values]


def zero_pad(g,dy,dx):
    return translate(g,dy,dx).astype(bool)


def topology(grid,ink):
    m=grid==ink
    neighbors=[zero_pad(m,dy,dx) for dy,dx in ((-1,0),(0,1),(1,0),(0,-1))]
    diagonals=[zero_pad(m,dy,dx) for dy,dx in ((-1,1),(1,1),(1,-1),(-1,-1))]
    degree=np.sum(neighbors,axis=0)
    diagonal=np.sum(diagonals,axis=0)
    turn=(degree==2)&~((neighbors[0]&neighbors[2])|(neighbors[1]&neighbors[3]))
    hole=ndimage.binary_fill_holes(m)&~m
    nearby_turns=np.sum([zero_pad(turn,dy,dx) for dy,dx in ((-1,0),(0,1),(1,0),(0,-1))],axis=0)
    orbits=[]
    for k in range(4):
        for flip in (False,True):
            h=np.zeros(m.shape,dtype=np.int16)
            for i in range(4):
                j=(k+(3-i if flip else i))%4
                h|=neighbors[j].astype(np.int16)<<i
                j=(k+(2-i if flip else i))%4
                h|=diagonals[j].astype(np.int16)<<(i+4)
            orbits.append(h)
    return {'degree':degree,'diagonal':diagonal,'turn':turn,'turn neighbors':nearby_turns,
            'orbit':np.minimum.reduce(orbits),'interior border':ndimage.binary_dilation(hole)&m,
            'gap border':ndimage.binary_dilation(~(m|hole))&m}


def topological_colors(grids,ys,n):
    before=[g!=y for g,y in zip(grids,ys)]
    edited=set.union(*(set(g[m]) for g,m in zip(grids,before)))
    if len(edited)!=1:
        return
    ink=int(edited.pop())
    # This rule reuses the palette only on pixels belonging to the foreground graph.
    if any(ink==np.bincount(g.ravel()).argmax() for g in grids):
        return
    frames=[topology(g,ink) for g in grids]
    fields=list(frames[0])
    masks=[g==ink for g in grids]
    truth=np.concatenate([y[m] for y,m in zip(ys,masks)])
    for keys in [*((k,) for k in fields),*combinations(fields,2)]:
        values=[]
        for f in frames:
            u=f[keys[0]].astype(np.int32)
            if len(keys)==2:
                u=u+256*f[keys[1]].astype(np.int32)
            values.append(u)
        train=np.concatenate([v[m] for v,m in zip(values[:n],masks[:n])])
        table=np.full(65536,ink,np.int8)
        table[train]=truth
        if not np.array_equal(table[train],truth) or np.count_nonzero(table!=ink)>16:
            continue
        out=[np.where(m,table[v],g) for m,v,g in zip(masks[n:],values[n:],grids[n:])]
        yield 3+len(keys)+int('orbit' in keys),out


def structure(task):
    n=len(task['train'])
    grids=[np.asarray(e['input']) for e in task['train']+task['test']]
    ys=[np.asarray(e['output']) for e in task['train']]
    if any(g.shape!=y.shape for g,y in zip(grids,ys)) or all(np.array_equal(g,y) for g,y in zip(grids,ys)):
        return
    for solver in (marker_paths,rectangles,bands,topological_colors):
        for cost,outputs in solver(grids,ys,n):
            yield solver.__name__,cost,[g.tolist() for g in outputs]
