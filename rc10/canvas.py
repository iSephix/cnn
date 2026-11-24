"""Infer output dimensions, source window and coordinate sampling from demonstrations."""
from itertools import product
import numpy as np
from scipy import ndimage as nd
from memory import Frame


def windows(g):
    yield 'whole', g
    bg = int(np.bincount(g.ravel()).argmax())
    def crop(mask):
        yy,xx = np.where(mask)
        if len(yy):
            return g[yy.min():yy.max()+1,xx.min():xx.max()+1]
    for name,mask in [('ink',g!=bg),*[(f'color-{c}',g==c) for c in range(10)]]:
        box = crop(mask)
        if box is not None:
            yield name,box
    if min(g.shape)>2:
        yield 'interior',g[1:-1,1:-1]
    rows = np.r_[True,(g[1:]!=g[:-1]).any(axis=1)]
    cols = np.r_[True,(g[:,1:]!=g[:,:-1]).any(axis=0)]
    yield 'compressed',g[rows][:,cols]
    for diag in (False,True):
        labels,_ = nd.label(g!=bg,nd.generate_binary_structure(2,2 if diag else 1))
        boxes = nd.find_objects(labels)
        if boxes:
            areas = np.bincount(labels.ravel())[1:]
            for name,idx in [('largest',areas.argmax()),('smallest',areas.argmin())]:
                yield (name,diag),g[boxes[idx]]


def sizes(shapes,axis,targets,n):
    values = np.array([s[axis] for s in shapes])
    options = [('same',values),('square',values**2)]
    for stride in range(2,6):
        options.extend([(('times',stride),values*stride),(('divide',stride),values//stride)])
    options.append((('fixed',targets[0]),np.repeat(targets[0],len(values))))
    seen = set()
    for name,seq in options:
        if not np.array_equal(seq[:n],targets) or seq.min()<1 or seq.max()>30:
            continue
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            yield name,seq


def indices(size, out):
    i = np.arange(out)
    yield 'nearest',(i*size)//out
    yield 'periodic',i%size
    yield 'reflected',np.where((i//size)%2,size-1-i%size,i%size)
    yield 'center',np.clip(i+(size-out)//2,0,size-1)
    yield 'end',np.clip(i+size-out,0,size-1)
    if size>1:
        yield 'bounce',(size-1)-np.abs(i%(2*size-2)-(size-1))


def axes(shapes,lengths,axis):
    variants = [dict(indices(s[axis],v)) for s,v in zip(shapes,lengths)]
    seen = set()
    for mode in variants[0]:
        if any(mode not in v for v in variants):
            continue
        for flip in (False,True):
            ix = [s[axis]-1-v[mode] if flip else v[mode] for s,v in zip(shapes,variants)]
            key = tuple(i.tobytes() for i in ix)
            if key not in seen:
                seen.add(key)
                yield (mode,flip),ix


def infer(task):
    demos = task['train']
    n = len(demos)
    all_examples = demos+task['test']
    for canonical in (False,True):
        frames = [Frame(e['input'],canonical) for e in all_examples]
        targets = [f.target(e['output']) for f,e in zip(frames,demos)]
        if all(f.g.shape==y.shape for f,y in zip(frames,targets)):
            continue
        windows_by_grid = [dict(windows(f.g)) for f in frames]
        seen = set()
        for region in windows_by_grid[0]:
            if any(region not in d for d in windows_by_grid):
                continue
            for transpose in (False,True):
                src = [d[region].T if transpose else d[region] for d in windows_by_grid]
                sig = tuple((s.shape,s.tobytes()) for s in src)
                if sig in seen:
                    continue
                seen.add(sig)
                shapes = [a.shape for a in src]
                dims = [list(sizes(shapes,axis,[y.shape[axis] for y in targets],n)) for axis in (0,1)]
                for (yr,h),(xr,w) in product(*dims):
                    for (ym,rows),(xm,cols) in product(axes(shapes,h,0),axes(shapes,w,1)):
                        lut = np.full(10,-1)
                        for a,y,iy,ix in zip(src,targets,rows,cols):
                            sample = a[np.ix_(iy,ix)]
                            # Reject contradictions inside or across demonstrations.
                            update = lut.copy()
                            update[sample.ravel()] = y.ravel()
                            if np.any((lut>=0)&(lut!=update)) or not np.array_equal(update[sample],y):
                                break
                            lut = update
                        else:
                            remap = int(np.sum((lut>=0)&(lut!=np.arange(10))))
                            table = np.where(lut<0,np.arange(10),lut)
                            outs = [f.decode(table[a[np.ix_(iy,ix)]]).tolist()
                                    for f,a,iy,ix in zip(frames[n:],src[n:],rows[n:],cols[n:])]
                            cost = remap + int(region!='whole') + int(transpose) + int(ym[1])+int(xm[1])
                            yield cost,['canvas',region,transpose,yr,xr,ym,xm,canonical,table.tolist()],outs
