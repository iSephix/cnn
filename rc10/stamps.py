"""Induce displacement and ray programs that jointly explain all observed edits."""
from itertools import product
import numpy as np
from scipy import ndimage as nd
from memory import Frame

# A padded bit board allows the same displacement to operate on every example.
STRIDE = 48


def sources(g):
    bg = int(np.bincount(g.ravel()).argmax())
    occupied = g!=bg
    yield 'ink', occupied
    for c in range(10):
        mask = g==c
        yield ('color',c),mask
    for neighbors in (1,2):
        kernel = nd.generate_binary_structure(2,neighbors)
        labels,n = nd.label(occupied,kernel)
        area = np.bincount(labels.ravel(),minlength=n+1)
        if n:
            for kind,size in [('smallest',area[1:].min()),('largest',area[1:].max()),('singleton',1)]:
                yield (neighbors,kind),occupied & (area[labels]==size)
        boxes = nd.find_objects(labels)
        for name,ry,rx in [('nw',0,0),('ne',0,-1),('sw',-1,0),('se',-1,-1),('center',1,1)]:
            mask = np.zeros_like(occupied)
            for row,col in boxes:
                def point(s,v):
                    if v==1:
                        return (s.start+s.stop-1)//2
                    return s.start if v==0 else s.stop-1
                mask[point(row,ry),point(col,rx)] = True
            yield (neighbors,name),mask


def pack(mask, start):
    bits = np.zeros((len(mask),STRIDE),dtype=np.uint8)
    bits[:,:mask.shape[1]] = mask
    return int.from_bytes(np.packbits(bits,axis=1,bitorder='little').tobytes(),'little') << start


def unpack(bits,start,shape):
    h,w = shape
    part = (bits>>start) & ((1<<(STRIDE*h))-1)
    data = np.frombuffer(part.to_bytes(STRIDE*h//8,'little'),dtype=np.uint8).reshape(h,-1)
    return np.unpackbits(data,axis=1,bitorder='little')[:,:w].astype(bool)


def shifted(bits,dy,dx):
    shift = STRIDE*dy+dx
    return bits << shift if shift>=0 else bits >> -shift


def actions(seed,board):
    for dy,dx in sorted(product(range(-5,6),repeat=2), key=lambda p:abs(p[0])+abs(p[1])):
        yield ('shift',dy,dx),shifted(seed,dy,dx)&board,abs(dx)+abs(dy)
    for dy,dx in product((-1,0,1),repeat=2):
        if not (dy or dx):
            continue
        for step in (1,2,3):
            current = seed
            ray = 0
            while current:
                current = shifted(current,step*dy,step*dx)&board
                ray |= current
            yield ('ray',dy,dx,step),ray,step+1


def fit_cover(atoms,need):
    # Alternatives must agree on all demonstrated pixels, so selected atoms commute on them.
    uncovered = need
    chosen = []
    while uncovered and len(chosen)<16:
        best = max(atoms,key=lambda a: ((a[0]&uncovered).bit_count(),-a[1]),default=None)
        if best is None or not (best[0]&uncovered):
            return
        uncovered &= ~best[0]
        chosen.append(best)
    if uncovered:
        return
    return chosen


def infer(task):
    demos = task['train']
    if any(np.shape(e['input'])!=np.shape(e['output']) for e in demos):
        return
    n = len(demos)
    for canonical in (False,True):
        fs = [Frame(e['input'],canonical) for e in demos+task['test']]
        ys = [f.target(e['output']) for f,e in zip(fs,demos)]
        starts = []
        offset = 0
        board = 0
        seeds = {}
        color = [0]*10
        for f in fs:
            starts.append(offset)
            board |= pack(np.ones_like(f.g),offset)
            for name,mask in sources(f.g):
                seeds[name] = seeds.get(name,0)|pack(mask,offset)
            for c in np.unique(f.g):
                color[c] |= pack(f.g==c,offset)
            offset += (len(f.g)+8)*STRIDE
        train = (1<<starts[n])-1
        target = [0]*10
        for yy,pos in zip(ys,starts):
            for c in np.unique(yy):
                target[c] |= pack(yy==c,pos)
        background = 0
        for f,pos in zip(fs,starts):
            background |= pack(f.g==np.bincount(f.g.ravel()).argmax(),pos)
        atoms = {}
        seen = set()
        for name,seed in seeds.items():
            if not seed&train or seed in seen:
                continue
            seen.add(seed)
            for op,mask,dist in actions(seed,board):
                if not mask&train:
                    continue
                for gate,part in [('all',mask),('background',mask&background),('ink',mask&~background)]:
                    observed = part&train
                    if not observed:
                        continue
                    for c,t in enumerate(target):
                        if observed & ~t:
                            continue
                        weight = dist+int(gate!='all')+int(name!='ink')
                        key = (c,observed)
                        if key not in atoms or weight<atoms[key][1]:
                            atoms[key] = observed,weight,part,c,(name,op,gate,c)
                        break
        for blank in (False,True):
            bases = [np.full_like(f.g,np.bincount(f.g.ravel()).argmax()) if blank else f.g for f in fs]
            need = 0
            for b,y,pos in zip(bases,ys,starts):
                need |= pack(b!=y,pos)
            if not need:
                continue
            chosen = fit_cover(list(atoms.values()),need)
            if not chosen:
                continue
            outs = []
            for f,b,pos in zip(fs[n:],bases[n:],starts[n:]):
                out = b.copy()
                for _,_,mask,c,_ in chosen:
                    out[unpack(mask,pos,b.shape)] = c
                outs.append(f.decode(out).tolist())
            cost = (len(chosen),sum(a[1] for a in chosen),blank,canonical)
            yield cost,['stamps',canonical,blank,[a[-1] for a in chosen]],outs
