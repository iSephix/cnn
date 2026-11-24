"""Synthesize each output color plane as a Boolean expression over scene measurements."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json,sys
import numpy as np
from scipy import ndimage as nd


def bits(a):
    return int.from_bytes(np.packbits(a.ravel(),bitorder='little').tobytes(),'little')


def masks(g,c):
    m=g==c
    y,x=np.indices(g.shape)
    r,s=np.where(m)
    if not len(r):return {}
    box=(y>=min(r))&(y<=max(r))&(x>=min(s))&(x<=max(s))
    border=box&((y==min(r))|(y==max(r))|(x==min(s))|(x==max(s)))
    out=dict(color=m,box=box,border=border,hole=nd.binary_fill_holes(m)&~m,rows=np.isin(y,r),cols=np.isin(x,s),
             diagonal=np.isin(y+x,r+s),antidiagonal=np.isin(y-x,r-s))
    for axis in (0,1):
        forward=np.logical_or.accumulate(m,axis)
        backward=np.flip(np.logical_or.accumulate(np.flip(m,axis),axis),axis)
        out['span',axis]=forward&backward
        out['forward',axis]=forward
        out['backward',axis]=backward
    for i in (1,2,3):
        for metric in ('chessboard','taxicab'):
            out[metric,i]=nd.distance_transform_cdt(~m,metric)==i
    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
        a=np.roll(m,(dy,dx),axis=(0,1))
        if dy:a[0 if dy>0 else -1,:]=False
        if dx:a[:,0 if dx>0 else -1]=False
        out[dy,dx]=a
    return out


def frame(g,canonical):
    g=np.array(g)
    freq=np.bincount(g.ravel(),minlength=10)
    colors=np.argsort(-freq,kind='stable') if canonical else np.arange(10)
    code=np.argsort(colors)[g]
    planes={}
    for c in np.unique(code):
        for op,mask in masks(code,c).items():
            planes[int(c),op]=bits(mask)
    return colors,planes


def expressions(atoms,target,full):
    exact={m:(cost,key) for m,cost,key in reversed(atoms)}
    rules=[]
    for a,ca,ka in atoms:
        if a==target:rules.append((ca,'one',ka,ka))
        b=a^target
        if b in exact:
            cb,kb=exact[b];rules.append((ca+cb+2,'xor',ka,kb))
    subsets=[v for v in atoms if not v[0]&~target]
    supersets=[v for v in atoms if not target&~v[0]]
    for kind,group in [('or',subsets),('and',supersets)]:
        for i,(a,ca,ka) in enumerate(group):
            for b,cb,kb in group[:i]:
                if (a|b if kind=='or' else a&b)==target:
                    rules.append((ca+cb+1,kind,ka,kb))
    for a,ca,ka in supersets:
        for b,cb,kb in atoms:
            if a&~b==target:rules.append((ca+cb+1,'diff',ka,kb))
    return sorted(rules,key=lambda r:r[0])[:3]


def infer(t):
    demos=t['train']
    if any(np.shape(e['input'])!=np.shape(e['output']) for e in demos):return
    n=len(demos)
    grids=[np.array(e['input']) for e in demos+t['test']]
    sizes=[g.size for g in grids]
    for canonical in (False,True):
        frames=[frame(g,canonical) for g in grids]
        keys=set.union(*(set(d) for _,d in frames[:n]))
        total=sum(sizes[:n]);full=(1<<total)-1
        def joined(seq):
            acc=0
            for v,s in zip(seq[::-1],sizes[:len(seq)][::-1]):acc=(acc<<s)|v
            return acc
        atoms={}
        for key in sorted(keys,key=repr):
            mask=joined([d.get(key,0) for _,d in frames[:n]])
            cost=1 if key[1]=='color' else 2
            for inv,value in [(False,mask),(True,full^mask)]:
                if value in (0,full):continue
                desc=cost+int(inv), (key,inv)
                if value not in atoms or desc[0]<atoms[value][0]:atoms[value]=desc
        bank=sorted([(v,c,k) for v,(c,k) in atoms.items()],key=lambda a:a[1])
        ys=[np.argsort(colors)[np.array(e['output'])] for (colors,_),e in zip(frames,demos)]
        counts=np.bincount(np.concatenate([y.ravel() for y in ys]),minlength=10)
        bg=int(counts.argmax())
        specs={}
        for c in np.flatnonzero(counts):
            if c==bg:continue
            rules=expressions(bank,joined([bits(y==c) for y in ys]),full)
            if not rules:break
            specs[c]=rules
        else:
            for rank in range(3):
                grids_out=[]
                complexity=0
                for g,(colors,planes) in zip(grids[n:],frames[n:]):
                    out=np.full(g.size,bg)
                    for c,opts in specs.items():
                        cost,op,ka,kb=opts[min(rank,len(opts)-1)]
                        complexity+=cost
                        def value(k):
                            key,inv=k
                            v=planes.get(key,0)
                            return ((1<<g.size)-1)^v if inv else v
                        a,b=value(ka),value(kb)
                        v={'one':a,'xor':a^b,'and':a&b,'or':a|b,'diff':a&~b}[op]
                        mask=np.unpackbits(np.frombuffer(v.to_bytes((g.size+7)//8,'little'),dtype=np.uint8),bitorder='little')[:g.size]
                        out[mask.astype(bool)]=c
                    grids_out.append(colors[out].reshape(g.shape).tolist())
                yield complexity+rank,grids_out


def one(args):
    p,out=args;t=json.loads(p.read_text())
    if any('output' in q for q in t['test']):raise ValueError('Test labels present')
    res=[dict(status='ok',candidates=[]) for _ in t['test']]
    n=len(t['train'])
    seen=[set() for _ in res]
    for cost,grids in sorted(infer(t),key=lambda a:a[0]):
        for row,chosen,g in zip(res,seen,grids):
            key=json.dumps(g)
            if key not in chosen:
                row['candidates'].append([n-min(.99,.005*cost),n,g])
                chosen.add(key)
    (out/p.name).write_text(json.dumps(res,separators=(',',':')))


if __name__=='__main__':
    d,out=map(Path,sys.argv[1:3]);out.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(3) as ex:list(ex.map(one,[(p,out) for p in sorted(d.glob('*.json'))]))
