"""Fit object extraction programs and categorical cell transformations from demonstrations."""
from itertools import product
import numpy as np
from scene import Scene


def extract(scene, card):
    mode,stat,largest,render=card
    objs=scene.objects(mode)
    if not objs:
        return None
    o=(max if largest else min)(objs,key=lambda o:o.stats[stat])
    if render=='color':
        return np.array([[o.color]])
    return o.box if render=='box' else o.patch


def lookup(xs,ys):
    keys=np.concatenate([x.ravel() for x in xs])
    vals=np.concatenate([y.ravel() for y in ys])
    uniq,first,inv=np.unique(keys,return_index=True,return_inverse=True)
    if not np.array_equal(vals[first][inv],vals):
        return None
    return dict(zip(map(int,uniq),map(int,vals[first])))


def objects(train,ys,queries):
    for mode in product((True,False),(4,8),('mode',0)):
        if not all(s.objects(mode) for s in train):
            continue
        for stat,largest,render in product(train[0].objects(mode)[0].stats,(True,False),('box','patch','color')):
            card=mode,stat,largest,render
            xs=[extract(s,card) for s in train]
            if any(x.shape!=y.shape for x,y in zip(xs,ys)):
                continue
            mapping=lookup(xs,ys)
            if mapping is None:
                continue
            outs=[extract(q,card) for q in queries]
            if any(x is None for x in outs):
                continue
            lut=np.array([mapping.get(i,i) for i in range(10)])
            cost=2+sum(k!=v for k,v in mapping.items())
            yield cost,('object',card),[lut[x] for x in outs]


def pixels(train,ys,queries):
    if any(s.grid.shape!=y.shape for s,y in zip(train,ys)):
        return
    # Order low-complexity readouts first. Preserve unseen combinations by copying input.
    y=np.concatenate([a.ravel() for a in ys])
    color=np.concatenate([s.grid.ravel() for s in train])
    by_key={}
    for key in train[0].features:
        v=np.concatenate([s.features[key].ravel() for s in train])
        for paired in (False,True):
            encoded=v*10+color if paired else v
            code=encoded.tobytes()
            if code in by_key:
                continue
            by_key[code]=True
            mapping=lookup([encoded],[y])
            if mapping is None:
                continue
            if paired:
                mapping={k:v for k,v in mapping.items() if k%10!=v}
            cost=2+len(mapping)+(1 if paired else 0)
            outs=[]
            for q in queries:
                vals=q.features[key]
                if paired:
                    vals=vals*10+q.grid
                out=q.grid.copy()
                for k,v in mapping.items():
                    out[vals==k]=v
                outs.append(out)
            yield cost,('pixel',key,paired),outs
    # If a ray or neighboring object supplies the new color, learn a gating
    # predicate. This also transfers to colors absent from the demonstrations.
    changed=y!=color
    if not changed.any():
        return
    for key in train[0].features:
        if not any(k in key for k in ('near_color','ray','between','neighbor')):
            continue
        src=np.concatenate([s.features[key].ravel() for s in train])
        if not np.all(src[changed]==y[changed]):
            continue
        for gate in (None,*range(10)):
            use=(src>=0)&((color==gate) if gate is not None else True)
            if not np.array_equal(np.where(use,src,color),y):
                continue
            outs=[]
            for q in queries:
                srcq=q.features[key]
                use=(srcq>=0)&((q.grid==gate) if gate is not None else True)
                outs.append(np.where(use,srcq,q.grid))
            yield 1+int(gate is not None),('copy',key,gate),outs


def solve(t,prior,trace):
    if any('output' in q for q in t['test']):
        raise ValueError('Remove query labels before inference')
    train=[Scene(e['input']) for e in t['train']]
    ys=[np.array(e['output']) for e in t['train']]
    queries=[Scene(q['input']) for q in t['test']]
    ranked=sorted([*objects(train,ys,queries),*pixels(train,ys,queries)],key=lambda x:x[0])
    result=[]
    sources=[]
    for i,q in enumerate(queries):
        choices=[]
        def add(grid,origin):
            if len(choices)<2 and all(grid!=v for v,_ in choices):
                choices.append((grid,origin))
        for a in (1,2):
            if any(p['input']==i and p['attempt']==a for p in trace):
                add(prior[i][f'attempt_{a}'],'rc2')
        for _,origin,outs in ranked:
            if len(choices)==2:
                break
            add(outs[i].tolist(),origin)
        add(q.grid.tolist(),'identity')
        if len(choices)==1:
            choices.append(choices[0])
        result.append({f'attempt_{n}':v for n,(v,_) in enumerate(choices,1)})
        sources.append([s for _,s in choices])
    return result,sources
