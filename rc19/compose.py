"""Synthesize edits by composing masks from the existing concept engines."""
from itertools import product
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ('rc18', 'rc13')]
import layers
import relations
import render


def measures(grid, mode):
    bg = int(np.bincount(grid.ravel()).argmax())
    if mode == 'pixel':
        _, fields = layers.frame(grid, False)
        return {key: (1 if key[1]=='color' else 2, v) for key,v in fields.items()}
    split = mode != 'span'
    _, fields = render.measures(grid, split)
    atoms = {key: (2 + (len(key[0])>1), layers.bits(v)) for key,v in fields.items()}
    if split:
        sc = relations.scene(grid, True)
        for kind, directions in (('centers',('axial','diagonal')),('gap',(0,1))):
            for link, direction in product(('same','different','all'),directions):
                result = relations.link_mask(grid, sc, kind, link, direction)
                if result is not None:
                    atoms['link',kind,link,direction] = 3,layers.bits(result)
    atoms['bg'] = 1,layers.bits(grid==bg)
    return atoms


def pack(values, widths):
    result = 0
    for mask,width in reversed(list(zip(values,widths))):
        result = (result<<width)|mask
    return result


def mask_programs(bank, changed, allowed, full):
    """Cover every changed cell while guaranteeing every write is safe on demos."""
    proposals = {}
    def add(cost, mask, query):
        if changed&~mask == 0 and mask&~allowed == 0:
            proposals[query] = min(cost,proposals.get(query,cost))
    usable = []
    for (mask,query),cost in bank.items():
        if mask&changed:
            add(cost,mask,query)
            usable.append((cost,mask,query))
    usable.sort(key=lambda a:(a[0],-((a[1]&changed).bit_count())))
    # Apply the same operations across all demonstrations and queries at once.
    usable=usable[:160]
    subsets=[(c,m,q) for c,m,q in usable if m&~allowed==0]
    for i,(c,a,x) in enumerate(usable):
        for d,b,y in usable[:i]:
            add(c+d+1,a&b,x&y)
            add(c+d+2,a&~b,x&~y)
            add(c+d+2,b&~a,y&~x)
            add(c+d+2,a^b,x^y)
    for i,(c,a,x) in enumerate(subsets):
        for j,(d,b,y) in enumerate(subsets[:i]):
            combined=a|b
            add(c+d+1,combined,x|y)
            missing=changed&~combined
            if missing and i<36:
                for e,m,z in subsets[:j]:
                    if not missing&~m:
                        add(c+d+e+2,combined|m,x|y|z)
    return sorted(((v,k&full) for k,v in proposals.items()),key=lambda a:a[0])[:4]


def rules(task):
    n=len(task['train'])
    grid=[np.array(e['input']) for e in task['train']+task['test']]
    target=[np.array(e['output']) for e in task['train']]
    if any(x.shape!=y.shape for x,y in zip(grid,target)):
        return
    delta=[g!=y for g,y in zip(grid,target)]
    paints=sorted(set.union(*(set(y[d]) for y,d in zip(target,delta))))
    if not 1<=len(paints)<=4:
        return
    widths=[g.size for g in grid]
    for canonical in (False,True):
        if canonical:
            palettes=[np.argsort(-np.bincount(g.ravel(),minlength=10),kind='stable') for g in grid]
            xs=[np.argsort(c)[g] for g,c in zip(grid,palettes)]
            ys=[np.argsort(c)[y] for c,y in zip(palettes,target)]
            colors=sorted(set.union(*(set(y[d]) for y,d in zip(ys,delta))))
        else:
            palettes=[np.arange(10)]*len(grid)
            xs,ys,colors=grid,target,paints
        if len(colors)>4:
            continue
        bank={}
        for mode in ('pixel','span','object'):
            frames=[measures(g,mode) for g in xs]
            keys=sorted(set.union(*(set(frame) for frame in frames[:n])),key=repr)
            for key in keys:
                cost, _ = next(f[key] for f in frames[:n] if key in f)
                values=[f.get(key,(cost,0))[1] for f in frames]
                demo=pack(values[:n],widths[:n])
                if demo:
                    keypair=demo,pack(values[n:],widths[n:])
                    bank[keypair]=min(cost,bank.get(keypair,cost))
        full=(1<<sum(widths[n:]))-1
        variants=[]
        for ink in colors:
            changed=pack([layers.bits(d&(y==ink)) for d,y in zip(delta,ys)],widths[:n])
            allowed=pack([layers.bits(y==ink) for y in ys],widths[:n])
            fits=mask_programs(bank,changed,allowed,full)
            if not fits:
                break
            variants.append(fits)
        else:
            for choices in sorted(product(*variants),key=lambda row:sum(c for c,_ in row))[:24]:
                occupied=0
                for _,q in choices:
                    if occupied&q:
                        break
                    occupied|=q
                else:
                    outputs=[g.copy() for g in xs[n:]]
                    for ink,(_,q) in zip(colors,choices):
                        for out in outputs:
                            cells=np.unpackbits(np.frombuffer((q&((1<<out.size)-1)).to_bytes((out.size+7)//8,'little'),
                                                               dtype=np.uint8),bitorder='little')[:out.size]
                            out.ravel()[cells.astype(bool)]=ink
                            q >>= out.size
                    yield sum(c for c,_ in choices)+int(canonical),[palette[out].tolist() for palette,out in zip(palettes[n:],outputs)]
