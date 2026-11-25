"""Predict a color-agnostic output skeleton and reconstruct its paint program."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'rc23'),str(ROOT/'rc20')]
import bank
from project import fields


def background(grid):
    return int(np.bincount(grid.ravel(),minlength=10).argmax())


def encodings(inputs,targets):
    common_colors=set.intersection(*(set(g.ravel()) for g in targets))
    foreground=[background(g) for g in targets]
    candidates=[('dominant output',foreground),
                ('dominant input',[background(g) for g in inputs])]
    for c in sorted(common_colors):
        if all(np.mean(y==c)>=.22 for y in targets):
            candidates.append((f'color {c}',[c]*len(targets)))
    seen=set()
    for _,colors in candidates:
        if tuple(colors) in seen:
            continue
        seen.add(tuple(colors))
        masks=[(g!=c).astype(np.int16) for g,c in zip(targets,colors)]
        if all(np.any(m) and np.any(m==0) for m in masks):
            yield masks
        if len(seen)>=3:
            break


def properties(image,mask):
    h,w=mask.shape
    r,c=np.indices(mask.shape)
    maps={'skeleton':mask,'row half':(2*r>=h),'column half':(2*c>=w),
          'boundary distance':np.minimum.reduce([r,c,h-r-1,w-c-1]),
          'foreground depth':np.minimum(9,np.maximum(0,ndimage.distance_transform_cdt(mask))),
          'row load':np.broadcast_to(mask.sum(axis=1)[:,None],mask.shape),
          'column load':np.broadcast_to(mask.sum(axis=0),mask.shape)}
    labels,n=ndimage.label(mask)
    sizes=np.bincount(labels.ravel(),minlength=n+1)
    sizes[0]=0
    maps['component size']=sizes[labels]
    maps['largest component']=(mask!=0)&(sizes[labels]==sizes.max())
    for name,values in fields(image,mask.shape):
        if name in ('checker','row parity','column parity','object scale','object tile','reshape') or isinstance(name,tuple):
            maps[name]=values
    return maps


def compile_decoder(frames,expected):
    common=sorted(set.intersection(*(set(f) for f in frames)),key=repr)
    colors=np.concatenate([y.ravel() for y in expected])
    states=[('skeleton',),*(('skeleton',k) for k in common if k!='skeleton')]
    states.extend(('skeleton',a,b) for a,b in combinations([k for k in common if k!='skeleton'],2))
    bases=32,1024
    used=set()
    for keys in states:
        blocks=[codes(f,keys,bases) for f in frames]
        merged=np.concatenate([a.ravel() for a in blocks])
        signature=merged.tobytes()
        if signature in used:
            continue
        used.add(signature)
        labels=np.full(int(merged.max())+1,-1,dtype=np.int8)
        labels[merged]=colors
        if not np.array_equal(labels[merged],colors) or np.count_nonzero(labels>=0)>24:
            continue
        yield keys,labels


def codes(frame,keys,bases):
    a=frame[keys[0]].astype(np.int32)
    for k,m in zip(keys[1:],bases):
        values=frame[k].astype(np.int32)
        # Coordinate and count fields are bounded before packing key tuples.
        a+=np.minimum(values,31)*m
    return a


def solve(task):
    demos=task['train']
    inputs=[np.array(e['input']) for e in demos]
    targets=[np.array(e['output']) for e in demos]
    queries=[np.array(q['input']) for q in task['test']]
    result=[{} for _ in queries]
    if not any(len(np.unique(y))>=3 for y in targets):
        return [dict(status='inapplicable',candidates=[]) for _ in queries]
    n=len(demos)
    for masks in encodings(inputs,targets):
        train_frames=[properties(x,m) for x,m in zip(inputs,masks)]
        decoders=[]
        for keys,lookup in compile_decoder(train_frames,targets):
            decoders.append((keys,lookup))
            if len(decoders)>=30:
                break
        if not decoders:
            continue
        for stripped in (False,True):
            arrays=[(x!=background(x)).astype(int) if stripped else x for x in inputs+queries]
            sub=dict(train=[dict(input=x.tolist(),output=m.tolist()) for x,m in zip(arrays,masks)],
                     test=[dict(input=x.tolist()) for x in arrays[n:]])
            claims=bank.generate(sub,bank.DETECTORS,(0,))
            for row,options,q in zip(result,claims,queries):
                for option in options[:6]:
                    m=np.asarray(option['grid'])
                    if not set(m.ravel())<={0,1}:
                        continue
                    frame=properties(q,m)
                    for keys,lut in decoders:
                        if any(k not in frame for k in keys):
                            continue
                        indices=codes(frame,keys,(32,1024))
                        if indices.max()>=len(lut):
                            continue
                        g=lut[indices]
                        if (g<0).any():
                            continue
                        name=g.shape,g.tobytes()
                        evidence=dict(grid=g.tolist(),channels=len(keys),stripped=stripped,
                                      detectors=list(option['detectors']))
                        if name not in row or len(keys)<row[name]['channels']:
                            row[name]=evidence
    return [dict(status='ok',candidates=sorted(row.values(),key=lambda c:(c['channels'],c['stripped'],
                  -len(c['detectors'])))[:24]) for row in result]


def process(job):
    path,out=job
    task=json.loads(path.read_text())
    if not task['train'] or not task['test'] or any('output' in q for q in task['test']):
        raise ValueError('Expected demonstrations and unlabeled queries')
    bank.write_rows(out/path.name,solve(task))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('blind','output','detectors'):
        p.add_argument(name,type=Path)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    paths=[p for p in sorted(a.blind.glob('*.json')) if not (a.resume and (a.output/p.name).is_file())]
    with ProcessPoolExecutor(a.workers,initializer=bank.init,
                             initargs=(a.detectors.resolve(),(0,))) as pool:
        list(pool.map(process,[(p,a.output) for p in paths]))
