"""Acquire offset patterns from solved tasks and bind their colors on a new task."""
from itertools import product
import numpy as np

DIRECTIONS = tuple(p for p in product((-1,0,1), repeat=2) if p != (0,0))


def shift(mask, dy, dx):
    h,w = mask.shape
    result = np.zeros_like(mask)
    if abs(dy)<h and abs(dx)<w:
        result[max(0,dy):min(h,h+dy),max(0,dx):min(w,w+dx)] = (
            mask[max(0,-dy):min(h,h-dy),max(0,-dx):min(w,w-dx)]
        )
    return result


def evidence(task):
    inputs = [np.array(e['input']) for e in task['train']]
    outputs = [np.array(e['output']) for e in task['train']]
    if any(a.shape != b.shape for a,b in zip(inputs,outputs)):
        return
    changes = [a!=b for a,b in zip(inputs,outputs)]
    inks = set.union(*(set(b[m]) for b,m in zip(outputs,changes)))
    if len(inks)!=1 or not all(m.any() for m in changes):
        return
    ink = int(inks.pop())
    anchors = sorted(set.intersection(*(set(a.ravel()) for a in inputs)))
    for anchor in anchors:
        masks = [a==anchor for a in inputs]
        if any(m.sum()>60 for m in masks):
            continue
        for mode in ('all','background'):
            eligible = [np.ones_like(a,dtype=bool) if mode=='all' else a==np.bincount(a.ravel()).argmax() for a in inputs]
            yield ink,int(anchor),mode,masks,eligible,inputs,outputs,changes


def mine(task):
    found=set()
    for ink,_,mode,masks,eligible,_,outputs,changes in evidence(task):
        hits={}
        for dy,dx in product(range(-5,6),repeat=2):
            cover=[shift(m,dy,dx)&e for m,e in zip(masks,eligible)]
            if any(np.any(y[c]!=ink) for y,c in zip(outputs,cover)):
                continue
            matches=[c&m for c,m in zip(cover,changes)]
            if any(m.any() for m in matches):
                hits[dy,dx]=matches
        remaining=[m.copy() for m in changes]
        offsets=[]
        while hits and len(offsets)<12 and any(m.any() for m in remaining):
            best=max(hits,key=lambda k:(sum(np.count_nonzero(a&b) for a,b in zip(hits[k],remaining)),
                                        -sum(map(abs,k))))
            merged=[a&~b for a,b in zip(remaining,hits.pop(best))]
            if all(np.array_equal(a,b) for a,b in zip(remaining,merged)):
                break
            offsets.append(best)
            remaining=merged
        if offsets and not any(m.any() for m in remaining):
            found.add((mode,tuple(sorted(offsets))))
    return sorted(found)


def controls():
    return [(mode,tuple(sorted((dy*i,dx*i) for i in range(1,length+1))))
            for mode,length,(dy,dx) in product(('all','background'),(1,2),DIRECTIONS)]


def neighborhood_control():
    center = ((0,0),)
    ring = tuple(sorted(DIRECTIONS))
    box = tuple(sorted(ring + center))
    return [('all',center),('all',box),('all',ring),('background',ring)]


def rules(task, patterns):
    for ink,anchor,mode,masks,eligible,_,outputs,changes in evidence(task):
        for candidate_mode,offsets in patterns:
            if candidate_mode!=mode:
                continue
            for m,e,y,changed in zip(masks,eligible,outputs,changes):
                covered=np.logical_or.reduce([shift(m,*d) for d in offsets])&e
                if np.any(changed&~covered) or np.any(y[covered]!=ink):
                    break
            else:
                predictions=[]
                for q in task['test']:
                    grid=np.array(q['input'])
                    covered=np.logical_or.reduce([shift(grid==anchor,*d) for d in offsets])
                    if mode=='background':
                        covered &= grid==np.bincount(grid.ravel()).argmax()
                    grid[covered]=ink
                    predictions.append(grid.tolist())
                yield len(offsets),predictions
