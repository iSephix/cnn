"""Synthesize a shared local action selected by numerical object features."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rc18'))
from relations import scene


def objects(grid, split):
    bg,parts=scene(grid,split)
    if not parts or len(parts)>35:
        return bg,[]
    result=[]
    for color,coords,lo,hi in parts:
        rr,cc=coords.T-lo[:,None]
        box=tuple(slice(a,b+1) for a,b in zip(lo,hi))
        patch=grid[box]
        mask=np.zeros(patch.shape,bool)
        mask[rr,cc]=True
        result.append((box,patch,mask,color))
    return bg,result


def attributes(parts):
    shapes=[(mask.shape,mask.tobytes()) for _,_,mask,_ in parts]
    shape_count=Counter(shapes)
    areas=[int(m.sum()) for _,_,m,_ in parts]
    colors=Counter(c for _,_,_,c in parts)
    result=[]
    for index,(box,patch,mask,color) in enumerate(parts):
        area=areas[index]
        h,w=patch.shape
        result.append(dict(color=color,area=area,height=h,width=w,
                           size_rank=sum(v<area for v in areas),large_rank=sum(v>area for v in areas),
                           copies=shape_count[shapes[index]],color_copies=colors[color],
                           filled=int(mask.all()),aspect=int(np.sign(h-w)),
                           x_rank=sum(b[1].start<box[1].start for b,_,_,_ in parts),
                           y_rank=sum(b[0].start<box[0].start for b,_,_,_ in parts)))
    return result


def actions(patch, mask, color, bg, palette):
    yield ('keep',),patch
    yy,xx=np.indices(patch.shape)
    h,w=patch.shape
    edge=(yy==0)|(yy==h-1)|(xx==0)|(xx==w-1)
    holes=ndimage.binary_fill_holes(mask)&~mask
    for kind,r in [('ink',mask),('fill',np.ones_like(mask)),('edge',edge),
                   ('interior',~edge),('holes',holes),('not ink',~mask),
                   ('checker',(yy+xx)%2==0),('diag',(yy==xx)|(yy==w-1-xx))]:
        if not r.any():
            continue
        for name,value in [('background',bg),('own',color),*((c,c) for c in palette)]:
            estimate=np.where(r,value,patch)
            yield (kind,name),estimate
    for k in range(1,8):
        transformed=np.rot90(patch if k<4 else patch[:,::-1],k%4)
        if transformed.shape==patch.shape:
            yield ('orient',k),transformed


def rules(task):
    demos=task['train']
    n=len(demos)
    raw=[np.array(e['input']) for e in demos+task['test']]
    targets=[np.array(e['output']) for e in demos]
    if any(g.shape!=y.shape for g,y in zip(raw,targets)) or all(np.array_equal(g,y) for g,y in zip(raw,targets)):
        return
    palette=sorted(set.union(*(set(y[g!=y]) for g,y in zip(raw,targets))))
    for split in (True,False):
        scenes=[objects(g,split) for g in raw]
        if any(not parts for _,parts in scenes):
            continue
        records=[]
        for g,y,(bg,parts) in zip(raw,targets,scenes):
            coverage=np.zeros(g.shape,dtype=np.int16)
            feature=attributes(parts)
            for box,_,_,_ in parts:
                coverage[box]+=1
            if np.any(coverage[g!=y]!=1):
                break
            for attrs,(box,patch,mask,color) in zip(feature,parts):
                options={key for key,v in actions(patch,mask,color,bg,palette) if np.array_equal(v,y[box])}
                if not options:
                    break
                records.append((attrs,options))
            else:
                continue
            break
        else:
            features=list(records[0][0])
            seen=set()
            for keys in [(),*((f,) for f in features),*combinations(features,2)]:
                groups={}
                for attrs,opts in records:
                    key=tuple(attrs[f] for f in keys)
                    groups[key]=groups.get(key,opts)&opts
                    if not groups[key]:
                        break
                else:
                    if len(groups)>6:
                        continue
                    policy={k:min(choices,key=lambda a:(a!=('keep',),len(a),repr(a))) for k,choices in groups.items()}
                    outputs=[]
                    for grid,(bg,parts) in zip(raw[n:],scenes[n:]):
                        out=grid.copy()
                        written=np.zeros(grid.shape,bool)
                        for attrs,(box,patch,mask,color) in zip(attributes(parts),parts):
                            key=tuple(attrs[f] for f in keys)
                            action=policy.get(key)
                            proposal=dict(actions(patch,mask,color,bg,palette)).get(action)
                            if proposal is None:
                                break
                            changed=proposal!=patch
                            if np.any(written[box]&changed):
                                break
                            out[box][changed]=proposal[changed]
                            written[box]|=changed
                        else:
                            outputs.append(out.tolist())
                            continue
                        break
                    if len(outputs)==len(task['test']):
                        signature=tuple(tuple(map(tuple,g)) for g in outputs)
                        if signature not in seen:
                            seen.add(signature)
                            yield 2+len(keys)+len(groups),outputs
