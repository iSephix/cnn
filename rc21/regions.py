"""Learn region paint rules from boundaries, connectivity, and enclosing colors."""
from itertools import combinations
import numpy as np
from scipy import ndimage


def describe(grid, color, diagonal):
    kernel=ndimage.generate_binary_structure(2,diagonal)
    labels,n=ndimage.label(grid==color,kernel)
    if n>100:
        return None
    result=[]
    for r,s in enumerate(ndimage.find_objects(labels),1):
        region=labels==r
        edge=ndimage.binary_dilation(region,kernel)&~region
        around=np.bincount(grid[edge],minlength=10)
        in_box=region[s]
        h,w=in_box.shape
        area=int(in_box.sum())
        outside={int(g) for g in np.flatnonzero(around)}
        y,x=np.where(region)
        borders=(int(y.min()==0)+int(x.min()==0)+int(y.max()==grid.shape[0]-1)+int(x.max()==grid.shape[1]-1))
        attributes=dict(area=area,borders=borders,enclosed=borders==0,height=h,width=w,
                        rectangular=area==h*w,around_count=len(outside),around=tuple(sorted(outside)),
                        background_adjacent=0 in outside)
        paints={'keep':int(color)}
        if outside:
            paints['boundary majority']=int(around.argmax())
            paints['boundary minority']=min(outside,key=lambda c:(around[c],c))
        result.append((region,attributes,paints))
    for _,attrs,_ in result:
        attrs['largest']=attrs['area']==max(a['area'] for _,a,_ in result)
        attrs['smallest']=attrs['area']==min(a['area'] for _,a,_ in result)
    return result


def rules(task):
    demos=task['train']; n=len(demos)
    grids=[np.array(e['input']) for e in demos+task['test']]
    targets=[np.array(e['output']) for e in demos]
    if any(a.shape!=b.shape for a,b in zip(grids,targets)):
        return
    edits=[g!=y for g,y in zip(grids,targets)]
    changed_colors=set.union(*(set(g[d]) for g,d in zip(grids,edits)))
    if len(changed_colors)!=1:
        return
    ink=int(changed_colors.pop())
    for diagonal in (1,2):
        frames=[describe(g,ink,diagonal) for g in grids]
        if any(f is None for f in frames):
            continue
        examples=[]
        for target,groups in zip(targets,frames):
            for mask,attrs,paints in groups:
                colors=np.unique(target[mask])
                if len(colors)!=1:
                    break
                color=int(colors[0])
                actions={rule for rule,reference in paints.items() if reference==color}
                actions.add(color)
                examples.append((attrs,actions))
            else:
                continue
            break
        else:
            if not examples:
                continue
            attributes=list(examples[0][0])
            for keys in [(),*((k,) for k in attributes),*combinations(attributes,2)]:
                lookup={}
                for attrs,options in examples:
                    key=tuple(attrs[f] for f in keys)
                    lookup[key]=lookup.get(key,options)&options
                    if not lookup[key]:
                        break
                else:
                    if len(lookup)>5:
                        continue
                    preference={'keep':0,'boundary majority':1,'boundary minority':2}
                    model={k:min(ops,key=lambda a:(preference.get(a,3),str(a))) for k,ops in lookup.items()}
                    results=[]
                    for g,groups in zip(grids[n:],frames[n:]):
                        out=g.copy()
                        for mask,attrs,paints in groups:
                            key=tuple(attrs[f] for f in keys)
                            action=model.get(key)
                            if action is None or (isinstance(action,str) and action not in paints):
                                break
                            out[mask]=paints[action] if isinstance(action,str) else action
                        else:
                            results.append(out.tolist())
                            continue
                        break
                    if len(results)==len(task['test']):
                        yield 3+len(keys)+len(lookup),results
