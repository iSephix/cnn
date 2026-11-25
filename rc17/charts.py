"""Render scene measurements as ordered colored bars."""
from itertools import product
import numpy as np
from scipy import ndimage


def measurements(g, mode, background):
    colors,first,counts=np.unique(g,return_index=True,return_counts=True)
    bg=0 if background=='zero' else colors[counts.argmax()]
    stats={}
    for color,index,count in zip(colors,first,counts):
        if color==bg:
            continue
        mask=g==color
        if mode=='components':
            _,count=ndimage.label(mask)
        elif mode=='rows':
            count=np.count_nonzero(mask.any(axis=1))
        elif mode=='columns':
            count=np.count_nonzero(mask.any(axis=0))
        stats[int(color)]=(int(count),int(index))
    return int(bg),stats


def render(scene, order, reverse, length, empty, align, orient, constant):
    bg,stats=scene
    if not stats:
        return None
    key={'count':lambda c:(stats[c][0],c),'appearance':lambda c:stats[c][1],'color':lambda c:c}[order]
    colors=sorted(stats,key=key,reverse=reverse)
    values=[stats[c][0] for c in colors]
    width=max(values) if length=='max' else constant
    if max(values)>width or width>30:
        return None
    fill=bg if empty=='scene' else 0
    out=np.full((len(colors),width),fill)
    for i,(c,n) in enumerate(zip(colors,values)):
        if align:
            out[i,-n:]=c
        else:
            out[i,:n]=c
    return out.T if orient else out


def rules(task):
    grids=[np.array(e['input']) for e in task['train']+task['test']]
    targets=[np.array(e['output']) for e in task['train']]
    if any(min(y.shape)>9 for y in targets):
        return
    for mode,bg in product(('pixels','components','rows','columns'),('scene','zero')):
        scenes=[measurements(g,mode,bg) for g in grids]
        for orient in (False,True):
            dim=int(orient)
            if any(len(s[1])!=target.shape[dim] for s,target in zip(scenes,targets)):
                continue
            widths={y.shape[1-dim] for y in targets}
            lengths=('max','fixed') if len(widths)==1 else ('max',)
            for order,reverse,length,empty,align in product(
                ('count','appearance','color'),(False,True),lengths,('scene','zero'),(False,True)
            ):
                spec=(order,reverse,length,empty,align,orient,max(widths))
                if not all(np.array_equal(render(s,*spec),y) for s,y in zip(scenes,targets)):
                    continue
                outputs=[render(s,*spec) for s in scenes[len(targets):]]
                if all(o is not None for o in outputs):
                    yield 2+int(length=='fixed')+int(mode!='pixels'),[o.tolist() for o in outputs]
