"""Infer positions for a rigid object from demonstrations, then replay its placement."""
from itertools import product
import numpy as np
from scene import Scene


def select(objects, key):
    if not objects:
        return None
    if isinstance(key, int):
        return next((o for o in objects if o.color == key), None)
    stat, side = key
    return (min if side == 'min' else max)(objects, key=lambda o: o.stats[stat])


def selectors(objects):
    return [*product(('area','shape_count','color_count','top','left'),('max','min')),
            *sorted({o.color for o in objects})]


def paint(scene, obj, row, col, copy, bg):
    dy,dx = row-obj.stats['top'],col-obj.stats['left']
    ys,xs = obj.points
    h,w=scene.grid.shape
    if row<0 or col<0 or row+obj.box.shape[0]>h or col+obj.box.shape[1]>w:
        return None
    result=scene.grid.copy()
    if not copy:
        result[ys,xs]=bg
    result[ys+dy,xs+dx]=scene.grid[ys,xs]
    return result


def destinations(scene, obj, y, copy, bg):
    # An exact rendering must cover every new non-background changed cell.
    g=scene.grid
    delta=(g!=y)
    area=obj.stats['area']
    if delta.sum()>area*(1 if copy else 2):
        return set()
    changes=np.argwhere(delta & (y!=bg))
    if not len(changes):
        return set()
    r,c=changes[0]
    ys,xs=obj.points
    points=np.flatnonzero(g[ys,xs]==y[r,c])
    out=set()
    for i in points:
        row,col=int(r-ys[i]+obj.stats['top']),int(c-xs[i]+obj.stats['left'])
        candidate=paint(scene,obj,row,col,copy,bg)
        if candidate is not None and np.array_equal(candidate,y):
            out.add((row,col))
    return out


def axis_positions(scene,obj,anchor,axis):
    start=('top','left')[axis]
    pos=int(obj.stats[start])
    size=obj.box.shape[axis]
    n=scene.grid.shape[axis]
    options={'keep':pos,'edge-start':0,'edge-end':n-size,'mirror':n-pos-size}
    if anchor is not None:
        a=int(anchor.stats[start])
        b=a+anchor.box.shape[axis]
        options.update(before=a-size,after=b,align=a,center=(a+b-size)//2,end=b-size)
    return options


def programs(scenes, targets, queries):
    if any(s.grid.shape!=y.shape or np.array_equal(s.grid,y) for s,y in zip(scenes,targets)):
        return
    for mono,conn,bg_spec in product((True,False),(4,8),('mode',0)):
        mode=mono,conn,bg_spec
        groups=[s.objects(mode) for s in scenes]
        if not all(groups):
            continue
        backgrounds=[s.bg if bg_spec=='mode' else bg_spec for s in scenes]
        cache={}
        for source in selectors(groups[0]):
            objs=[select(group,source) for group in groups]
            if any(o is None for o in objs):
                continue
            for copy in (False,True):
                def locations(i):
                    key=i,id(objs[i]),copy
                    if key not in cache:
                        cache[key]=destinations(scenes[i],objs[i],targets[i],copy,backgrounds[i])
                    return cache[key]
                if not all(locations(i) for i in range(len(objs))):
                    continue
                others=[[a for a in group if a is not o] for group,o in zip(groups,objs)]
                for anchor_key in [None,*selectors(others[0])]:
                    anchors=[None if anchor_key is None else select(group,anchor_key) for group in others]
                    if anchor_key is not None and any(a is None for a in anchors):
                        continue
                    axes=[[axis_positions(s,o,a,d) for d in range(2)] for s,o,a in zip(scenes,objs,anchors)]
                    for row,col in sorted(locations(0)):
                        options=[]
                        for d,v in enumerate((row,col)):
                            opts=[(k,0) for k,p in axes[0][d].items() if p==v]
                            opts.append(('offset',v-axes[0][d]['keep']))
                            options.append(opts)
                        for ry,rx in product(*options):
                            specs=(ry,rx)
                            anchored=any(k in ('before','after','align','center','end') for k,_ in specs)
                            if anchored!=(anchor_key is not None):
                                continue
                            def destination(axes_for_grid):
                                return tuple(a['keep' if k=='offset' else k]+off for a,(k,off) in zip(axes_for_grid,specs))
                            if not all(destination(axes[i]) in locations(i) for i in range(1,len(objs))):
                                continue
                            outs=[]
                            for q in queries:
                                group=q.objects(mode)
                                o=select(group,source)
                                a=select([p for p in group if p is not o],anchor_key) if anchor_key is not None else None
                                if o is None or (anchor_key is not None and a is None):
                                    break
                                pos=destination([axis_positions(q,o,a,d) for d in range(2)])
                                res=paint(q,o,*pos,copy,q.bg if bg_spec=='mode' else bg_spec)
                                if res is None:
                                    break
                                outs.append(res)
                            if len(outs)==len(queries):
                                cost=2+anchored+sum(k=='offset' for k,_ in specs)*2+isinstance(source,int)+isinstance(anchor_key,int)
                                yield cost,('place',mode,source,anchor_key,specs,copy),outs
