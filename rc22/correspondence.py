"""Recover conserved objects and infer a shared placement program."""
from itertools import combinations, product
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rc21'))
from bind import objects


def orient(points, turn):
    r,c=points.T
    if turn>=4:
        c=-c
    return np.column_stack(((r,c),(-c,r),(-r,-c),(c,-r))[turn%4])


def local(points, turn):
    transformed=orient(points,turn)
    return transformed-transformed.min(axis=0)


def shape_key(points):
    return tuple(sorted(tuple(map(int,p)) for p in points))


def selectors(items):
    colors=sorted({p['color'] for p in items})
    areas=[p['area'] for p in items]
    if not areas:
        return {}
    return {'all':items,'smallest':[p for p in items if p['area']==min(areas)],
            'largest':[p for p in items if p['area']==max(areas)],
            **{int(c):[p for p in items if p['color']==c] for c in colors}}


def anchor_of(item, peers, sort):
    if not peers:
        return
    if sort=='nearest':
        return min(peers,key=lambda p:np.abs(p['center']-item['center']).sum())
    if sort=='farthest':
        return max(peers,key=lambda p:np.abs(p['center']-item['center']).sum())
    return peers[0] if len(peers)==1 else None


def axis_options(source, anchor, extent):
    start=source.min(axis=0)
    lower,upper=anchor.min(axis=0),anchor.max(axis=0)
    return {'stay':start,
            'start':lower,'center':(lower+upper-extent)//2,'end':upper-extent,
            'before':lower-extent-1,'after':upper+1,
            'reflect':lower+upper-start-extent}


def points_for(item, anchor, turn, placement):
    points=item['points']
    rel=local(points,turn)
    if anchor is None:
        return rel+points.min(axis=0)+placement
    extent=rel.max(axis=0)
    axis=axis_options(points,anchor['points'],extent)
    return rel+np.array([axis[k][i]+pad for i,(k,pad) in enumerate(placement)])


def render(grid, items, program):
    selected,turn,relation,pick,placement,style,paint,consume=program
    pool=selectors(items)
    movers=pool.get(selected,[])
    if not movers:
        return
    mover_ids={id(p) for p in movers}
    anchors=[p for p in pool.get(relation,[]) if id(p) not in mover_ids] if relation is not None else []
    out=grid.copy()
    bg=int(np.bincount(grid.ravel()).argmax())
    moves=[]
    for item in movers:
        anchor=anchor_of(item,anchors,pick) if relation is not None else None
        if relation is not None and anchor is None:
            return
        new=points_for(item,anchor,turn,placement)
        if np.any(new<0) or np.any(new>=grid.shape):
            return
        old=item['points']
        color=(grid[tuple(old.T)] if paint=='source' else
               anchor['color'] if paint=='anchor' and anchor is not None else paint)
        if paint=='anchor' and anchor is None:
            return
        if consume and anchor is not None:
            out[tuple(anchor['points'].T)]=bg
        moves.append((old,new,color,anchor))
    if style=='move':
        for old,_,_,_ in moves:
            out[tuple(old.T)]=bg
    for _,new,color,_ in moves:
        out[tuple(new.T)]=color
    return out,moves


def proposals(grid,target,parts,targets):
    pool=selectors(parts)
    dest={}
    for piece in targets:
        coords=piece['points']
        key=shape_key(local(coords,0))
        dest.setdefault(key,[]).append(coords.min(axis=0))
    seen=set()
    for selected,movers in pool.items():
        if not movers:
            continue
        member=movers[0]
        mover_ids={id(p) for p in movers}
        pivots={k:[p for p in ps if id(p) not in mover_ids] for k,ps in pool.items()}
        old=member['points']
        colors=grid[tuple(old.T)]
        if len(colors)>180:
            continue
        for turn in range(8):
            rel=local(old,turn)
            key=shape_key(rel)
            origins=list(dest.get(key,[]))
            # A moved object can touch or overlap another object, hiding its component boundary.
            if len(rel)>=2:
                h,w=target.shape
                far=rel.max(axis=0)
                candidate=np.argwhere(target==colors[0])-rel[0]
                valid=np.all((candidate>=0)&(candidate+far<[h,w]),axis=1)
                for new in candidate[valid][:120]:
                    coords=rel+new
                    if np.array_equal(target[tuple(coords.T)],colors):
                        origins.append(new)
            old_origin=old.min(axis=0)
            visited=set()
            for new in origins:
                new_key=tuple(new)
                if new_key in visited:
                    continue
                visited.add(new_key)
                target_colors=target[tuple((rel+new).T)]
                paints=[]
                if np.array_equal(colors,target_colors):
                    paints.append('source')
                if len(set(target_colors))==1:
                    paints.append(int(target_colors[0]))
                if not paints:
                    continue
                offset=tuple(map(int,new-old_origin))
                for mode,paint in product(('move','copy'),paints):
                    if not turn and offset==(0,0) and paint=='source':
                        continue
                    yield 3+int(turn!=0)+int(mode=='copy')+int(paint!='source'),(
                        selected,turn,None,None,offset,mode,paint,False)
                for anchor_key,peers in pivots.items():
                    for pick in ('unique','nearest','farthest'):
                        anchor=anchor_of(member,peers,pick)
                        if anchor is None:
                            continue
                        opts=axis_options(old,anchor['points'],rel.max(axis=0))
                        dims=[[(name,int(want-point[i])) for name,point in opts.items()
                               if abs(want-point[i])<=1] for i,want in enumerate(new)]
                        anchor_paints=paints+(['anchor'] if np.all(target_colors==anchor['color']) else [])
                        anchor_pixels=tuple(anchor['points'].T)
                        modes=(False,True) if np.any(grid[anchor_pixels]!=target[anchor_pixels]) else (False,)
                        for pose,mode,paint,consume in product(product(*dims),('move','copy'),anchor_paints,modes):
                            if not turn and paint=='source' and not consume and all(name=='stay' and pad==0 for name,pad in pose):
                                continue
                            p=(selected,turn,anchor_key,pick,pose,mode,paint,consume)
                            if p in seen:
                                continue
                            seen.add(p)
                            cost=(4+int(turn!=0)+int(pick=='farthest')+int(mode=='copy')
                                  +sum(abs(pad) for _,pad in pose)+int(isinstance(paint,int))+int(consume))
                            yield cost,p


def mask_key(masks):
    key=0
    for mask in reversed(masks):
        raw=np.packbits(mask.ravel(),bitorder='little').tobytes()
        key=(key<<mask.size)|int.from_bytes(raw,'little')
    return key


def explain(program, evidence, separated, diagonal):
    selected,turn,anchor,pick,placement,style,paint,consume=program
    return dict(selection=selected,orientation=turn,anchor=anchor,anchor_choice=pick,
                placement=placement,operation=style,paint=paint,consume_anchor=consume,
                diagonal_connectivity=diagonal==2,color_components=separated,
                demonstrations=[dict(objects=[
                    dict(from_origin=old.min(axis=0).tolist(),to_origin=new.min(axis=0).tolist(),
                         anchor_origin=pivot['points'].min(axis=0).tolist() if pivot is not None else None)
                    for old,new,_,pivot in moves]) for moves in evidence])


def combine_writes(grids, parts):
    outputs=[]
    for i,g in enumerate(grids):
        out=g.copy()
        for outputs_of_rule in parts:
            v=outputs_of_rule[i]
            writes=v!=g
            if np.any(writes&(out!=g)&(out!=v)):
                return
            out[writes]=v[writes]
        outputs.append(out.tolist())
    return outputs


def fit(task):
    demos=task['train']
    n=len(demos)
    imgs=[np.array(e['input']) for e in demos+task['test']]
    targets=[np.array(e['output']) for e in demos]
    if any(x.shape!=y.shape for x,y in zip(imgs,targets)):
        return
    required=mask_key([x!=y for x,y in zip(imgs,targets)])
    if not required:
        return
    for separated,diagonal in ((True,1),(False,1),(True,2),(False,2)):
        parts=[objects(g,separated,diagonal) for g in imgs]
        if any(not 1<=len(items)<=20 for items in parts):
            continue
        target_parts=objects(targets[0],separated,diagonal)
        if not target_parts or len(target_parts)>32:
            continue
        rules=sorted(proposals(imgs[0],targets[0],parts[0],target_parts),key=lambda pair:pair[0])
        tested=set()
        partial={}
        for cost,program in rules:
            if program in tested:
                continue
            tested.add(program)
            if len(tested)>850:
                break
            evidence=[]
            masks=[]
            for grid,items,want in zip(imgs,parts,targets):
                applied=render(grid,items,program)
                if applied is None:
                    break
                output,moves=applied
                changed=output!=grid
                if np.any(changed&(output!=want)):
                    break
                masks.append(changed)
                evidence.append(moves)
            else:
                covered=mask_key(masks)
                if not covered:
                    continue
                outputs=[]
                for grid,items in zip(imgs[n:],parts[n:]):
                    applied=render(grid,items,program)
                    if applied is None:
                        break
                    outputs.append(applied[0])
                if len(outputs)!=len(task['test']):
                    continue
                if covered==required:
                    trace=explain(program,evidence,separated,diagonal)
                    yield cost,[g.tolist() for g in outputs],dict(exact=True,programs=[trace])
                else:
                    # Keep competing interpretations with different query behavior.
                    signature=(covered,tuple(g.tobytes() for g in outputs))
                    if signature not in partial:
                        partial[signature]=(cost,program,evidence,outputs)
        bank=sorted(partial.items(),key=lambda item:(item[1][0],-item[0][0].bit_count()))[:36]
        for count in (2,3):
            for group in combinations(bank,count):
                masks=[key[0] for key,_ in group]
                coverage=0
                for mask in masks:
                    coverage|=mask
                if coverage!=required:
                    continue
                if any(masks[i]&~union==0 for i in range(count)
                       for union in [masks[(i+1)%count] | (masks[(i+2)%count] if count==3 else 0)]):
                    continue
                outputs=combine_writes(imgs[n:],[v[3] for _,v in group])
                if outputs is not None:
                    trace=[explain(p,e,separated,diagonal) for _,(_,p,e,_) in group]
                    yield count+sum(v[0] for _,v in group),outputs,dict(exact=True,programs=trace)
