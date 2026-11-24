"""Learn a dictionary of output tiles indexed by local input context."""
from itertools import product
import numpy as np

def patches(xs,ys,qs):
    a=xs[0].grid
    sy,sx=ys[0].shape[0]//len(a),ys[0].shape[1]//a.shape[1]
    if min(sy,sx)<1 or max(sy,sx)>8:return
    if any(y.shape!=(len(s.grid)*sy,s.grid.shape[1]*sx) for s,y in zip(xs,ys)):return
    if any(max(len(q.grid)*sy,q.grid.shape[1]*sx)>30 for q in qs):return
    neighborhoods=[[(0,0)],[(0,0),(-1,0),(1,0)],[(0,0),(0,-1),(0,1)],
                   [(0,0),(-1,0),(0,1),(1,0),(0,-1)],list(product(range(-1,2),repeat=2))]
    for offsets in neighborhoods:
        for relative in (True,False):
            def features(s):
                g=s.grid
                p=np.pad(g,1,constant_values=-1)
                views=[p[1+y:1+y+len(g),1+x:1+x+g.shape[1]] for y,x in offsets]
                if relative:
                    views=[np.where(v<0,3,np.where(v==s.bg,0,np.where(v==g,1,2))) for v in views]
                return np.stack(views,axis=-1).reshape(-1,len(offsets))
            rules={}
            valid=True
            for s,y in zip(xs,ys):
                tiles=y.reshape(len(s.grid),sy,s.grid.shape[1],sx).transpose(0,2,1,3).reshape(-1,sy,sx)
                for key,tile,color in zip(features(s),tiles,s.grid.flat):
                    key=tuple(key)
                    # Intersect possible output expressions at every tile position.
                    expr=(1<<tile) | ((tile==color).astype(int)<<10)
                    if key in rules:expr=rules[key]&expr
                    if not expr.all():
                        valid=False
                        break
                    rules[key]=expr
                if not valid:break
            if not valid:continue
            outs=[]
            for q in qs:
                result=[]
                for key,c in zip(features(q),q.grid.flat):
                    expr=rules.get(tuple(key))
                    if expr is None:break
                    result.append(np.where(expr>=1024,c,np.log2(np.maximum(expr,1)).astype(int)))
                if len(result)!=q.grid.size:break
                out=np.array(result).reshape(len(q.grid),q.grid.shape[1],sy,sx).transpose(0,2,1,3)
                outs.append(out.reshape(len(q.grid)*sy,q.grid.shape[1]*sx))
            if len(outs)==len(qs):
                yield len(offsets)+len(rules),['tile',sy,sx,offsets,relative],outs
