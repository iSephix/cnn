"""Learn reusable conditional recolorings and translated paint strokes per task."""
from itertools import product
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from card import compile_tree,predict


def cell_features(scene):
    g=scene.grid
    y,x=np.indices(g.shape)
    # Keep relative coordinates available when sizes change across demonstrations.
    coords=[y,x,len(g)-1-y,g.shape[1]-1-x,y%2,x%2,(x-y)%3,(x+y)%3]
    base=list(scene.features.values())
    extra=[g[::-1],g[:,::-1],g[::-1,::-1]]
    if g.shape[0]==g.shape[1]:
        extra.append(g.T)
    else:
        extra.append(np.full_like(g,-1))
    return np.stack([a.ravel() for a in [*base,*coords,*extra]],axis=1)


def trees(xs,ys,qs):
    if any(s.grid.shape!=y.shape for s,y in zip(xs,ys)) or len(xs)<2:
        return
    train=[cell_features(s) for s in xs]
    query=[cell_features(s) for s in qs]
    for copy in (True,False):
        targets=[np.where(y==s.grid,10,y).ravel() if copy else y.ravel() for s,y in zip(xs,ys)]
        def fit(indices):
            model=DecisionTreeClassifier(max_depth=9,min_samples_leaf=1,random_state=0)
            a=np.concatenate([train[i] for i in indices])
            b=np.concatenate([targets[i] for i in indices])
            model.fit(a,b)
            return model
        def output(model,s,features):
            colors=model.predict(features).reshape(s.grid.shape)
            return np.where(colors==10,s.grid,colors)
        model=fit(range(len(xs)))
        if any(not np.array_equal(output(model,s,a),y) for s,a,y in zip(xs,train,ys)):
            continue
        # Demand exact whole-grid validation on every held-out demonstration.
        for i in range(len(xs)):
            fold=fit([j for j in range(len(xs)) if j!=i])
            if not np.array_equal(output(fold,xs[i],train[i]),ys[i]):
                break
        else:
            card=compile_tree(model)
            outs=[]
            for q,a in zip(qs,query):
                colors=predict(card,a).reshape(q.grid.shape)
                outs.append(np.where(colors==10,q.grid,colors))
            yield model.get_n_leaves(),['tree',copy,card],outs


def strokes(xs,ys,qs):
    if any(s.grid.shape!=y.shape for s,y in zip(xs,ys)):
        return
    for selector in ['ink',*map(int,np.unique(xs[0].grid))]:
        def points(s):
            return np.where(s.grid!=s.bg if selector=='ink' else s.grid==selector)
        seeds=[points(s) for s in xs]
        if any(not len(y) or len(y)>80 for y,_ in seeds):
            continue
        rules=[]
        for dy,dx in product(range(-4,5),repeat=2):
            observed=[]
            changed=False
            for s,target,(r,c) in zip(xs,ys,seeds):
                y,x=r+dy,c+dx
                valid=(y>=0)&(y<len(target))&(x>=0)&(x<target.shape[1])
                y,x=y[valid],x[valid]
                if not len(y):continue
                vals=target[y,x]
                changed|=bool(np.any(s.grid[y,x]!=vals))
                observed.append((vals,s.grid[r[valid],c[valid]]))
            if not changed:continue
            if all(np.array_equal(v,c) for v,c in observed):
                paint='source'
            else:
                colors=np.unique(np.concatenate([v for v,_ in observed]))
                if len(colors)!=1:continue
                paint=int(colors[0])
            rules.append((dy,dx,paint))
        if not rules:continue
        def render(s):
            out=s.grid.copy()
            r,c=points(s)
            for dy,dx,paint in rules:
                y,x=r+dy,c+dx
                valid=(y>=0)&(y<len(out))&(x>=0)&(x<out.shape[1])
                out[y[valid],x[valid]]=s.grid[r[valid],c[valid]] if paint=='source' else paint
            return out
        if all(np.array_equal(render(s),y) for s,y in zip(xs,ys)):
            yield len(rules),['stroke',selector,rules],[render(q) for q in qs]
