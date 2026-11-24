"""Induce threshold programs and rank them by held-out demonstration performance."""
from pathlib import Path
import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rc8'))
from memory import Frame
from card import compile_tree
from fields import local, relative, components


def fit(sets, targets, indices):
    x = np.concatenate([sets[i] for i in indices])
    y = np.concatenate([targets[i] for i in indices])
    return DecisionTreeClassifier(random_state=0,max_depth=12).fit(x,y)


def infer(task):
    demos = task['train']
    n = len(demos)
    if n<2 or any(np.shape(e['input'])!=np.shape(e['output']) for e in demos):
        return
    for canonical in (False,True):
        frames = [Frame(e['input'],canonical) for e in demos+task['test']]
        targets = [f.target(e['output']).ravel() for f,e in zip(frames,demos)]
        groups = []
        for make in (local,relative,components):
            groups.append([np.stack([p.ravel() for p in make(f.g)],axis=1) for f in frames])
        for subset in ((0,), (1,), (2,), (0,1,2)):
            sets = [np.column_stack([f.g.ravel(),*(groups[s][i] for s in subset)])
                    for i,f in enumerate(frames)]
            for copy in (True,False):
                ys = [np.where(y==f.g.ravel(),10,y) if copy else y for f,y in zip(frames,targets)]
                model = fit(sets,ys,range(n))
                if any(not np.array_equal(model.predict(x),y) for x,y in zip(sets,ys)):
                    continue
                wrong, failed = 0.0, 0
                for i in range(n):
                    fold = fit(sets,ys,[j for j in range(n) if j!=i])
                    pred = fold.predict(sets[i])
                    decoded = np.where(pred==10,frames[i].g.ravel(),pred)
                    err = np.mean(decoded!=targets[i])
                    wrong += err
                    failed += int(err>0)
                outs = []
                for x,f in zip(sets[n:],frames[n:]):
                    v = model.predict(x).reshape(f.g.shape)
                    outs.append(f.decode(np.where(v==10,f.g,v)).tolist())
                yield (failed,wrong,model.get_n_leaves(),len(subset)), ['induce',subset,canonical,copy,compile_tree(model)],outs
