"""Regression checks for separability, panel parsing and query prediction."""
import itertools
import numpy as np
from combine import compatible
from geometry import infer
from tiles import infer as infer_tiles

# A joint XOR rule has no single-mask explanation.
a = np.array([0,0,1,1], dtype=bool)
b = np.array([0,1,0,1], dtype=bool)
masks = np.array([a,b])
assert len(compatible(np.zeros(4), a ^ b, masks)[1]) == 0
assert list(compatible(a, a ^ b, masks)[1]) == [1]

# Compare candidate filtering against brute-force label consistency.
rng = np.random.default_rng(9)
for _ in range(100):
    keys = rng.integers(3, size=8)
    labels = rng.integers(3, size=8)
    masks = rng.integers(2, size=(15,8)).astype(bool)
    expected = [i for i,m in enumerate(masks)
                if all(len(set(labels[(keys==k) & (m==v)]))<=1
                       for k,v in itertools.product(np.unique(keys), (False,True)))]
    assert list(compatible(keys, labels, masks)[1]) == expected

def cross(n, r, c):
    g = np.zeros((n,n), dtype=int)
    g[r,c] = 1
    y = g.copy()
    y[r,:] = y[:,c] = 2
    y[r,c] = 1
    return {'input':g.tolist(), 'output':y.tolist()}

demo = [cross(4,1,2), cross(5,3,1)]
query = cross(6,2,4)
task = dict(train=demo, test=[{'input':query['input']}])
assert any(grids==[query['output']] for _,_,grids in infer(task))

def mosaic(colors):
    square = np.array(colors).repeat(2,0).repeat(2,1)
    return dict(input=np.insert(np.insert(square,2,9,axis=0),2,9,axis=1).tolist(),output=colors)

demo = [mosaic([[1,2],[3,4]]), mosaic([[4,3],[1,2]])]
query = mosaic([[2,1],[4,3]])
task = dict(train=demo, test=[{'input':query['input']}])
assert any(grids==[query['output']] for _,_,grids in infer_tiles(task))
print('Pair filtering, spatial paint, and panel reduction passed.')
