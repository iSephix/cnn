import unittest
import numpy as np
from predict import solve

def example(source,anchor,colors):
    g=np.zeros((10,12),int)
    y,x=source
    r,c=anchor
    g[y:y+2,x:x+2]=colors[0]
    g[r,c]=colors[1]
    target=g.copy()
    target[y:y+2,x:x+2]=0
    target[r:r+2,c-2:c]=colors[0]
    return dict(input=g.tolist(),output=target.tolist())


def infer(demos,queries):
    task={'train':demos,'test':[{'input':e['input']} for e in queries]}
    prior=[{'attempt_1':e['input'],'attempt_2':e['input']} for e in queries]
    return solve(task,prior,[['identity','identity'] for _ in queries])


class PlacementTest(unittest.TestCase):
    def test_new_anchors_and_colors(self):
        demos=[example((0,0),(4,8),(1,2)),example((5,0),(2,7),(3,4)),example((7,8),(3,3),(2,1))]
        queries=[example((0,0),(r,c),(8,9)) for r in (2,4,6,8) for c in (4,7,10)]
        preds,trace,_=infer(demos,queries)
        for e,p,t in zip(queries,preds,trace):
            self.assertEqual(e['output'],p['attempt_1'])
            self.assertEqual(t[0][0],'place')

    def test_copy_and_retain(self):
        demos=[example((0,0),(4,8),(1,2)),example((5,0),(2,7),(3,4))]
        query=example((1,0),(5,8),(7,8))
        for e in demos+[query]:
            g=np.array(e['input'])
            y=np.array(e['output'])
            y[g!=0]=g[g!=0]
            e['output']=y.tolist()
        preds,_,_=infer(demos,[query])
        self.assertIn(query['output'],preds[0].values())

    def test_panel_overlap(self):
        def sample(a,b):
            return {'input':np.hstack([a,b]).tolist(),'output':np.where((a!=0)&(b!=0),7,0).tolist()}
        a=np.array([[1,1],[0,0]])
        b=np.array([[0,2],[2,0]])
        demos=[sample(a,b),sample(a.T,b.T)]
        query=sample(3*np.flip(a),4*b)
        pred,_,_=infer(demos,[query])
        self.assertIn(query['output'],pred[0].values())

    def test_reject_labels(self):
        with self.assertRaises(ValueError):
            solve({'test':[{'input':[[0]],'output':[[1]]}]},[],[])

if __name__=='__main__':
    unittest.main()
