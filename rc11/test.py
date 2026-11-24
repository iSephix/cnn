"""Behavior checks for intermediate representations and their decoders."""
import numpy as np
from compose import synth
from exemplar import infer as exemplars
from latent import factor,decode
from motion import infer as motion
from rewrite import infer as rewrite
from regions import partition,measure


def task(pairs,q):
    return dict(train=[dict(input=g,output=y) for g,y in pairs],test=[dict(input=q)])


def check():
    for panels in (True,False):
        arr = np.array([[1,2,1,2,3,3],[2,1,2,1,3,3],[5,6,5,6,0,0],[6,5,6,5,0,0]])
        motifs,labels = factor([arr],2,2,panels)
        assert np.array_equal(decode(labels[0],motifs,panels),arr)
    def bars(colors):
        left = np.array(colors)
        g = np.column_stack([left,np.full(len(left),9),left[::-1]])
        return g.tolist(),np.where((left+left[::-1])==2,4,0).tolist()
    a,b = bars([[1,0],[1,1],[0,0]]),bars([[1,0],[0,1],[1,1],[1,0]])
    q,y = bars([[1,1],[0,1],[1,1]])
    assert any(outs==[y] for _,_,outs in synth(task([a,b],q)))
    def complete(color,h):
        g = np.zeros((h,8),int)
        g[0,1:3] = [color,2]
        g[h-1,4] = color
        y = g.copy()
        y[h-1,5] = 2
        return g.tolist(),y.tolist()
    a,b = complete(3,4),complete(4,5)
    q,y = complete(6,6)
    assert any(outs==[y] for _,_,outs in exemplars(task([a,b],q)))
    def slide(w,c):
        g = np.zeros((3,w),int)
        g[1,1],g[1,-2] = c,8
        y = g.copy()
        y[1,1],y[1,-3] = 0,c
        return g.tolist(),y.tolist()
    a,b = slide(8,2),slide(9,2)
    q,y = slide(11,2)
    assert any(outs==[y] for _,_,outs in motion(task([a,b],q)))
    def marker(color,pos):
        g = np.zeros((5,6),int)
        y,x = pos
        g[y,x] = color
        target = g.copy()
        target[y-1,x] = color
        target[y,x+1] = 5
        return g.tolist(),target.tolist()
    a,b = marker(1,(2,1)),marker(2,(3,4))
    q,y = marker(3,(1,2))
    assert any(outs==[y] for _,_,outs in rewrite(task([a,b],q)))
    g = np.array([[0,2,2],[0,0,0],[3,0,0]])
    features = measure(g,partition(g,True,False)).reshape(3,3,-1)
    assert features[1,2,1]==2 and features[1,2,12]==2
    try:
        list(synth({'train':[],'test':[{'input':[[0]],'output':[[0]]}]}))
    except ValueError:
        pass
    else:
        raise AssertionError('Leaked test labels')
    print('Intermediate-grid and blind-input checks passed.')


if __name__=='__main__':
    check()
