"""Behavior checks for source pointers, canvas measurements, and independent boards."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from induce import infer as induce
from resample import infer as resample
from fields import shift
from canvas import infer as canvas
from catalog import infer as catalog
from pointers import infer as pointers
from stamps import actions,pack,unpack,infer as stamps
from merge import merge


def task(pairs,query):
    return dict(train=[dict(input=x,output=y) for x,y in pairs],test=[dict(input=query)])


def predicts(learner,t,wanted):
    return any(outputs==[wanted] for _,_,outputs in learner(t))


def run():
    g = np.array([[1,2],[3,4]])
    assert shift(g,0,1).tolist()==[[2,-1],[4,-1]]
    # A ray from the bottom of the first board cannot reach the second board.
    second_start = 12*48
    board = pack(np.ones((3,4)),0) | pack(np.ones((4,3)),second_start)
    source = pack(np.eye(3,4),0)
    for _,bits,_ in actions(source,board):
        assert not unpack(bits,second_start,(4,3)).any()
    # Periodic sampling extrapolates to a new width.
    p = [([[2,3]],[[2,3,2,3]]),([[4,5,6]],[[4,5,6,4,5,6]])]
    assert predicts(canvas,task(p,[[4,1,2,3]]),[[4,1,2,3,4,1,2,3]])
    # Reuse colors from the query to draw one step to the right.
    def move(c,pos):
        x = np.zeros((4,5),int)
        x[pos] = c
        y = x.copy()
        y[pos[0],pos[1]+1] = c
        return x.tolist(),y.tolist()
    p = [move(1,(1,1)),move(4,(2,2)),move(2,(0,1))]
    q,expected = move(7,(2,1))
    assert predicts(pointers,task(p,q),expected)
    assert predicts(stamps,task(p,q),expected)
    # Whole-grid color rules and learning which source pixel fills a new canvas.
    def recolor(grid):
        g = np.array(grid)
        return grid,np.where(g==1,7,g).tolist()
    examples = [recolor([[0,1],[0,0]]),recolor([[1,0,1],[0,0,0]])]
    q,y = recolor([[1,0],[0,1],[1,0]])
    assert predicts(induce,task(examples,q),y)
    def double(grid):
        return grid,np.repeat(grid,2,axis=0).tolist()
    examples = [double([[1,0],[0,1]]),double([[0,2],[2,2],[0,2]])]
    q,y = double([[0,3,0],[3,0,3]])
    assert predicts(resample,task(examples,q),y)
    # Count repeated colors into a descending histogram.
    examples = [([[1,0,2],[1,0,0]],[[1,2],[1,0]]),([[3,4,4],[0,0,4]],[[4,3],[4,0],[4,0]])]
    assert predicts(catalog,task(examples,[[2,1,2],[2,1,2]]),[[2,1],[2,1],[2,0],[2,0]])
    with TemporaryDirectory() as d:
        def fixture(name,grids,tags):
            p = str(Path(d)/name)
            Path(p).write_text(json.dumps({'t':[dict(attempt_1=grids[0],attempt_2=grids[1])]}))
            Path(p+'.trace.json').write_text(json.dumps({'t':[tags]}))
            return p
        old = fixture('old',([[0]],[[9]]),('prior','identity'))
        a = fixture('a',([[0]],[[1]]),('prior','induce'))
        b = fixture('b',([[0]],[[2]]),('prior','canvas'))
        c = fixture('c',([[0]],[[2]]),('prior','pointers'))
        out = str(Path(d)/'out')
        merge(old,out,[a,b,c])
        assert json.loads(Path(out).read_text())['t']==[dict(attempt_1=[[0]],attempt_2=[[2]])]
    print('Spatial induction and fusion checks passed.')


if __name__=='__main__':
    run()
