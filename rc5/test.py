import numpy as np
from compose import infer

def example(colors, flipped):
    patches=[]
    outputs=[]
    for i,c in enumerate(colors):
        p=np.zeros((4,5),int)
        shape=np.array([[c,c],[c,0]])
        if flipped ^ bool(i):
            shape=np.flipud(shape)
        p[1:3,1:3]=shape
        p[0,4]=8
        patches.append(p)
        outputs.append(np.fliplr(shape))
    return np.hstack([patches[0],np.full((4,1),9),patches[1]]),np.hstack(outputs)

if __name__=='__main__':
    a,y=example([2,3],False)
    b,z=example([4,6],True)
    q,target=example([1,7],True)
    predictions=list(infer([a,b],[y,z],[q]))
    assert any(rule==[9,'largest',2,'join'] and grids[0]==target.tolist()
               for _,rule,grids in predictions)
    print('Recovered split, largest object, reflection and reassembly on unseen colors.')
