import numpy as np
from search import infer,pack,step,unpack

def example(seed):
    rng=np.random.default_rng(seed)
    a=rng.integers(1,9,size=(3,3))
    g=np.pad(a,((0,1),(1,2)))
    y=unpack(step(step(pack(g),'mirror-side'),'crop'))
    return g,y

if __name__=='__main__':
    train=[example(i) for i in range(3)]
    q,target=example(10)
    gen=infer([g for g,_ in train],[y for _,y in train],[q],True,700)
    assert any(p==['mirror-side','crop'] and o[0]==target.tolist() for _,p,o in gen)
    print('Recovered mirror-side followed by crop on an unseen grid.')
