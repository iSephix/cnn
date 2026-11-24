import json
import numpy as np
from memory import Frame,read,fit_binding,codes
from spatial import project

bank=read('rc8/bank.json')
grid=np.array([[2,0,1,3],[0,2,4,3]])
for k in range(8):
    for canonical in (False,True):
        f=Frame(grid,canonical,k)
        assert np.array_equal(f.decode(f.g),grid)
for card in bank:
    a=Frame(grid,card['canonical'])
    assert np.array_equal(codes(card,a),codes(json.loads(json.dumps(card)),a))
assert fit_binding([np.array([0,0])],[np.array([2,3])]) is None
mask=np.array([[False,True,False],[True,False,True],[False,True,False]])
assert project(mask,'enclosed').sum()==1
print(f'{len(bank)} cards replayed; eight spatial views round-tripped.')
