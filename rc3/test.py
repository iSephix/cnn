import unittest
import numpy as np
from learn import solve, pixels, Scene

def grid(*rows):
    return [[int(c) for c in row] for row in rows]

class RelationalTest(unittest.TestCase):
    def test_color_transfer(self):
        x=[grid('20002'),grid('03030')]
        y=[grid('22222'),grid('03330')]
        new=grid('70007')
        candidates=list(pixels(list(map(Scene,x)),list(map(np.array,y)),[Scene(new)]))
        self.assertTrue(any(tag[0]=='copy' and arr[0].tolist()==grid('77777') for _,tag,arr in candidates))

    def test_largest_shape(self):
        t={'train':[{'input':grid('100000','000220','000220'),'output':grid('22','22')},
                    {'input':grid('00040','33000','30000'),'output':grid('33','30')}],
           'test':[{'input':grid('60000','00777','00007')}]}
        p,s=solve(t,[{'attempt_1':[[9]],'attempt_2':[[9]]}],[])
        self.assertIn(grid('777','007'),p[0].values())
        self.assertIn('object',[v[0] for v in s[0] if isinstance(v,tuple)])

    def test_label_boundary(self):
        with self.assertRaises(ValueError):
            solve({'test':[{'output':[[1]]}]},[],[])

if __name__=='__main__':
    unittest.main()
