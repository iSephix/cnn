import unittest
import numpy as np

from correspondence import fit, local
from extract import extract


def scene(width, left, anchor, answer=False):
    grid=np.zeros((5,width),dtype=int)
    marker=anchor-2 if answer else left
    grid[1,marker:marker+2]=2
    grid[2,marker]=2
    grid[1:4,anchor]=4
    return grid.tolist()


class CorrespondenceTest(unittest.TestCase):
    def test_rotation_is_relative_to_object_origin(self):
        a=np.array([[3,5],[3,6],[4,5]])
        b=local(a,1)
        np.testing.assert_array_equal(b,[[1,0],[0,0],[1,1]])

    def test_learns_variable_distance_to_anchor(self):
        demo=[{'input':scene(w,l,a),'output':scene(w,l,a,True)}
              for w,l,a in [(8,0,5),(9,1,7)]]
        task={'train':demo,'test':[{'input':scene(11,2,9)}]}
        target=scene(11,2,9,True)
        matches=[t for _,out,t in fit(task) if out==[target] and t['programs'][0]['anchor'] is not None]
        self.assertTrue(matches)
        self.assertTrue(matches[0]['exact'])
        self.assertEqual(len(matches[0]['programs'][0]['demonstrations']),2)

    def test_two_independent_motions_compose(self):
        def make(width, row, answer=False):
            grid=np.zeros((7,width),dtype=int)
            a=3 if answer else 1
            b=width-5 if answer else width-3
            grid[row,a:a+2]=2
            grid[row+3,b]=4
            grid[row+4,b]=4
            return grid.tolist()
        task={'train':[{'input':make(12,0),'output':make(12,0,True)},
                       {'input':make(14,1),'output':make(14,1,True)}],
              'test':[{'input':make(16,2)}]}
        target=make(16,2,True)
        self.assertTrue(any(out==[target] and len(trace['programs'])==2
                            for _,out,trace in fit(task)))

    def test_extracts_congruent_objects_by_frequency(self):
        def make(color, shift):
            grid=np.zeros((8,11),dtype=int)
            grid[1:3,1+shift]=color
            grid[2,2+shift]=color
            grid[4:6,5+shift]=color
            grid[5,6+shift]=color
            grid[1:4,8+shift]=3
            return grid.tolist()
        target=[[4],[4],[4]]
        task={'train':[{'input':make(2,0),'output':target},{'input':make(2,1),'output':target}],
              'test':[{'input':make(2,2)}]}
        self.assertIn([target],[out for _,out,_ in extract(task)])


if __name__=='__main__':
    unittest.main()
