import unittest
import numpy as np
from concepts import generate
from repair import recover
from objects import rules

class ConceptsTest(unittest.TestCase):
    def test_occluded_tile(self):
        expected = np.tile([[1,2],[2,1]], (3,3))
        damaged = expected.copy()
        damaged[2:4,1:3] = 9
        np.testing.assert_array_equal(recover(damaged,9)['plain'], expected)

    def test_unique_shape(self):
        demo = {'input': [[0,2,0,3,0,4,4,0]], 'output': [[0,2,0,3,0,8,8,0]]}
        query = {'input': [[0,1,1,0,2,0,3,0]]}
        expected = [[0,8,8,0,2,0,3,0]]
        self.assertTrue(any(out[0].tolist()==expected for _,out in
                            rules({'train':[demo],'test':[query]})))

    def test_panel(self):
        demo = {'input':[[1,1,0,0,0,1,0,1]],'output':[[2,0,0,2]]}
        query = {'input':[[0,1,1,0,0,0,1,1]]}
        rows = generate({'train':[demo],'test':[query]})
        self.assertIn([[0,2,0,2]], [c[-1] for c in rows[0]['candidates']])
        with self.assertRaises(ValueError):
            generate({'train':[demo],'test':[demo]})

if __name__ == '__main__':
    unittest.main()
