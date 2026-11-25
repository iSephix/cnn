import unittest
from search import solve
from stamp import rules


class CompositionTest(unittest.TestCase):
    def test_crop_then_decode(self):
        def example(color):
            return {'input':[[0,0,0,0],[0,color,2,0],[0,0,0,0]],
                    'output':[[4,4,3,3]]}
        task={'train':[example(1),example(1)],'test':[{'input':[[0,0,0],[2,1,0]]}]}
        self.assertIn([[3,3,4,4]],[c[-1] for c in solve(task)[0]['candidates']])

    def test_stamp_transfers(self):
        task={'train':[{'input':[[0,1,0,0]],'output':[[0,1,2,0]]},
                       {'input':[[1,0,0,0]],'output':[[1,2,0,0]]}],
              'test':[{'input':[[0,0,1,0,1,0]]}]}
        self.assertTrue(any(out[0].tolist()==[[0,0,1,2,1,2]] for _,out in rules(task)))

    def test_labeled_query(self):
        e={'input':[[0]],'output':[[1]]}
        with self.assertRaises(ValueError):
            solve({'train':[e],'test':[e]})

if __name__=='__main__':
    unittest.main()
