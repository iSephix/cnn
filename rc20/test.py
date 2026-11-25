import unittest
import numpy as np

import objects
import project
from shape import rank_sizes


class SolverTests(unittest.TestCase):
    def test_dimension_tracks_object_extent(self):
        dims=rank_sizes([np.array([[0,0,1,0,0]]),np.array([[0,0,2,2,0,0]])],
                        [np.zeros((1,1)),np.zeros((1,2))],np.array([[0,0,3,3,3,0,0]]))
        self.assertIn((1,3),dims[:3])

    def test_projection_transfers_tile(self):
        task={'train':[{'input':[[1,2]],'output':[[1,2,1,2]]},
                       {'input':[[2,1]],'output':[[2,1,2,1]]}],
              'test':[{'input':[[2,1,2]]}]}
        self.assertIn([[[2,1,2,2,1,2]]],[out for _,out in project.rules(task)])

    def test_object_action_depends_on_size_rank(self):
        task={'train':[{'input':[[0,1,0,2,2,0]],'output':[[0,3,0,2,2,0]]},
                       {'input':[[0,1,1,0,2,0]],'output':[[0,1,1,0,3,0]]}],
              'test':[{'input':[[0,4,0,5,5,5,0]]}]}
        self.assertIn([[[0,3,0,5,5,5,0]]],[out for _,out in objects.rules(task)])


if __name__=='__main__':
    unittest.main()
