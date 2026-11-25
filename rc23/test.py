import unittest
from unittest.mock import patch
import numpy as np

from bank import generate, orient, restore
from select_attempts import select


class PortfolioTests(unittest.TestCase):
    def test_orientation_reverses_on_non_square_grid(self):
        g=np.arange(6).reshape(2,3)
        for code in range(8):
            np.testing.assert_array_equal(restore(orient(g,code),code),g)

    def test_detectors_have_to_match_every_demonstration(self):
        task={'train':[{'input':[[1,0]],'output':[[0,1]]},{'input':[[2,0]],'output':[[0,2]]}],
              'test':[{'input':[[3,0]]}]}
        def mirror(_):
            return lambda g:g[:,::-1]
        def memorized(_):
            return lambda g:np.array([[0,1]])
        with patch('bank.signal.setitimer'):
            rows=generate(task,[mirror,memorized],(0,))
        self.assertEqual([r['grid'] for r in rows[0]],[[[0,3]]])
        self.assertEqual(rows[0][0]['identity_votes'],1)

    def test_two_sources_keep_a_distinct_attempt(self):
        base=[dict(attempt_1=[[1]],attempt_2=[[2]])]
        self.assertEqual(select(base,[[dict(grid=[[1]],identity_votes=1)]],'reserve'),base)
        self.assertEqual(select(base,[[dict(grid=[[3]],identity_votes=1)]],'reserve'),
                         [dict(attempt_1=[[1]],attempt_2=[[3]])])

    def test_query_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            generate({'train':[{'input':[[1]],'output':[[1]]}],
                      'test':[{'input':[[1]],'output':[[1]]}]},[],(0,))


if __name__=='__main__':
    unittest.main()
