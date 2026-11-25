import unittest
from unittest.mock import patch
import numpy as np

from palette import variants
from verify import combine, heldout, prepare


class EvidenceTests(unittest.TestCase):
    def test_subset_rule_must_predict_the_excluded_example(self):
        def memorizer(demos):
            out=demos[0][1]
            return lambda _:out
        def mirror(_):
            return lambda x:x[:,::-1]
        task={'train':[{'input':[[1,0]],'output':[[0,1]]},
                       {'input':[[2,0]],'output':[[0,2]]}],
              'test':[{'input':[[3,0]]}]}
        demos,queries=prepare(task)
        with patch('verify.signal.setitimer'):
            claims=heldout(demos,queries,[memorizer,mirror])
        rows=combine([[]],claims)
        self.assertEqual(len(rows[0]),1)
        self.assertEqual(rows[0][0]['grid'],[[0,3]])
        self.assertFalse(rows[0][0]['full_fit'])
        self.assertEqual(rows[0][0]['folds'],[0,1])

    def test_palette_inverse_recovers_output_only_colors(self):
        task={'train':[{'input':[[7,7,3]],'output':[[3,7,9]]}],
              'test':[{'input':[[7,3,7]]}]}
        demos,_=prepare(task)
        views=list(variants(task,demos))
        self.assertTrue(views)
        for transformed,inverse in views:
            result=inverse[np.array(transformed['train'][0]['output'])]
            np.testing.assert_array_equal(result,task['train'][0]['output'])


if __name__=='__main__':
    unittest.main()
