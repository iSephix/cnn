import unittest

import charts
import motion
import stencils


class TransferTests(unittest.TestCase):
    def test_stencil_rebinds_ink_and_anchor(self):
        source={'train':[{'input':[[0,1,0]],'output':[[2,1,2]]}]}
        target={'train':[{'input':[[0,4,0]],'output':[[7,4,7]]}],
                'test':[{'input':[[0,0,4,0,0]]}]}
        self.assertIn([[[0,7,4,7,0]]],
                      [out for _,out in stencils.rules(target,stencils.mine(source))])

    def test_colors_move_independently(self):
        task={'train':[{'input':[[0,1,0,2,0]],'output':[[1,0,0,0,2]]}],
              'test':[{'input':[[0,0,1,0,2,0]]}]}
        self.assertIn([[[0,1,0,0,0,2]]],[out for _,out in motion.rules(task)])

    def test_chart_changes_width_with_counts(self):
        task={'train':[{'input':[[1,2,2,0]],'output':[[1,0],[2,2]]},
                       {'input':[[1,2,2,2,0]],'output':[[1,0,0],[2,2,2]]}],
              'test':[{'input':[[1,1,2,2,2,2,0]]}]}
        self.assertIn([[[1,1,0,0],[2,2,2,2]]],[out for _,out in charts.rules(task)])


if __name__=='__main__':
    unittest.main()
