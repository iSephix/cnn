import unittest

import latent
import relations
import render


class SceneTests(unittest.TestCase):
    def test_bridge_stretches_with_distance(self):
        task={'train':[{'input':[[0,0,0,0,0],[0,1,0,2,0],[0,0,0,0,0]],
                       'output':[[0,0,0,0,0],[0,1,3,2,0],[0,0,0,0,0]]}],
              'test':[{'input':[[0,0,0,0,0,0],[0,1,0,0,2,0],[0,0,0,0,0,0]]}]}
        expected=[[[0,0,0,0,0,0],[0,1,3,3,2,0],[0,0,0,0,0,0]]]
        self.assertIn(expected,[outputs for _,outputs in relations.rules(task)])

    def test_render_expands_a_box(self):
        task={'train':[{'input':[[0,0,0,0],[0,1,0,0],[0,1,1,0],[0,0,0,0]],
                       'output':[[0,0,0,0],[0,1,2,0],[0,1,1,0],[0,0,0,0]]}],
              'test':[{'input':[[0,0,0,0],[0,0,1,0],[0,1,1,0],[0,0,0,0]]}]}
        expected=[[[0,0,0,0],[0,2,1,0],[0,1,1,0],[0,0,0,0]]]
        self.assertIn(expected,[outputs for _,outputs in render.rules(task)])

    def test_latent_scenarios_leave_targets_intact(self):
        task={'train':[{'input':[[0,0,0],[0,1,0],[0,0,0]],'output':[[2]]}],
              'test':[{'input':[[0,4,0]]}]}
        variants=list(latent.scenarios(task))
        self.assertIn({'train':[{'input':[[1]],'output':[[2]]}],
                       'test':[{'input':[[4]]}]},variants)


if __name__=='__main__':
    unittest.main()
