import unittest

from compose import mask_programs, pack
from panels import merge
from pack import rules


class CompositionTest(unittest.TestCase):
    def test_independent_masks_cover_one_edit(self):
        bank={(1,2):1,(2,4):1,(4,8):1,(9,1):1}
        self.assertIn((5,14),mask_programs(bank,7,7,15))

    def test_pack_keeps_query_boundaries(self):
        self.assertEqual(pack([1,4],[2,3]),17)

    def test_merge_inserts_separator(self):
        self.assertEqual(merge([[1,2]],[[3,4]],1,9),[[1,2,9,3,4]])
        self.assertIsNone(merge([[1]],[[2,3]],1,None))

    def test_objects_align_and_change_width(self):
        task={'train':[{'input':[[0,0,0,0,0],[0,1,0,2,2],[0,0,0,0,0]],'output':[[1,2,2]]}],
              'test':[{'input':[[0,0,0,0,0],[0,0,0,0,0],[0,1,1,0,2]]}]}
        self.assertIn([[[1,1,2]]],[out for _,out in rules(task)])


if __name__=='__main__':
    unittest.main()
