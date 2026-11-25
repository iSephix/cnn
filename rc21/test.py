import unittest
import numpy as np

import bind
import markers
import place
import regions
import repair
import route


class ConceptTests(unittest.TestCase):
    def test_repair_needs_two_generators(self):
        grid=np.array([[1,9,9],[2,3,2],[9,4,1]])
        options=repair.variants(grid,9)
        target=[[1,4,1],[2,3,2],[1,4,1]]
        self.assertTrue(any(g.tolist()==target and len(key[0])==2 for key,g in options.items()))

    def test_color_is_bound_to_neighbor_object(self):
        grid=np.array([[0,0,0,0,0],[0,1,1,0,2],[0,0,0,0,0]])
        parts=bind.objects(grid,True,1)
        result=bind.canvas(grid,parts,1,'marker','row','nearest',False)
        self.assertEqual(result[1].tolist(),[0,2,2,0,2])

    def test_place_mover_against_anchor(self):
        grid=np.array([[0,0,0,0,0,0],[1,1,0,0,0,2],[0,0,0,0,0,0]])
        parts=bind.objects(grid,True,1)
        result=place.transform(grid,parts,1,2,'axial','center','move')
        self.assertEqual(result[1].tolist(),[0,0,0,1,1,2])

    def test_marker_receives_template(self):
        grid=np.array([[1,1,0,0,0],[1,0,0,2,0],[0,0,0,0,0]])
        result=markers.render(grid,markers.scene(grid,1),'largest','corner','marker',True,False)
        self.assertEqual(result.tolist(),[[1,1,0,0,0],[1,0,0,2,2],[0,0,0,2,0]])

    def test_closed_region_uses_its_border_color(self):
        grid=np.array([[0,0,0,0,0],[0,4,4,4,0],[0,4,0,4,0],[0,4,4,4,0],[0,0,0,0,0]])
        enclosed=[p for _,a,p in regions.describe(grid,0,1) if a['enclosed']]
        self.assertEqual([p['boundary majority'] for p in enclosed],[4])

    def test_tree_splits_on_color_correctness(self):
        x=np.array([[0],[1],[0],[1]])
        refs=np.array([[4,9],[4,9],[5,9],[4,8]])
        labels=np.array([4,9,5,8])
        tree=route.build(x,refs==labels[:,None],np.arange(2),2)
        np.testing.assert_array_equal(route.evaluate(tree,refs,x),labels)


if __name__=='__main__':
    unittest.main()
