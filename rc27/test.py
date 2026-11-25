"""Regression checks for relational actions, constraint solving, and arbitration."""
import unittest

import numpy as np

from arbitrate import select
from concepts import broadcast, staircase, transfer
from constraints import exact_cover, legend, projected_pins
from layout import fits_cavity, object_grid
from topological import fit as topology_rules


class Concepts(unittest.TestCase):
    def test_scaled_color_template(self):
        grid = np.zeros((12,12),int)
        template = np.array([[1,2],[1,0]])
        grid[1:3,1:3] = template
        enlarged = template.repeat(2,0).repeat(2,1)
        grid[6:10,6:10] = np.where(enlarged != 0,8,0)
        expected = grid.copy()
        expected[6:10,6:10] = enlarged
        np.testing.assert_array_equal(transfer(grid,'zero','smallest',0),expected)

    def test_connected_marker_broadcast_preserves_bridge(self):
        grid = np.zeros((6,12),int)
        grid[1:4,1:4] = grid[1:4,7:10] = 1
        grid[2,4:7] = 1
        grid[2,2] = 2
        expected = grid.copy()
        expected[2,8] = 2
        np.testing.assert_array_equal(broadcast(grid,'zero',1,1,1,3),expected)

    def test_staircase_uses_previous_object_extent(self):
        grid = np.zeros((10,10),int)
        grid[7:10,0:2] = 1
        grid[8:10,4:6] = 2
        expected = np.zeros_like(grid)
        expected[:3,:2] = 1
        expected[2:4,1:3] = 2
        np.testing.assert_array_equal(staircase(grid,'zero',0,'column',False,-1,-1,'original',0),expected)

    def test_exact_cover_conserves_pieces_and_fixed_host(self):
        grid = np.zeros((12,12),int)
        grid[1:5,1:6] = 1
        grid[2:4,2:5] = 0
        grid[7:9,1:3] = 2
        grid[7:9,7] = 3
        solutions = exact_cover(grid,'zero',(),False)
        self.assertTrue(solutions)
        for board in solutions:
            np.testing.assert_array_equal(board[grid == 1],grid[grid == 1])
            self.assertEqual(np.count_nonzero(board == 2),4)
            self.assertEqual(np.count_nonzero(board == 3),2)
            self.assertTrue(np.all(board[2:4,2:5] != 0))
            self.assertTrue(np.all(board[7:] == 0))

    def test_legend_matches_relative_scene_positions(self):
        grid = np.zeros((16,16),int)
        grid[1:6,1:6] = 5
        grid[2,2],grid[4,4] = 2,3
        grid[8:10,8:10] = grid[12:14,12:14] = 1
        expected = grid.copy()
        expected[8:10,8:10],expected[12:14,12:14] = 2,3
        np.testing.assert_array_equal(legend(grid,'zero',0,'square'),expected)

    def test_external_pins_define_object_relative_rectangle(self):
        grid = np.zeros((12,12),int)
        grid[3:9,3:9] = 5
        grid[3,3] = grid[6,2] = grid[2,5] = 3
        expected = grid.copy()
        expected[3:7,3:6] = 3
        expected[6,2] = expected[2,5] = 0
        np.testing.assert_array_equal(projected_pins(grid,'zero',1),expected)

    def test_cavity_accepts_an_overhanging_body(self):
        body = np.array([[0,1,1,0],[0,1,1,0],[1,1,1,1]],bool)
        cavity = np.ones((2,2),bool)
        self.assertTrue(fits_cavity(body,cavity,False,True))
        self.assertFalse(fits_cavity(body,cavity,False,False))

    def test_grid_layout_aligns_shifted_icons(self):
        grid = np.zeros((15,15),int)
        for row,col,color in [(1,1,1),(2,10,2),(10,2,3),(9,11,4)]:
            grid[row:row+2,col:col+2] = color
        expected = np.array([[1,1,0,2,2],[1,1,0,2,2],[0,0,0,0,0],[3,3,0,4,4],[3,3,0,4,4]])
        np.testing.assert_array_equal(object_grid(grid,'zero',2,0,1,0),expected)

    def test_topology_rule_transfers_across_foreground_colors(self):
        def example(height,width,color):
            grid = np.zeros((height+4,width+4),int)
            grid[2:2+height,2:2+width] = color
            target = grid.copy()
            for r in (2,height+1):
                for c in (2,width+1):
                    target[r,c] = 4
            return dict(input=grid.tolist(),output=target.tolist())
        a,b,q = example(3,3,3),example(4,4,8),example(5,4,6)
        task = dict(train=[a,b],test=[dict(input=q['input'])])
        rows = list(topology_rules(task))
        self.assertTrue(any(row['grids'] == [q['output']] and row['folds'] == 2 for row in rows))


class Arbitration(unittest.TestCase):
    def test_requires_fold_verification_and_keeps_backup(self):
        task = {'test':[{'input':[[0]]}]}
        baseline = [{'attempt_1':[[1]],'attempt_2':[[2]]}]
        candidate = dict(family='local',folds=1,demonstrations=2,grids=[[[3]]])
        sources = [dict(candidates=[candidate])]
        self.assertEqual(select(task,baseline,sources,promote=True),baseline)
        candidate['folds'] = 2
        self.assertEqual(select(task,baseline,sources,promote=True),
                         [{'attempt_1':[[3]],'attempt_2':[[1]]}])
        sources.append(dict(candidates=[dict(family='pins',grids=[[[4]]])]))
        self.assertEqual(select(task,baseline,sources,promote=True),baseline)

    def test_rejects_query_labels(self):
        with self.assertRaisesRegex(ValueError,'unlabeled'):
            select({'test':[{'input':[[0]],'output':[[0]]}]},[],[])


if __name__ == '__main__':
    unittest.main()
