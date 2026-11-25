"""Regression checks for the new candidate generators and arbitration."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from emit import process
from layout import mosaics, spread, sorted_latent
from repair import fit
from arbitrate import key, unanimous, verified_trace
from structured import l_path, structure, topology


class Concepts(unittest.TestCase):
    def test_trace_repair(self):
        source = np.array([[0, 8, 1], [0, 1, 0], [0, 0, 0]])
        target = np.where(source == 8, 2, source)
        query = np.rot90(source)
        expected = np.rot90(target)
        results = fit([source], [target], [source], query, query)
        self.assertTrue(any(np.array_equal(g, expected) for _, g, _, _ in results))

    def test_marker_path_with_new_coordinates(self):
        def example(start, end):
            grid = np.zeros((6, 6), dtype=int)
            grid[start], grid[end] = 1, 3
            route = l_path(grid, start, end, horizontal_first=True)
            return {'input': grid.tolist(),
                    'output': np.where(route & (grid == 0), 2, grid).tolist()}

        a = example((1, 1), (4, 5))
        b = example((4, 2), (1, 4))
        query = example((2, 1), (5, 3))
        task = {'train': [a, b], 'test': [{'input': query['input']}]}
        results = [grid for name, _, (grid,) in structure(task) if name == 'marker_paths']
        self.assertIn(query['output'], results)
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'task.json'
            task['test'] = [query]
            path.write_text(json.dumps(task))
            with self.assertRaisesRegex(ValueError, 'unlabeled'):
                process((path, Path(tmp), structure))

    def test_topology_commutes_with_reflection_and_rotation(self):
        grid = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 0]])
        fields = topology(grid, 1)
        for transform in (np.fliplr, np.rot90, np.transpose):
            changed = topology(transform(grid), 1)
            for name, values in fields.items():
                np.testing.assert_array_equal(changed[name], transform(values))

    def test_mosaic_with_scene_relative_separator(self):
        def scene(sep, inks):
            grid = np.zeros((7, 7), dtype=int)
            grid[3, :] = grid[:, 3] = sep
            grid[1, 1], grid[2, 5], grid[4, 2], grid[5, 6] = inks
            return grid
        a = scene(5, [1, 2, 3, 4])
        b = scene(8, [4, 3, 2, 1])
        query = scene(9, [6, 7, 4, 2])
        candidates = mosaics([a, b, query], [np.array([[1, 2], [3, 4]]),
                                            np.array([[4, 3], [2, 1]])], 2)
        self.assertTrue(any(g.tolist() == [[6, 7], [4, 2]] for _, (g,) in candidates))

    def test_spread_learns_from_input_transitions(self):
        frames = [np.array([[1]*occupied+[2]*(4-occupied)+[3]])
                  for occupied in (1, 2, 3)]
        self.assertTrue(any(g.tolist() == [[1, 1, 1, 1, 3]] for g in spread(frames)))

    def test_latent_sort_keeps_ties_stable(self):
        grid = np.array([[0, 3, 0], [1, 2, 0], [0, 0, 4], [1, 1, 1]])
        self.assertEqual(sorted_latent(grid, 0, 0, True).tolist(),
                         [grid[i].tolist() for i in (0, 2, 1, 3)])


class Selection(unittest.TestCase):
    def test_distinguish_exact_and_partial_traces(self):
        a, b, c = [[1]], [[2]], [[3]]
        direct = [[2.7, 3, a], [1.7, 3, b]]
        self.assertEqual(verified_trace({'direct': direct, 'repair': []}, key(c)), key(a))
        self.assertIsNone(verified_trace({'direct': direct, 'repair': []}, key(b)))
        matching = {'direct': direct[1:], 'repair': [[2.9, 3, b]]}
        self.assertEqual(verified_trace(matching, key(c)), key(b))
        conflicting = {'direct': direct[1:], 'repair': [[2.9, 3, a]]}
        self.assertIsNone(verified_trace(conflicting, key(c)))

    def test_require_symbolic_answer_for_every_query(self):
        candidate = {'grid': [[1]]}
        other = {'grid': [[2]]}
        self.assertEqual(unanimous([[candidate], [other]]), [key([[1]]), key([[2]])])
        self.assertIsNone(unanimous([[candidate], []]))
        self.assertIsNone(unanimous([[candidate, other]]))


if __name__ == '__main__':
    unittest.main()
