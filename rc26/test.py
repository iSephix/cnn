"""Check composition, inverse rendering, abstraction extraction, and query isolation."""
import unittest

import numpy as np

from programs import canonical, execute, goals, library, restore, simplify
from search import synthesize
from predict import merge


class Programs(unittest.TestCase):
    def test_inverse_goals(self):
        target = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
        for operation, (latent,) in goals([target]):
            np.testing.assert_array_equal(restore(latent, operation), target)

    def test_composition_with_crop_and_compression(self):
        grid = [[0, 0, 0, 0], [0, 2, 2, 3], [0, 2, 2, 3]]
        result = execute(grid, [['crop'], ['view', 'compress']])
        self.assertEqual(result.tolist(), [[2, 3]])

    def test_shared_palette_constraint(self):
        a = [np.array([[1, 2]]), np.array([[2, 1]])]
        b = [np.array([[7, 8]]), np.array([[8, 7]])]
        self.assertEqual(canonical(a), canonical(b))
        b[1][0, 0] = 7
        self.assertNotEqual(canonical(a), canonical(b))

    def test_fragments_need_multiple_source_programs(self):
        program = [['crop'], ['view', 'compress']]
        self.assertEqual(library([program]), [])
        result = library([program, program])
        self.assertEqual(result, [{'program': program, 'support': 2}])
        self.assertEqual(library([program, program], groups=['same', 'same']), [])
        self.assertEqual(simplify([['orient', 1], ['orient', 3]]), [])

    def test_synthesize_composed_rule(self):
        first = [[0, 0, 0, 0], [0, 2, 2, 3], [0, 2, 2, 3]]
        second = [[0, 0, 0, 0], [0, 3, 3, 2], [0, 3, 3, 2]]
        question = [[0, 0, 0, 0], [0, 2, 3, 3], [0, 2, 3, 3]]
        task = {'train': [{'input': first, 'output': [[2, 3]]},
                          {'input': second, 'output': [[3, 2]]}],
                'test': [{'input': question}]}
        result = synthesize(task, budget=768)
        self.assertTrue(any(row['grids'] == [[[2, 3]]] for row in result['candidates']))
        task['test'][0]['output'] = [[2, 3]]
        with self.assertRaisesRegex(ValueError, 'unlabeled'):
            synthesize(task)

    def test_macro_replay_cache(self):
        grid = np.array([[0, 1, 2], [0, 1, 2]])
        program = [['crop'], ['view', 'compress']]
        expected = execute(grid, program)
        cache = {}
        np.testing.assert_array_equal(execute(grid, program, cache), expected)
        np.testing.assert_array_equal(execute(grid, program, cache), expected)
        self.assertGreater(len(cache), 0)

    def test_prediction_merge_and_query_count(self):
        task = {'test': [{'input': [[0]]}]}
        baseline = [{'attempt_1': [[1]], 'attempt_2': [[2]]}]
        candidates = {'candidates': [{'grids': [[[3]]]}]}
        self.assertEqual(merge(task, candidates, baseline),
                         [{'attempt_1': [[1]], 'attempt_2': [[3]]}])
        self.assertEqual(baseline[0]['attempt_2'], [[2]])
        with self.assertRaisesRegex(ValueError, 'query count'):
            merge(task, candidates, [])


if __name__ == '__main__':
    unittest.main()
