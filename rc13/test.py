"""Check decoder inverses, color-plane transfer, and rejection of labeled queries."""
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import numpy as np
from layers import infer, one
from search import expand, reduce_grid

def test_decoders():
    grid = np.array([[3, 0, 2], [5, 1, 7]])
    for rule in product(('keep', 'reflect', 'hinge', 'repeat', 'scale'), repeat=2):
        assert np.array_equal(reduce_grid(expand(grid, rule), rule), grid)
    assert reduce_grid(grid, ('reflect', 'keep')) is None

def test_layers():
    def example(grid):
        return {'input': grid, 'output': [[3 if cell == 1 else 0 for cell in row] for row in grid]}
    query = example([[1, 0], [0, 1], [0, 0]])
    task = {'train': [example([[1, 0, 0], [0, 0, 1]]), example([[0, 1], [0, 0]])],
            'test': [{'input': query['input']}]}
    assert any(outputs == [query['output']] for _, outputs in infer(task))
    with TemporaryDirectory() as root:
        path = Path(root) / 'task.json'
        path.write_text(json.dumps(dict(task, test=[query])))
        try:
            one((path, Path(root)))
        except ValueError:
            pass
        else:
            raise AssertionError('Accepted a labeled query')

if __name__ == '__main__':
    test_decoders()
    test_layers()
