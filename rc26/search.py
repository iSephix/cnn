"""Bounded compositional synthesis with backward goals and reusable program fragments."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from programs import canonical, execute, goals, library, recolor, restore, signature, valid, variants


def priority(program, macros):
    path = tuple(tuple(op) for op in program)
    return max((item['support'] for item in macros
                if tuple(tuple(op) for op in item['program'][:len(path)]) == path), default=0)


def synthesize(task, model=None, budget=768, beam=32, depth=3, backward_search=True):
    if not task['train'] or not task['test'] or any('output' in q for q in task['test']):
        raise ValueError('Expected demonstrations and unlabeled queries')
    if budget < 1 or beam < 1 or depth < 1:
        raise ValueError('Search budget, beam width, and depth must be positive')
    n = len(task['train'])
    inputs = [np.asarray(e['input']) for e in task['train']+task['test']]
    targets = [np.asarray(e['output']) for e in task['train']]
    if any(not valid(g) for g in inputs+targets):
        raise ValueError('Expected rectangular grids of colors 0 through 9, at most 30 by 30')
    inputs = [g.astype(np.int16) for g in inputs]
    targets = [g.astype(np.int16) for g in targets]
    macros = (model or {}).get('macros', [])[:16]
    backward = {}
    target_shapes = []
    for operation, latent in goals(targets):
        if operation is not None and not backward_search:
            continue
        backward.setdefault(canonical(latent), []).append((operation, latent))
        target_shapes.append(tuple(g.shape for g in latent))
    frontier = [(inputs, [])]
    seen = {signature(inputs)}
    candidates = {}
    solutions = []
    tested = 0
    for level in range(depth+1):
        following = []
        level_limit = min(budget, tested+max(1, budget//max(1, depth)))
        for grids, program in frontier:
            for operation, latent in backward.get(canonical(grids[:n]), []):
                table = recolor(grids[:n], latent)
                if table is None:
                    continue
                predictions = [restore(table[g], operation) for g in grids[n:]]
                if any(g is None for g in predictions):
                    continue
                key = signature(predictions)
                cost = len(program)+int(operation is not None)+int(np.any(table != np.arange(10)))
                record = dict(program=program, render=operation, palette=table.tolist(),
                              cost=cost, grids=[g.tolist() for g in predictions])
                solutions.append(dict(program=program, render=operation, cost=cost))
                if key not in candidates or cost < candidates[key]['cost']:
                    candidates[key] = record
            if level == depth or tested >= level_limit:
                continue
            frames = []
            for grid in grids:
                frame = {}
                for name, value in variants(grid):
                    if valid(value):
                        frame[name] = value
                frames.append(frame)
            common = set.intersection(*(set(frame) for frame in frames))
            operations = sorted(common, key=lambda op: (-priority(program+[list(op)], macros), repr(op)))
            expansions = []
            caches = [{(op,): value for op, value in frame.items()} for frame in frames]
            for item in macros:
                values = [execute(grid, item['program'], cache) for grid, cache in zip(grids, caches)]
                if all(value is not None for value in values):
                    expansions.append((item['program'], values))
            expansions.extend(([list(op)], [frame[op] for frame in frames]) for op in operations)
            for steps, values in expansions:
                if tested >= level_limit:
                    break
                key = signature(values)
                if key in seen:
                    continue
                tested += 1
                seen.add(key)
                path = program+steps
                shapes = tuple(g.shape for g in values[:n])
                distance = min(sum(abs(a-b) for s, t in zip(shapes, target)
                                   for a, b in zip(s, t)) for target in target_shapes)
                # Exact output dimensions guide a bounded search over intermediate representations.
                following.append(((distance, -priority(path, macros), len(path)), values, path))
        if not following:
            break
        following.sort(key=lambda row: row[0])
        frontier = [(grids, program) for _, grids, program in following[:beam]]
    records = sorted(candidates.values(), key=lambda rec: (rec['cost'], repr(rec['program'])))[:24]
    solutions.sort(key=lambda row: (row['cost'], repr(row['program']), repr(row['render'])))
    return dict(candidates=records, solutions=solutions[:64], states=tested, depth=depth,
                budget=budget, backward_search=backward_search)


def process(job):
    path, destination, model, budget, backward_search = job
    result = synthesize(json.loads(path.read_text()), model, budget=budget, backward_search=backward_search)
    (destination/path.name).write_text(json.dumps(result, separators=(',', ':'))+'\n')


def training_result(job):
    path, budget = job
    task = json.loads(path.read_text())
    # Queries and their answers have no role in abstraction acquisition.
    blind = dict(train=task['train'], test=[dict(input=task['train'][0]['input'])])
    return path.stem, synthesize(blind, budget=budget)


def train(tasks, out, budget, workers):
    programs = []
    groups = []
    solved = []
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(training_result, [(path, budget) for path in sorted(tasks.glob('*.json'))]))
    for task_id, result in results:
        if result['candidates']:
            solved.append(task_id)
            for candidate in result['solutions']:
                program = candidate['program'][:]
                if candidate['render'] is not None:
                    program.append(list(candidate['render']))
                programs.append(program)
                groups.append(task_id)
    out.write_text(json.dumps(dict(macros=library(programs, groups)[:16], source_tasks=solved,
                                    acquisition_tasks=[p.stem for p in sorted(tasks.glob('*.json'))],
                                    budget=budget, beam=32, depth=3,
                                    source_split='official ARC1 training, demonstrations only'),
                              indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('train', 'predict'))
    parser.add_argument('tasks', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--budget', type=int, default=768)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--forward-only', action='store_true')
    args = parser.parse_args()
    if args.mode == 'train':
        train(args.tasks, args.output, args.budget, args.workers)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        model = json.loads(args.model.read_text()) if args.model else None
        with ProcessPoolExecutor(args.workers) as pool:
            list(pool.map(process, [(p, args.output, model, args.budget, not args.forward_only)
                                    for p in sorted(args.tasks.glob('*.json'))]))
