"""Mine reusable grid programs and test transfer under a fixed program budget."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
from itertools import product
import json
import random
from pathlib import Path

import numpy as np
from scipy import ndimage


def crop(grid, mask):
    y, x = np.where(mask)
    if len(y):
        return grid[y.min():y.max()+1, x.min():x.max()+1]


def apply(grid, op):
    name, arg = op
    if name == 'orient':
        return np.rot90(np.fliplr(grid) if arg >= 4 else grid, arg % 4)
    if name == 'repeat':
        axis, count = arg
        return np.repeat(grid, count, axis=axis)
    if name == 'tile':
        return np.tile(grid, arg)
    if name == 'mirror':
        axis, hinge = arg
        indices = np.arange(grid.shape[axis]-1-hinge, -1, -1)
        return np.concatenate([grid, grid.take(indices, axis=axis)], axis=axis)
    if name == 'panel':
        axis, count, gap, index = arg
        size, remainder = divmod(grid.shape[axis] - (count-1)*gap, count)
        if size > 0 and remainder == 0:
            start = index*(size+gap)
            return grid.take(range(start,start+size), axis=axis)
        return None
    colors = np.bincount(grid.ravel(), minlength=10)
    background = colors.argmax()
    if name == 'crop':
        return crop(grid, grid!=background if arg==0 else grid==arg)
    if name == 'interior':
        if min(grid.shape) > 2:
            return grid[1:-1,1:-1]
        return None
    if name == 'compress':
        rows = grid[np.r_[True,np.any(grid[1:]!=grid[:-1],axis=1)]]
        return rows[:,np.r_[True,np.any(rows[:,1:]!=rows[:,:-1],axis=0)]]
    if name == 'object':
        labels, count = ndimage.label(grid!=background)
        if count:
            sizes = np.bincount(labels.ravel())[1:]
            index = (sizes.argmax() if arg==0 else sizes.argmin())+1
            return crop(np.where(labels==index,grid,background), labels==index)
    return None


OPS = (
    [('orient', k) for k in range(1,8)]
    + [('crop', k) for k in range(10)]
    + [('object', k) for k in range(2)]
    + [('interior',0),('compress',0)]
    + [('repeat', p) for p in product(range(2),(2,3))]
    + [('tile', p) for p in ((2,1),(1,2),(2,2),(3,3))]
    + [('mirror', p) for p in product(range(2),range(2))]
    + [('panel',(axis,count,gap,i)) for axis,count,gap in product(range(2),(2,3),range(2))
       for i in range(count)]
)
PROGRAMS = [(), *((i,) for i in range(len(OPS))), *product(range(len(OPS)),repeat=2)]


def encode(grid):
    colors = np.argsort(-np.bincount(grid.ravel(), minlength=10), kind='stable')
    return np.argsort(colors)[grid], colors


def search(task, programs, collect=False):
    if any('output' in e for e in task['test']):
        raise ValueError('Expected unlabeled queries')
    demos = task['train']
    # Test cheaper demonstrations first so most programs fail after one grid.
    demos = sorted(demos,key=lambda e:np.size(e['input']))
    raw = [np.array(e['input']) for e in demos+task['test']]
    targets = [np.array(e['output']) for e in demos]
    n = len(demos)
    frames = [encode(g) for g in raw]
    labels_by_mode = {
        False: targets,
        True: [np.argsort(frame[1])[target] for frame,target in zip(frames,targets)],
    }
    caches = [{(False,()):g,(True,()):frame[0]} for g,frame in zip(raw,frames)]

    def run(index, canonical, program):
        bank = caches[index]
        key = canonical,program
        if key in bank:
            return bank[key]
        parent = run(index,canonical,program[:-1])
        value = apply(parent,OPS[program[-1]]) if parent is not None else None
        if value is not None and max(value.shape)>30:
            value = None
        if len(program)==1:
            bank[key] = value
        return value

    for canonical,program in programs:
        lookup = np.full(10,-1)
        for i,target in enumerate(targets):
            grid = run(i,canonical,program)
            if grid is None or grid.shape != target.shape:
                break
            labels = labels_by_mode[canonical][i]
            updates = lookup.copy()
            updates[grid.ravel()] = labels.ravel()
            if np.any((lookup>=0)&(lookup!=updates)) or np.any(updates[grid]!=labels):
                break
            lookup = updates
        else:
            if collect:
                yield canonical,program
                continue
            outputs = []
            for i in range(n,len(raw)):
                grid = run(i,canonical,program)
                if grid is None or np.any(lookup[grid]<0):
                    break
                decoded = lookup[grid]
                outputs.append((frames[i][1][decoded] if canonical else decoded).tolist())
            if len(outputs)==len(raw)-n:
                yield outputs


BASELINE = [(canonical,p) for p in PROGRAMS for canonical in (True,False)]
BUDGET = 384


def shuffled_order(seed):
    """Keep shorter programs first and randomize ties in the grammar order."""
    rng = random.Random(seed)
    ordered = []
    for length in range(3):
        group = [item for item in BASELINE if len(item[1]) == length]
        rng.shuffle(group)
        ordered.extend(group)
    return ordered[:BUDGET]


CONTROLS = {f'shuffled_{seed}': shuffled_order(seed) for seed in (0, 1, 2)}


def read(path):
    return json.loads(path.read_text())


def blind(task):
    return dict(train=task['train'],test=[{'input':e['input']} for e in task['test']])


def mine(path):
    task=read(path)
    # The library receives each source query as one more labeled demonstration.
    task=dict(train=task['train']+task['test'],test=[])
    found=list(search(task,BASELINE,True))
    shortest=min((len(p) for _,p in found),default=-1)
    return [p for p in found if len(p[1])==shortest][:12]


def order(model):
    learned=[(canonical,tuple(program)) for canonical,program,_ in model]
    return list(dict.fromkeys(learned+BASELINE))[:BUDGET]


def compare(job):
    path, learned = job
    task=read(path)
    truth=[e['output'] for e in task['test']]
    result={'id':path.stem}
    variants = {'baseline': BASELINE[:BUDGET], **CONTROLS, 'learned': learned}
    for name,programs in variants.items():
        guesses=[]
        for grids in search(blind(task),programs):
            if grids not in guesses:
                guesses.append(grids)
        result[name]=truth in guesses[:2]
        result[name+'_coverage']=truth in guesses
    return result


def ranked_paths(paths):
    return sorted(paths,key=lambda p:hashlib.sha256(('rc16:'+p.stem).encode()).digest())


def split(paths):
    ranked=ranked_paths(paths)
    return ranked[100:], ranked[:100]


def model_from(examples):
    counts=Counter(p for found in examples for p in found)
    return [[*p,count] for p,count in counts.most_common() if count>=2]


def report(rows, path):
    hits = {key:[row['id'] for row in rows if row[key]] for key in rows[0] if key!='id'}
    summary = {key:len(ids) for key, ids in hits.items()}
    result = dict(tasks=len(rows), budget=BUDGET, summary=summary, hits=hits)
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(summary,flush=True)


def predict_job(job):
    path, programs, folder = job
    task=read(path)
    rows=[{} for _ in task['test']]
    n=len(task['train'])
    for grids in search(task,programs):
        for row,grid in zip(rows,grids):
            row.setdefault(json.dumps(grid),[n-.01,n,grid])
    (folder/path.name).write_text(json.dumps([dict(status='ok',candidates=list(row.values())) for row in rows]))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('train','evaluate','cross-validate','predict'))
    parser.add_argument('data',type=Path)
    parser.add_argument('model',type=Path)
    parser.add_argument('output',type=Path,nargs='?')
    parser.add_argument('--holdout',action='store_true')
    args=parser.parse_args()
    paths=sorted(args.data.glob('*.json'))
    if not paths:
        parser.error('No task files found')
    if (args.mode=='train' or args.holdout) and len(paths)<=100:
        parser.error('The source split requires more than 100 tasks')
    if args.mode!='train' and args.output is None:
        parser.error('Output path is required for evaluation and prediction')
    with ProcessPoolExecutor(4) as workers:
        if args.mode=='train':
            source,_=split(paths)
            model=model_from(workers.map(mine,source))
            args.model.write_text(json.dumps(model,separators=(',',':'))+'\n')
        elif args.mode=='evaluate':
            if args.holdout:
                _,paths=split(paths)
            learned=order(read(args.model))
            rows=list(workers.map(compare,[(p,learned) for p in paths]))
            report(rows,args.output)
        elif args.mode=='cross-validate':
            ranked=ranked_paths(paths)
            if len(ranked)!=400:
                parser.error('Cross-validation expects the 400 ARC-1 training tasks')
            mined=list(workers.map(mine,ranked))
            rows=[]
            for start in range(0,len(ranked),100):
                end=start+100
                learned=order(model_from(mined[:start]+mined[end:]))
                rows.extend(workers.map(compare,[(p,learned) for p in ranked[start:end]]))
            report(rows,args.output)
        else:
            args.output.mkdir(parents=True,exist_ok=True)
            programs=order(read(args.model))
            list(workers.map(predict_job,[(p,programs,args.output) for p in paths]))
