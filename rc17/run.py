"""Train, emit candidates, or compare transferred stencils at matched budget."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from collections import Counter

import motion
import charts
import stencils


def read(path):
    return json.loads(path.read_text())


def compare(job):
    path,patterns=job
    task=read(path)
    expected=[q['output'] for q in task['test']]
    blind=dict(train=task['train'],test=[{'input':q['input']} for q in task['test']])
    return [any(out==expected for _,out in stencils.rules(blind,bank))
            for bank in (stencils.controls(),stencils.neighborhood_control(),patterns)]


def predict(job):
    path,patterns,folder,family=job
    task=read(path)
    if not task['train'] or not task['test'] or any('output' in q for q in task['test']):
        raise ValueError('Expected demonstrations and unlabeled queries')
    rules = stencils.rules(task,patterns) if family=='stencil' else {'motion':motion.rules,'chart':charts.rules}[family](task)
    candidates=[{} for _ in task['test']]
    n=len(task['train'])
    for cost,outs in sorted(rules,key=lambda rule:rule[0]):
        for rows,grid in zip(candidates,outs):
            rows.setdefault(json.dumps(grid),[n-.01*cost,n,grid])
    (folder/path.name).write_text(json.dumps([dict(status='ok',candidates=list(rows.values())) for rows in candidates]))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('train','compare','predict'))
    parser.add_argument('data',type=Path)
    parser.add_argument('model',type=Path)
    parser.add_argument('out',type=Path,nargs='?')
    parser.add_argument('--family',choices=('stencil','motion','chart'),default='stencil')
    parser.add_argument('--holdout',action='store_true')
    args=parser.parse_args()
    if args.mode!='train' and args.out is None:
        parser.error('Output path is required')
    paths=sorted(args.data.glob('*.json'))
    if not paths:
        parser.error('No tasks found')
    with ProcessPoolExecutor(4) as workers:
        if args.mode=='train':
            if args.holdout:
                if len(paths)<=100:
                    parser.error('Holdout training needs more than 100 tasks')
                paths=sorted(paths,key=lambda p:hashlib.sha256(('rc16:'+p.stem).encode()).digest())[100:]
            source=[read(p) for p in paths]
            counts=Counter(rule for rules in workers.map(stencils.mine,source) for rule in rules)
            patterns=[rule for rule,_ in sorted(counts.items(),key=lambda row:(-row[1],row[0]))[:32]]
            args.model.write_text(json.dumps(patterns,separators=(',',':'))+'\n')
        elif args.mode=='compare':
            if args.holdout:
                paths=sorted(paths,key=lambda p:hashlib.sha256(('rc16:'+p.stem).encode()).digest())[:100]
            patterns=read(args.model)
            results=list(workers.map(compare,[(p,patterns) for p in paths]))
            result={name:[p.stem for p,row in zip(paths,results) if row[i]]
                    for i,name in enumerate(('directional','neighborhood','learned'))}
            args.out.write_text(json.dumps(dict(tasks=len(paths),pattern_cap=32,learned_patterns=len(patterns),**result),indent=2)+'\n')
        else:
            args.out.mkdir(parents=True,exist_ok=True)
            patterns=read(args.model)
            list(workers.map(predict,[(p,patterns,args.out,args.family) for p in paths]))
