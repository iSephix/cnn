"""Audit candidate coverage after blind predictions have been written."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def evaluate(labels, folders, prior=None):
    paths=sorted(labels.glob('*.json'))
    baseline=read(prior) if prior else {}
    metrics={folder.name:{'active':0,'coverage':0,'unanimous':0,'unanimous_correct':0} for folder in folders}
    covered=[]
    baseline_solved=[]
    for path in paths:
        target=[row['output'] for row in read(path)['test']]
        members=[set() for _ in target]
        for folder in folders:
            stats=metrics[folder.name]
            rows=read(folder/path.name)
            if len(target)!=len(rows):
                raise ValueError(f'Query count mismatch for {path.stem} in {folder}')
            hits=[]
            unique=all(len(r['candidates'])==1 for r in rows)
            stats['active']+=all(row['candidates'] for row in rows)
            for expected,row,union in zip(target,rows,members):
                choices={json.dumps(c[-1]) for c in row['candidates']}
                union.update(choices)
                hits.append(json.dumps(expected) in choices)
            stats['coverage']+=all(hits)
            stats['unanimous']+=unique
            stats['unanimous_correct']+=unique and all(hits)
        if all(json.dumps(g) in options for g,options in zip(target,members)):
            covered.append(path.stem)
        if prior is not None:
            attempts=baseline[path.stem]
            if len(attempts)!=len(target):
                raise ValueError(f'Baseline query count mismatch for {path.stem}')
            if all(g in row.values() for g,row in zip(target,attempts)):
                baseline_solved.append(path.stem)
    report=dict(tasks=len(paths),families=metrics,covered=covered)
    if prior is not None:
        report.update(baseline_solved=len(baseline_solved),
                      additional_candidates=sorted(set(covered)-set(baseline_solved)))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('labels',type=Path)
    p.add_argument('candidates',nargs='+',type=Path)
    p.add_argument('--baseline',type=Path)
    args=p.parse_args()
    print(json.dumps(evaluate(args.labels,args.candidates,args.baseline),indent=2))
