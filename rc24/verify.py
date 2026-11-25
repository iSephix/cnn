"""Induce rules on demonstration subsets and verify them against every example."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import signal
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rc23'))
import bank


def prepare(task):
    if not task['train'] or not task['test'] or any('output' in q for q in task['test']):
        raise ValueError('Expected demonstrations and unlabeled queries')
    return ([(np.array(e['input']),np.array(e['output'])) for e in task['train']],
            [np.array(e['input']) for e in task['test']])


def heldout(demos,queries,detectors):
    result=[{} for _ in queries]
    n=len(demos)
    if n<2:
        return result
    for i in range(n):
        other=demos[:i]+demos[i+1:]
        ordered=[demos[i],*other]
        for detector in detectors:
            try:
                try:
                    signal.setitimer(signal.ITIMER_REAL,.75)
                    apply=detector([(a.copy(),b.copy()) for a,b in other])
                    if apply is None:
                        continue
                    # Reject on the held-out demonstration before the more expensive replay.
                    if not all(np.array_equal(bank.grid(apply(a.copy())),b) for a,b in ordered):
                        continue
                    candidate=[bank.grid(apply(q.copy())) for q in queries]
                    if any(grid is None for grid in candidate):
                        continue
                    name=f'{detector.__module__}.{detector.__name__}'
                    for row,grid in zip(result,candidate):
                        key=tuple(map(tuple,grid.tolist()))
                        record=row.setdefault(key,dict(detectors=Counter(),folds=set()))
                        record['detectors'][name]+=1
                        record['folds'].add(i)
                finally:
                    signal.setitimer(signal.ITIMER_REAL,0)
            except Exception:
                # A detector is an optional hypothesis; discard failures and timeouts.
                continue
    return result


def combine(original, folds):
    if len(original)!=len(folds):
        raise ValueError('Query count mismatch')
    rows=[]
    for entries,claims in zip(original,folds):
        previous={tuple(map(tuple,c['grid'])):c for c in entries}
        candidates=list(previous)
        candidates.extend(sorted((key for key in claims if key not in previous),
                                 key=lambda key:(-len(claims[key]['folds']),-len(claims[key]['detectors']),key))[:12])
        row=[]
        for key in candidates:
            old=previous.get(key,dict(grid=[list(line) for line in key],
                                      detectors={},orientations=[],identity_votes=0))
            support=claims.get(key,dict(detectors={},folds=set()))
            row.append(dict(old,folds=sorted(support['folds']),fold_detectors=dict(support['detectors']),
                            full_fit=key in previous))
        rows.append(row)
    return rows


def process(job):
    path,original,output=job
    task=json.loads(path.read_text())
    demos,queries=prepare(task)
    evidence=heldout(demos,queries,bank.DETECTORS)
    source=json.loads((original/path.name).read_text())
    bank.write_rows(output/path.name,combine(source,evidence))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('blind','original','output','detector_bank'):
        parser.add_argument(name,type=Path)
    parser.add_argument('--workers',type=int,default=7)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    paths=[p for p in sorted(args.blind.glob('*.json'))
           if not (args.resume and (args.output/p.name).is_file())]
    with ProcessPoolExecutor(args.workers,initializer=bank.init,
                             initargs=(args.detector_bank.resolve(),(0,))) as workers:
        list(workers.map(process,[(p,args.original,args.output) for p in paths]))
