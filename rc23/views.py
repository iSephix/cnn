"""Compose verified detectors with reusable grid encoders and decoders."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

import bank

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/name) for name in ('rc18','rc13')]
from latent import scenarios
from search import expand, tasks


def experiments(task):
    for index,processed in enumerate(scenarios(task)):
        yield f'input-view-{index}',processed,None
    for (kind,rule),processed in tasks(task):
        if kind=='fold':
            yield 'factor:'+'/'.join(rule),processed,rule


def solve(task):
    rows=[{} for _ in task['test']]
    for name,processed,decoder in experiments(task):
        claims=bank.generate(processed,bank.DETECTORS,(0,))
        for row,values in zip(rows,claims):
            for item in values:
                g=np.array(item['grid'])
                if decoder is not None:
                    g=expand(g,decoder)
                if max(g.shape)>30:
                    continue
                sig=g.shape,g.astype(np.int8).tobytes()
                entry=row.setdefault(sig,dict(grid=g.tolist(),detectors=Counter(),views=[]))
                entry['detectors'].update(item['detectors'])
                entry['views'].append(name)
    return [sorted(list(row.values()),key=lambda item:(-len(item['detectors']),-len(item['views'])))[:24]
            for row in rows]


def process(job):
    path,out=job
    task=json.loads(path.read_text())
    if not task['train'] or not task['test'] or any('output' in q for q in task['test']):
        raise ValueError('Expected demonstrations and unlabeled queries')
    data=solve(task)
    dest=out/path.name
    bank.write_rows(dest,data)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('blind','out','bank_path'):
        p.add_argument(name,type=Path)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(a.workers,initializer=bank.init,initargs=(a.bank_path.resolve(),(0,))) as pool:
        list(pool.map(process,[(path,a.out) for path in sorted(a.blind.glob('*.json'))]))
