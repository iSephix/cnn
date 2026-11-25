"""Recover proposals that become expressible after a reversible color relabeling."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from verify import bank, prepare


def recodings(demos):
    counts=np.bincount(np.concatenate([a.ravel() for a,_ in demos]),minlength=10)
    order=np.argsort(-counts,kind='stable')
    yield np.argsort(order)
    # Keep the inferred background at zero while testing the opposite foreground order.
    active=order[1:][counts[order[1:]]>0]
    inactive=order[1:][counts[order[1:]]==0]
    yield np.argsort(np.r_[order[:1],active[::-1],inactive])


def variants(task,demos):
    seen={tuple(range(10))}
    for mapping in recodings(demos):
        key=tuple(mapping)
        if key in seen:
            continue
        seen.add(key)
        changed=dict(train=[dict(input=mapping[a].tolist(),output=mapping[b].tolist()) for a,b in demos],
                     test=[dict(input=mapping[np.array(q['input'])].tolist()) for q in task['test']])
        yield changed,np.argsort(mapping)


def process(job):
    path,output=job
    task=json.loads(path.read_text())
    demos,queries=prepare(task)
    rows=[{} for _ in queries]
    for transformed,inverse in variants(task,demos):
        for row,candidates in zip(rows,bank.generate(transformed,bank.DETECTORS,(0,))):
            for item in candidates:
                grid=inverse[np.asarray(item['grid'])]
                key=tuple(map(tuple,grid.tolist()))
                entry=row.setdefault(key,dict(grid=grid.tolist(),palette_fits=0,palette_detectors={}))
                entry['palette_fits']+=1
                for name,votes in item['detectors'].items():
                    entry['palette_detectors'][name]=entry['palette_detectors'].get(name,0)+votes
    bank.write_rows(output/path.name,[sorted(r.values(),key=lambda c:(-c['palette_fits'],
                                                                     -len(c['palette_detectors'])))[:12] for r in rows])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('blind','out','detector_bank'):
        p.add_argument(name,type=Path)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    a.out.mkdir(exist_ok=True,parents=True)
    paths=sorted(a.blind.glob('*.json'))
    with ProcessPoolExecutor(a.workers,initializer=bank.init,initargs=(a.detector_bank.resolve(),(0,))) as pool:
        list(pool.map(process,[(path,a.out) for path in paths]))
