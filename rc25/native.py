"""Run native synthesis and repair traced near-miss programs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import repair

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'rc12'))
from importlib.util import module_from_spec, spec_from_file_location
spec=spec_from_file_location('rc12_stdin',ROOT/'rc12'/'native.py')
codec=module_from_spec(spec)
spec.loader.exec_module(codec)


def parse(line,n):
    _,score,*encoded=line.split()
    values=list(map(int,encoded))
    arrays=[]
    start=0
    for _ in range(n+1):
        if len(values)<start+2:
            return
        h,w=values[start:start+2]
        start+=2
        size=h*w
        if min(h,w)<1 or max(h,w)>30 or start+size>len(values):
            return
        pixels=values[start:start+size]
        if not all(0<=v<=9 for v in pixels):
            return
        arrays.append(np.array(pixels,dtype=np.int16).reshape(h,w))
        start+=size
    if start==len(values):
        return float(score),arrays


def search(task,binary):
    codec.blind(task)
    size=len(task['train'])
    inputs=[np.array(d['input']) for d in task['train']]
    expected=[np.array(d['output']) for d in task['train']]
    header='\n'.join([str(size),*(codec.encode(d[k]) for d in task['train'] for k in ('input','output'))])
    for q in task['test']:
        query=np.array(q['input'])
        stdin=header+'\n'+codec.encode(q['input'])+'\n'
        direct,repaired={},{}
        statuses=[]
        for mode,depth,timeout in ((0,3,9),(1,2,4)):
            cmd=['prlimit','--as=1258291200','--core=0','--',binary,str(depth),str(mode)]
            try:
                run=subprocess.run(cmd,input=stdin,text=True,capture_output=True,timeout=timeout)
            except subprocess.TimeoutExpired:
                statuses.append('timeout')
                continue
            if run.returncode!=0:
                statuses.append(f'error:{run.returncode}')
                continue
            statuses.append('ok')
            for line in run.stdout.splitlines():
                if not line.startswith('TRACE '):
                    continue
                record=parse(line,size)
                if record is None:
                    continue
                score,grids=record
                query_output=grids[-1].tolist()
                key=json.dumps(query_output,separators=(',',':'))
                if key not in direct or score>direct[key][0]:
                    direct[key]=[score,size,query_output]
                for cost,result,rule,fit in repair.fit(inputs,expected,grids[:-1],query,grids[-1]):
                    answer=result.tolist()
                    key=json.dumps(answer,separators=(',',':'))
                    existing=repaired.get(key)
                    strength=size-min(.99,.015*cost+.2*(1-fit))
                    if existing is None or strength>existing[0]:
                        repaired[key]=[strength,size,answer]
        primary=sorted(direct.values(),key=lambda r:-r[0])[:24]
        corrections=sorted(repaired.values(),key=lambda r:-r[0])[:24]
        yield dict(status='ok' if 'ok' in statuses else statuses[0],
                   direct=primary,repair=corrections)


def process(job):
    path,binary,out=job
    task=json.loads(path.read_text())
    rows=list(search(task,binary))
    dest=out/path.name
    tmp=dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(rows,separators=(',',':')))
    tmp.replace(dest)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('blind','binary','output'):
        parser.add_argument(name,type=Path)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--resume',action='store_true')
    a=parser.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    jobs=[(p,str(a.binary.resolve()),a.output) for p in sorted(a.blind.glob('*.json'))
          if not (a.resume and (a.output/p.name).is_file())]
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(process,jobs))
