"""Run native synthesis with dimensions inferred from ARC demonstrations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import shape
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rc12'))
from native import blind, encode

def query_search(text, binary, depth, mode, size, limit):
    command=['prlimit','--as=1610612736','--core=0','--',binary,str(depth),str(mode),
             str(size[1]),str(size[0])]
    try:
        response=subprocess.run(command,input=text,text=True,capture_output=True,timeout=limit)
    except subprocess.TimeoutExpired:
        return 'timeout',[]
    if response.returncode:
        return f'error:{response.returncode}',[]
    candidates=[]
    for line in response.stdout.splitlines():
        if not line.startswith('GRID '):
            continue
        _,score,n,h,w,*tokens=line.split()
        h,w=int(h),int(w)
        values=list(map(int,tokens))
        if (h,w)==tuple(size) and len(values)==h*w and all(0<=v<=9 for v in values):
            candidates.append([float(score),int(n),[values[start:start+w] for start in range(0,h*w,w)]])
    return 'ok',candidates


def solve(task, binary):
    blind(task)
    demos=task['train']
    xs=[np.array(e['input']) for e in demos]
    ys=[np.array(e['output']) for e in demos]
    header='\n'.join([str(len(demos)),*(encode(e[k]) for e in demos for k in ('input','output'))])
    rows=[]
    for q in task['test']:
        query=q['input']
        sizes=shape.rank_sizes(xs,ys,np.array(query))
        text=header+'\n'+encode(query)+'\n'
        candidates={}
        status=[]
        for i,size in enumerate(sizes[:3]):
            depth=3 if i==0 else 2
            state,grids=query_search(text,binary,depth,0,size,7)
            status.append(state)
            for score,count,grid in grids:
                key=json.dumps(grid,separators=(',',':'))
                score-=.025*i
                if key not in candidates or score>candidates[key][0]:
                    candidates[key]=[score,count,grid]
        rows.append(dict(status='ok' if 'ok' in status else next(iter(status),'inapplicable'),
                         candidates=sorted(candidates.values(),key=lambda item:-item[0])[:20]))
    return rows


def process(job):
    path,output,binary=job
    rows=solve(json.loads(path.read_text()),binary)
    dest=output/path.name
    tmp=dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(rows,separators=(',',':')))
    tmp.replace(dest)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('blind','out','binary'):
        p.add_argument(name,type=Path)
    p.add_argument('--workers',type=int,default=7)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(process,[(path,a.out,str(a.binary.resolve())) for path in sorted(a.blind.glob('*.json'))]))
