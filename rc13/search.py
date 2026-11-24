"""Fit lossless output factorizations and solve the resulting smaller puzzles."""
from concurrent.futures import ThreadPoolExecutor
import argparse,json,math,sys
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rc12'))
from native import blind,search


def expand(g,rule):
    a=np.asarray(g)
    for axis,op in enumerate(rule):
        if op=='reflect':
            a=np.concatenate((a,np.flip(a,axis)),axis)
        elif op=='hinge':
            reflected=np.flip(a,axis).take(range(1,a.shape[axis]),axis)
            a=np.concatenate((a,reflected),axis)
        elif op=='repeat':
            a=np.concatenate((a,a),axis)
        elif op=='scale':
            a=np.repeat(a,2,axis)
    return a


def reduce_grid(g,rule):
    a=np.asarray(g)
    for axis,op in enumerate(rule):
        if op=='keep':continue
        n=a.shape[axis]
        if n<2 or (n%2)!=(op=='hinge'):return
        size=(n+int(op=='hinge'))//2
        a=a.take(range(0,n,2) if op=='scale' else range(size),axis)
    if np.array_equal(expand(a,rule),g):return a


def tasks(t):
    import itertools
    rules=[]
    for rule in itertools.product(('keep','reflect','hinge','repeat','scale'),repeat=2):
        if rule==('keep','keep'):continue
        latent=[reduce_grid(e['output'],rule) for e in t['train']]
        if all(y is not None for y in latent):
            rules.append((sum(y.size for y in latent),rule,latent))
    for _,rule,latent in sorted(rules,key=lambda v:v[0])[:3]:
        yield ('fold',rule),dict(train=[dict(input=e['input'],output=y.tolist()) for e,y in zip(t['train'],latent)],test=t['test'])
    # Rotate both halves of the task. Depth-limited search has orientation-specific paths.
    def rot(g):return np.rot90(g).tolist()
    yield ('rotate',None),dict(train=[dict(input=rot(e['input']),output=rot(e['output'])) for e in t['train']],test=[dict(input=rot(e['input'])) for e in t['test']])


def solve(path,binary,work):
    t=json.loads(path.read_text())
    blind(t)
    records=[dict(status='ok',candidates=[]) for _ in t['test']]
    sub=work/path.name
    for (method,rule),data in tasks(t):
        sub.write_text(json.dumps(data))
        for row,res in zip(records,search(sub,binary,2,0,120)):
            if res['status']!='ok':row['status']=res['status']
            for score,n,g in res['candidates']:
                out=expand(g,rule) if method=='fold' else np.rot90(g,-1)
                if max(out.shape)<=30:
                    row['candidates'].append([score,n,out.tolist()])
    for row in records:
        unique={}
        for item in sorted(row['candidates'],key=lambda item:-item[0]):
            unique.setdefault(json.dumps(item[-1]),item)
        row['candidates']=list(unique.values())
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('blind','binary','out'):p.add_argument(k,type=Path)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    a.out.mkdir(exist_ok=True,parents=True)
    with TemporaryDirectory() as d,ThreadPoolExecutor(a.workers) as workers:
        def one(path):
            rows=solve(path,str(a.binary.resolve()),Path(d))
            (a.out/path.name).write_text(json.dumps(rows,separators=(',',':')))
        list(workers.map(one,sorted(a.blind.glob('*.json'))))
