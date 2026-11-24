"""Generate candidates using an isolated, answer-blind native search process."""
from concurrent.futures import ThreadPoolExecutor
import argparse,json,subprocess
from pathlib import Path


def blind(data):
    if not data['train'] or not data['test'] or any('output' in q for q in data['test']):
        raise ValueError('Expected demonstrations and unlabeled queries')


def encode(g):
    if not 1<=len(g)<=30 or not 1<=len(g[0])<=30 or not all(len(r)==len(g[0]) for r in g):
        raise ValueError('Invalid grid')
    return ' '.join(map(str,[len(g),len(g[0]),*(v for r in g for v in r)]))


def search(path,binary,depth,mode,timeout):
    t=json.loads(path.read_text())
    blind(t)
    header='\n'.join([str(len(t['train'])),*(encode(e[k]) for e in t['train'] for k in ('input','output'))])
    results=[]
    for q in t['test']:
        try:
            cmd=['prlimit','--as=1610612736','--core=0','--',binary,str(depth),str(mode)]
            p=subprocess.run(cmd,input=header+'\n'+encode(q['input'])+'\n',text=True,capture_output=True,timeout=timeout)
            status='ok' if p.returncode==0 else 'error:'+str(p.returncode)
            lines=p.stdout.splitlines() if p.returncode==0 else []
        except subprocess.TimeoutExpired:
            status,lines='timeout',[]
        candidates=[]
        for line in lines:
            if not line.startswith('GRID '):
                continue
            _,score,n,h,w,*v=line.split()
            h,w=int(h),int(w)
            vals=list(map(int,v))
            if 1<=h<=30 and 1<=w<=30 and len(vals)==h*w and all(0<=i<=9 for i in vals):
                candidates.append([float(score),int(n),[vals[i:i+w] for i in range(0,len(vals),w)]])
        results.append(dict(status=status,candidates=candidates))
    return results


def main():
    p=argparse.ArgumentParser()
    for name in ('blind','binary','out'):
        p.add_argument(name,type=Path)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--depth',type=int,default=2)
    p.add_argument('--mode',type=int,default=0)
    p.add_argument('--timeout',type=int,default=120)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    def process(path):
        dest=a.out/(path.stem+'.json')
        result=search(path,str(a.binary.resolve()),a.depth,a.mode,a.timeout)
        tmp=dest.with_suffix('.tmp')
        tmp.write_text(json.dumps(result,separators=(',',':')))
        tmp.replace(dest)
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(process,sorted(a.blind.glob('*.json'))))


if __name__=='__main__':
    main()
