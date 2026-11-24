"""Mine shared expression DAGs from Hodel's training-only ARC programs."""
import argparse,ast,json,signal,sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np


class Budget(Exception):
    pass


class Bank:
    def __init__(self,source):
        sys.path.insert(0,str(source))
        import constants,dsl
        self.env={**vars(constants),**vars(dsl)}
        self.nodes=[]
        self.origins=[]
        self.sizes=[]
        cache={}
        for fn in ast.parse((source/'solvers.py').read_text()).body:
            if not isinstance(fn,ast.FunctionDef) or not fn.name.startswith('solve_'):
                continue
            origin=fn.name[6:]
            variables={'I':-1}
            for line in fn.body:
                if not isinstance(line,ast.Assign):
                    continue
                call=line.value
                terms=tuple(variables.get(x.id,x.id) for x in [call.func,*call.args])
                if terms not in cache:
                    cache[terms]=len(self.nodes)
                    self.nodes.append(terms)
                    self.origins.append(set())
                    self.sizes.append(1+sum(self.sizes[j] for j in terms if type(j)==int and j>=0))
                idx=cache[terms]
                variables[line.targets[0].id]=idx
                self.origins[idx].add(origin)

    def evaluate(self,grid):
        cache={-1:tuple(tuple(r) for r in grid)}
        def get(i):
            if isinstance(i,str):
                return self.env[i]
            if i in cache:
                return cache[i]
            terms=self.nodes[i]
            try:
                f,*args=[get(t) for t in terms]
                if not callable(f) or any(a is None for a in args):
                    res=None
                else:
                    res=f(*args)
                    if isinstance(res,(tuple,frozenset)) and len(res)>1200:
                        res=None
            except (ArithmeticError,ValueError,TypeError,IndexError,KeyError,StopIteration,RuntimeError,OverflowError):
                res=None
            cache[i]=res
            return res
        return get


def image(value,source,mode):
    if mode=='grid':
        if not isinstance(value,tuple) or not 1<=len(value)<=30 or not isinstance(value[0],tuple):
            return
        try:
            a=np.asarray(value)
        except (TypeError,ValueError):
            return
        if a.ndim==2 and 1<=a.shape[1]<=30 and a.dtype.kind in 'iu' and a.min()>=0 and a.max()<=9:
            return a
    elif isinstance(value,frozenset) and value:
        first=next(iter(value))
        if not isinstance(first,tuple) or len(first)!=2:
            return
        colored=isinstance(first[1],tuple)
        if colored and not all(isinstance(p,tuple) and len(p)==2 for p in value):
            return
        coords=[p for _,p in value] if colored else list(value)
        if not all(isinstance(p,tuple) and len(p)==2 and all(isinstance(v,int) for v in p) for p in coords):
            return
        rc=np.asarray(coords)
        r,c=rc.T
        h,w=source.shape
        if mode=='crop':
            y,x=max(0,int(r.min())),max(0,int(c.min()))
            end_y,end_x=min(h,int(r.max())+1),min(w,int(c.max())+1)
            if y<end_y and x<end_x:
                return source[y:end_y,x:end_x]
        elif mode=='mask':
            m=np.zeros_like(source)
            ok=(r>=0)&(r<h)&(c>=0)&(c<w)
            m[r[ok],c[ok]]=1
            return m


def proposals(task,bank,excluded):
    if any('output' in e for e in task['test']):
        raise ValueError('Expected unlabeled queries')
    demos=sorted(task['train'],key=lambda e:np.size(e['input']))
    targets=[np.array(e['output']) for e in demos]
    inputs=[np.array(e['input']) for e in demos+task['test']]
    n=len(demos)
    for canon in (False,True):
        sources=[]
        getters=[]
        for g in inputs:
            if canon:
                colors=sorted(range(10),key=lambda c:(-np.sum(g==c),c))
                v=np.argsort(colors)[g]
            else:
                v=g
            sources.append(v)
            getters.append(bank.evaluate(v.tolist()))
        first_seen={}
        for i in sorted(range(len(bank.nodes)),key=bank.sizes.__getitem__):
            if bank.origins[i]=={excluded}:
                continue
            value=getters[0](i)
            modes=('grid',) if isinstance(value,tuple) else ('mask','crop') if isinstance(value,frozenset) else ()
            for mode in modes:
                base=image(value,sources[0],mode)
                if base is None or base.shape!=targets[0].shape:
                    continue
                data=base.astype('B').tobytes()
                if data not in first_seen:
                    lut=np.full(10,-1)
                    lut[base.ravel()]=targets[0].ravel()
                    first_seen[data]=lut if np.array_equal(lut[base],targets[0]) else None
                lut=first_seen[data]
                if lut is None:
                    continue
                lut=lut.copy()
                outs=[]
                valid=True
                for j in range(1,len(inputs)):
                    v=image(getters[j](i),sources[j],mode)
                    if v is None:
                        valid=False
                        break
                    if j<n:
                        if v.shape!=targets[j].shape:
                            valid=False
                            break
                        prev=lut.copy()
                        lut[v.ravel()]=targets[j].ravel()
                        if ((prev>=0)&(prev!=lut)).any() or not np.array_equal(lut[v],targets[j]):
                            valid=False
                            break
                    else:
                        a=lut[v]
                        if (a<0).any():
                            valid=False
                            break
                        outs.append(a.tolist())
                if valid:
                    edits=int(np.sum((lut>=0)&(lut!=np.arange(10))))
                    cost=bank.sizes[i]+int(canon)+int(mode!='grid')+2*edits
                    yield cost,i,mode,canon,lut.tolist(),outs


def init(source,seconds):
    global BANK,SECONDS
    BANK=Bank(Path(source))
    SECONDS=seconds
    def interrupt(*_):
        raise Budget()
    signal.signal(signal.SIGPROF,interrupt)


def process(args):
    path,folder=args
    best={}
    status='ok'
    signal.setitimer(signal.ITIMER_PROF,SECONDS)
    try:
        for cost,i,mode,canon,lut,grids in proposals(json.loads(path.read_text()),BANK,path.stem):
            key=json.dumps(grids,separators=(',',':'))
            if key not in best:
                best[key]=[cost,0,[i,mode,canon,lut],grids]
            row=best[key]
            row[1]+=1
            if cost<row[0]:
                row[0],row[2]=cost,[i,mode,canon,lut]
    except Budget:
        status='budget'
    finally:
        signal.setitimer(signal.ITIMER_PROF,0)
    rows=sorted(best.values(),key=lambda v:(v[0],-v[1]))[:24]
    (folder/path.name).write_text(json.dumps(dict(status=status,candidates=rows),separators=(',',':')))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('blind','source','out'):p.add_argument(k,type=Path)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--seconds',type=int,default=8)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(a.workers,initializer=init,initargs=(a.source.resolve(),a.seconds)) as pool:
        list(pool.map(process,[(v,a.out) for v in sorted(a.blind.glob('*.json'))]))
