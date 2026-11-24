import json,sys,signal
import time
from pathlib import Path
import numpy as np
from search import infer,step
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rc5'))
from compose import infer as rc5_search,summarize

class Deadline(BaseException):pass

def stop(*args):raise Deadline()

def run(folder,prior,out,method,seconds):
    read=lambda p:json.loads(Path(p).read_text())
    pred=read(prior)
    traces=read(prior+'.trace.json')
    signal.signal(signal.SIGPROF,stop)
    for p in sorted(Path(folder).glob('*.json')):
        t=read(p)
        if any('output' in e for e in t['test']):raise ValueError('Strip labels first')
        xs=[np.array(e['input']) for e in t['train']]
        ys=[np.array(e['output']) for e in t['train']]
        qs=[np.array(e['input']) for e in t['test']]
        step.cache_clear()
        summarize.cache_clear()
        fits=[]
        try:
            start=time.process_time()
            signal.setitimer(signal.ITIMER_PROF,seconds)
            gen=rc5_search(xs,ys,qs) if method=='rc5' else infer(xs,ys,qs,method=='free')
            for cost,rule,answers in gen:
                fits.append((cost,rule,answers))
        except Deadline:pass
        except TypeError:
            # NumPy may wrap an asynchronous timeout during dtype comparison.
            if not seconds or time.process_time()-start<seconds:raise
        finally:signal.setitimer(signal.ITIMER_PROF,0)
        fits.sort(key=lambda e:e[0])
        for i,pair in enumerate(pred[p.stem]):
            opts=[(pair[f'attempt_{j}'],r) for j,r in enumerate(traces[p.stem][i],1) if r!='identity']
            opts.extend((answers[i],rule) for _,rule,answers in fits)
            opts.append((qs[i].tolist(),'identity'))
            first=opts[0]
            second=next((o for o in opts if o[0]!=first[0]),first)
            pair.update(attempt_1=first[0],attempt_2=second[0])
            traces[p.stem][i]=[first[1],second[1]]
    for name,data in [(out,pred),(out+'.trace.json',traces)]:
        Path(name).write_text(json.dumps(data,separators=(',',':'))+'\n')

if __name__=='__main__':run(*sys.argv[1:-1],float(sys.argv[-1]))
