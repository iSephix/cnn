from pathlib import Path
import json,sys
from spatial import expand
from fragments import fragments
from memory import read

def main(folder,prior,out):
    bank=read('rc8/bank.json')
    pred=read(prior)
    trace=read(prior+'.trace.json')
    for p in sorted(Path(folder).glob('*.json')):
        t=read(p)
        if any('output' in e for e in t['test']):raise ValueError('Strip labels')
        fits=sorted([*expand(t,bank),*fragments(t,bank)],key=lambda x:x[0])
        for i,e in enumerate(t['test']):
            opts=[(pred[p.stem][i][f'attempt_{j}'],r) for j,r in enumerate(trace[p.stem][i],1) if r!='identity']
            opts.extend((g[i],rule) for _,rule,g in fits)
            opts.append((e['input'],'identity'))
            a=opts[0]
            b=next((o for o in opts if o[0]!=a[0]),a)
            pred[p.stem][i]=dict(attempt_1=a[0],attempt_2=b[0])
            trace[p.stem][i]=[a[1],b[1]]
    for name,d in [(out,pred),(out+'.trace.json',trace)]:
        Path(name).write_text(json.dumps(d,separators=(',',':'))+'\n')

if __name__=='__main__':main(*sys.argv[1:])
