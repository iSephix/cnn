"""Search in an edit-mask space, then decode the predicted mask on the original canvas."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tempfile import TemporaryDirectory
import argparse,json
import numpy as np
from native import blind,search


def encode_task(t):
    blind(t)
    demos=t['train']
    if any(np.shape(e['input'])!=np.shape(e['output']) for e in demos):
        return
    table={}
    masked=[]
    for e in demos:
        g,y=np.asarray(e['input']),np.asarray(e['output'])
        change=g!=y
        if not change.any():
            continue
        for before,after in zip(g[change],y[change]):
            if table.setdefault(int(before),int(after))!=after:
                return
    if not table:
        return
    for e in demos:
        mask=np.not_equal(e['input'],e['output']).astype(int).tolist()
        masked.append(dict(input=e['input'],output=mask))
    return dict(train=masked,test=t['test']),table


def main():
    p=argparse.ArgumentParser()
    for name in ('blind','binary','out'):p.add_argument(name,type=Path)
    p.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory() as tmp:
        def process(path):
            t=json.loads(path.read_text())
            latent=encode_task(t)
            records=[dict(status='inapplicable',candidates=[]) for _ in t['test']]
            if latent:
                sub,table=latent
                subfile=Path(tmp)/path.name
                subfile.write_text(json.dumps(sub))
                records=search(subfile,str(a.binary.resolve()),2,0,120)
                for record,q in zip(records,t['test']):
                    g=np.array(q['input'])
                    colors=set(table.values())
                    lut=np.array([table.get(i,next(iter(colors)) if len(colors)==1 else i) for i in range(10)])
                    outputs=[]
                    for score,n,mask in record['candidates']:
                        m=np.asarray(mask)
                        if m.shape==g.shape and np.isin(m,[0,1]).all():
                            outputs.append([score,n,np.where(m,lut[g],g).tolist()])
                    record['candidates']=outputs
            (a.out/path.name).write_text(json.dumps(records,separators=(',',':')))
        with ThreadPoolExecutor(a.workers) as pool:
            list(pool.map(process,sorted(a.blind.glob('*.json'))))


if __name__=='__main__':main()
