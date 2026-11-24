"""Use native program search to predict a latent grid of learned output motifs."""
import argparse,json,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tempfile import TemporaryDirectory
import numpy as np
from scipy.optimize import linear_sum_assignment
from native import blind,search
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rc11'))
from latent import decompositions,decode


def encode(t):
    blind(t)
    ys=[np.asarray(e['output']) for e in t['train']]
    options=decompositions(ys)
    for desc,h,w,panels,tiles,codes in options:
        if not 2<=len(tiles)<=10:
            continue
        counts=np.array([np.bincount(tile.ravel(),minlength=10) for tile in tiles])
        _,colors=linear_sum_assignment(-counts)
        sub={'train':[dict(input=e['input'],output=colors[c].tolist()) for e,c in zip(t['train'],codes)],'test':t['test']}
        yield sub,tiles,colors,panels


def main():
    p=argparse.ArgumentParser()
    for k in ('blind','binary','out'):p.add_argument(k,type=Path)
    p.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory() as tmp:
        def process(path):
            t=json.loads(path.read_text())
            result=[dict(status='inapplicable',candidates=[]) for _ in t['test']]
            for j,(sub,tiles,colors,panels) in enumerate(encode(t)):
                if j==2:break
                file=Path(tmp)/path.name
                file.write_text(json.dumps(sub))
                native=search(file,str(a.binary.resolve()),2,0,120)
                lut=np.full(10,-1)
                lut[colors]=np.arange(len(tiles))
                for record,row in zip(result,native):
                    record['status']=row['status']
                    for score,n,g in row['candidates']:
                        ids=lut[g]
                        if ids.min()<0:continue
                        output=decode(ids,tiles,panels)
                        if max(output.shape)<=30:
                            record['candidates'].append([score,n,output.tolist()])
            for row in result:
                unique={}
                for score,n,g in sorted(row['candidates'],key=lambda v:-v[0]):
                    unique.setdefault(json.dumps(g),[score,n,g])
                row['candidates']=list(unique.values())[:12]
            (a.out/path.name).write_text(json.dumps(result,separators=(',',':')))
        with ThreadPoolExecutor(a.workers) as exe:
            list(exe.map(process,sorted(a.blind.glob('*.json'))))


if __name__=='__main__':main()
