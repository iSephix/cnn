"""Retain competing RC11 predictions and evidence from every fitting program."""
from pathlib import Path
import json,sys,warnings,argparse
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rc11'))
from run import LEARNERS
warnings.filterwarnings('ignore',message='The number of unique classes.*')


def collect(args):
    path,out=args
    t=json.loads(path.read_text())
    if any('output' in q for q in t['test']):
        raise ValueError('Expected unlabeled test grids')
    rows={}
    for name,fn in LEARNERS.items():
        votes={}
        for cost,_,grids in fn(t):
            key=json.dumps(grids,separators=(',',':'))
            if key not in votes:
                votes[key]=[list(map(float,cost)),0,grids]
            row=votes[key]
            row[0]=min(row[0],list(map(float,cost)))
            row[1]+=1
        rows[name]=sorted(votes.values(),key=lambda a:(a[0],-a[1]))[:32]
    (out/path.name).write_text(json.dumps(rows,separators=(',',':')))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('blind',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(a.workers) as exe:
        list(exe.map(collect,[(v,a.out) for v in sorted(a.blind.glob('*.json'))]))
