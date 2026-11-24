"""Learn numerical predicate cards, transport them between color roles and reuse their output."""
import json,sys
from pathlib import Path
from functools import cached_property
import numpy as np
from sklearn.tree import DecisionTreeClassifier
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'rc7'),str(ROOT/'rc3')]
from card import compile_tree,predict
from rules import cell_features
from scene import Scene
from pattern_helpers import views


class Frame:
    def __init__(self,g,canonical,k=0):
        self.grid=np.asarray(g)
        ordered=views(self.grid,k)
        self.raw=ordered
        self.colors=list(range(10))
        if canonical:
            freq=np.bincount(ordered.ravel(),minlength=10)
            # First observed breaks relative-frequency ties equivariantly.
            first={int(v):i for i,v in reversed(list(enumerate(ordered.flat)))}
            self.colors.sort(key=lambda c:(-freq[c],first.get(c,1000+c)))
        self.encode=np.argsort(self.colors)
        self.g=self.encode[ordered]
        self.k=k

    @cached_property
    def features(self):
        return cell_features(Scene(self.g))

    def target(self,y):
        return self.encode[views(np.asarray(y),self.k)]

    def decode(self,y):
        out=np.asarray(self.colors)[y]
        if self.k&2:out=out[:,::-1]
        if self.k&1:out=out[::-1]
        return out.T if self.k&4 else out


def read(p):return json.loads(Path(p).read_text())

def mine(folder):
    bank={}
    for path in sorted(Path(folder).glob('*.json')):
        t=read(path)
        demos=t['train']
        checks=t['test']
        if any(np.shape(e['input'])!=np.shape(e['output']) for e in demos+checks):continue
        for canonical in (False,True):
            frames=[Frame(e['input'],canonical) for e in demos+checks]
            y=[f.target(e['output']) for f,e in zip(frames,demos+checks)]
            for kind in ('paint','change'):
                labels=[np.where(f.g==v,10,v) if kind=='paint' else (f.g!=v).astype(int)
                        for f,v in zip(frames,y)]
                if all(np.unique(v).size==1 for v in labels):continue
                model=DecisionTreeClassifier(random_state=0,max_depth=8)
                n=len(demos)
                model.fit(np.concatenate([f.features for f in frames[:n]]),np.concatenate([v.ravel() for v in labels[:n]]))
                card=compile_tree(model)
                if not all(np.array_equal(predict(card,f.features),v.ravel()) for f,v in zip(frames,labels)):continue
                key=json.dumps([canonical,kind,card],sort_keys=True)
                if key not in bank:bank[key]=dict(canonical=canonical,kind=kind,card=card,origins=[])
                bank[key]['origins'].append(path.stem)
    return sorted(bank.values(),key=lambda c:(len(c['card']['leaves']),-len(c['origins'])))


def codes(entry,f):
    v=predict(entry['card'],f.features).reshape(f.g.shape)
    return np.where(v==10,f.g,v) if entry['kind']=='paint' else v


def fit_binding(keys,targets):
    lut=np.full(110,-1,dtype=int)
    for key,y in zip(keys,targets):
        lut[key.ravel()]=y.ravel()
    return lut if all(np.array_equal(lut[key],y) for key,y in zip(keys,targets)) else None


def reuse(task,bank):
    if any('output' in q for q in task['test']):raise ValueError('Strip labels before inference')
    demos=task['train']
    same=all(np.shape(e['input'])==np.shape(e['output']) for e in demos)
    for k in (0,4,1,2,3,5,6,7):
        for canonical in (False,True):
            frames=[Frame(e['input'],canonical,k) for e in demos+task['test']]
            train=frames[:len(demos)]
            tests=frames[len(demos):]
            targets=[f.target(e['output']) for f,e in zip(train,demos)]
            for idx,entry in enumerate(bank):
                if entry['canonical']!=canonical:continue
                z=[codes(entry,f) for f in train]
                if not same:
                    for invert in (False,True):
                        def extract(v,f):
                            mask=v!=f.g if entry['kind']=='paint' else v!=0
                            if invert:mask=~mask
                            yy,xx=np.where(mask)
                            return f.g[yy.min():yy.max()+1,xx.min():xx.max()+1] if len(yy) else None
                        crops=[extract(v,f) for v,f in zip(z,train)]
                        if any(a is None or a.shape!=b.shape for a,b in zip(crops,targets)):continue
                        lut=fit_binding(crops,targets)
                        if lut is None:continue
                        outs=[]
                        for f in tests:
                            a=extract(codes(entry,f),f)
                            if a is None:break
                            outs.append(f.decode(np.where(lut[a]<0,a,lut[a])).tolist())
                        else:yield 2,('mask-crop',idx,k,invert),outs
                    continue
                if entry['kind']=='paint' and all(np.array_equal(a,b) for a,b in zip(z,targets)):
                    yield 0,('frozen',idx,k),[f.decode(codes(entry,f)).tolist() for f in tests]
                for pair in (False,True):
                    keys=[v*10+f.g if pair else v for f,v in zip(train,z)]
                    lut=fit_binding(keys,targets)
                    if lut is None:continue
                    outs=[]
                    for f in tests:
                        v=codes(entry,f)
                        mapped=lut[v*10+f.g if pair else v]
                        outs.append(f.decode(np.where(mapped<0,f.g,mapped)).tolist())
                    yield 1+pair,('bound',idx,k,pair),outs


if __name__=='__main__':
    Path(sys.argv[2]).write_text(json.dumps(mine(sys.argv[1]),separators=(',',':'))+'\n')
