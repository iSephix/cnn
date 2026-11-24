"""Extract and rebind the learned decision paths of numerical concept cards."""
import numpy as np
from memory import Frame

def library(bank):
    by_key={}
    for idx,b in enumerate(bank):
        p=b['card']['predicates']
        for path,_ in b['card']['leaves']:
            terms=tuple(sorted(set((int(p[abs(i)-1][0]),float(p[abs(i)-1][1]),i>0) for i in path)))
            if terms:by_key.setdefault((b['canonical'],terms),idx)
    return sorted([(canon,terms,idx) for (canon,terms),idx in by_key.items()],key=lambda row:len(row[1]))


def matches(frames,parts):
    # Threshold activations are shared across all paths.
    x=np.concatenate([f.features for f in frames])
    gates={}
    for terms,idx in parts:
        mask=np.ones(len(x),dtype=bool)
        for test in terms:
            if test not in gates:
                j,t,greater=test
                gates[test]=(x[:,j]>t) if greater else (x[:,j]<=t)
            mask &= gates[test]
        yield mask,terms,idx


def fragments(task,bank):
    demos=task['train']
    if any(np.shape(e['input'])!=np.shape(e['output']) for e in demos):return
    parts=library(bank)
    for k in (0,4,1,2,3,5,6,7):
        for canonical in (False,True):
            fs=[Frame(e['input'],canonical,k) for e in demos+task['test']]
            n=len(demos)
            g=np.concatenate([f.g.ravel() for f in fs])
            y=np.concatenate([f.target(e['output']).ravel() for f,e in zip(fs,demos)])
            base=g[:len(y)]
            diff=y!=base
            if not diff.any():continue
            lut=np.arange(10)
            lut[base[diff]]=y[diff]
            if not np.array_equal(lut[base[diff]],y[diff]):continue
            # Every changed pixel must pass the path. Unchanged pixels may
            # pass only when their rebound color also stays the same.
            forbidden=(~diff)&(lut[base]!=base)
            for mask,terms,idx in matches(fs,[(t,i) for c,t,i in parts if c==canonical]):
                train=mask[:len(y)]
                if not train[diff].all() or train[forbidden].any():continue
                result=np.where(mask,lut[g],g)
                pos=len(y)
                outs=[]
                for f in fs[n:]:
                    outs.append(f.decode(result[pos:pos+f.g.size].reshape(f.g.shape)).tolist())
                    pos+=f.g.size
                yield 2+len(terms),['path',idx,terms,k,canonical,lut.tolist()],outs
