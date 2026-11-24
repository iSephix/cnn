"""Compare frozen paint cards to the best single-color readout on a supplementary split."""
from pathlib import Path
import json,sys
import numpy as np
from memory import Frame,codes,fit_binding,read


def probe(task,bank,terminal):
    demo=task['train']
    test=task['test']
    if any('output' in q for q in test):raise ValueError('Strip labels')
    default=[q['input'] for q in test]
    if any(np.shape(e['input'])!=np.shape(e['output']) for e in demo):return default,None
    for k in (0,4,1,2,3,5,6,7):
        for canonical in (False,True):
            train=[Frame(e['input'],canonical,k) for e in demo]
            queries=[Frame(e['input'],canonical,k) for e in test]
            ys=[f.target(e['output']) for f,e in zip(train,demo)]
            if terminal=='color':
                lut=fit_binding([f.g for f in train],ys)
                if lut is not None:
                    return [f.decode(np.where(lut[f.g]<0,f.g,lut[f.g])).tolist() for f in queries],'color'
            else:
                for i,c in enumerate(bank):
                    if c['canonical']!=canonical or c['kind']!='paint':continue
                    if all(np.array_equal(codes(c,f),y) for f,y in zip(train,ys)):
                        return [f.decode(codes(c,f)).tolist() for f in queries],dict(card=i,origins=c['origins'],view=k)
    return default,None

if __name__=='__main__':
    folder,out=sys.argv[1:3]
    bank=read(sys.argv[3] if len(sys.argv)>3 else 'rc8/bank.json')
    for mode in ('frozen','color'):
        pred,trace={},{}
        for p in sorted(Path(folder).glob('*.json')):
            answers,tag=probe(read(p),bank,mode)
            pred[p.stem]=[dict(attempt_1=g,attempt_2=g) for g in answers]
            trace[p.stem]=tag
        Path(out,mode+'.json').write_text(json.dumps(pred)+'\n')
        Path(out,mode+'-trace.json').write_text(json.dumps(trace)+'\n')
