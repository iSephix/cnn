"""Arbitrate verified structural programs and synthesized native candidates."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rc23'))
from calibrate import features, key, read


def unanimous(rows):
    if rows and all(len(row)==1 for row in rows):
        return [key(row[0]['grid']) for row in rows]


def verified_trace(trace,second):
    direct=trace['direct']
    if not direct or any(key(row[2])==second for row in direct):
        return None
    exact=[key(grid) for score,n,grid in direct if math.ceil(score)==n]
    if len(exact)==1:
        return exact[0]
    repair=trace['repair']
    if not exact and len(repair)==1 and key(repair[0][2])==key(direct[0][2]):
        return key(repair[0][2])


def features_by_query(args,path,task,old):
    native=read(args.traces/path.name)
    structure=read(args.structure/path.name)
    layout=read(args.layout/path.name)
    if not len(old)==len(native)==len(structure)==len(layout)==len(task['test']):
        raise ValueError(f'Query count mismatch for {path.stem}')
    symbolic=[x for x in (unanimous(structure),unanimous(layout)) if x is not None]
    agreed=symbolic[0] if symbolic and all(x==symbolic[0] for x in symbolic) else None
    learned=args.labels is not None or args.policy in ('learned','reserve')
    for index,(question,base,trace,struct,panels) in enumerate(zip(task['test'],old,native,structure,layout)):
        symbolic_answer=agreed[index] if agreed is not None else None
        trace_answer=verified_trace(trace,key(base['attempt_2']))
        if not learned:
            yield {},symbolic_answer,trace_answer
            continue
        metadata=defaultdict(dict)
        for source in ('direct','repair'):
            for i,(score,n,grid) in enumerate(trace[source]):
                record=metadata[key(grid)]
                record[source]=1
                record[source+'/position']=1/(i+1)
                record[source+'/good_examples']=math.ceil(score)/n
                record[source+'/cost']=math.ceil(score)-score
                record[source+'/only_candidate']=int(len(trace[source])==1)
        for source,rows in (('structural',struct),('layout',panels)):
            for i,rec in enumerate(rows):
                record=metadata[key(rec['grid'])]
                record[source]=1
                record[source+'/'+rec['source']]=1
                record[source+'/position']=1/(i+1)
                record[source+'/only_candidate']=int(len(rows)==1)
        claims=[dict(grid=g,identity_votes=0,detectors={},orientations=[]) for g in metadata]
        regular=features(task['train'],question['input'],base,claims)
        for grid,f in regular.items():
            f.update(metadata.get(grid,{}))
            f['verified_concept']=int(agreed is not None and grid==agreed[index])
        yield regular,symbolic_answer,trace_answer


def examples(args):
    baseline=read(args.baseline)
    for path in sorted(args.blind.glob('*.json')):
        task=read(path)
        if any('output' in q for q in task['test']):
            raise ValueError('Expected unlabeled queries')
        original=baseline[path.stem]
        labels=read(args.labels/path.name)['test'] if args.labels else [None]*len(original)
        for base,expected,(choices,unique,trace) in zip(original,labels,features_by_query(args,path,task,original)):
            yield path.stem,base,expected,choices,unique,trace


def train(args):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression

    x=[]
    y=[]
    for _,_,expected,choices,_,_ in examples(args):
        for grid,vector in choices.items():
            x.append(vector)
            y.append(grid==key(expected['output']))
    vec=DictVectorizer()
    data=vec.fit_transform(x)
    model=LogisticRegression(C=.2,max_iter=500).fit(data,y)
    weights=dict(zip(vec.get_feature_names_out(),map(float,model.coef_[0])))
    args.model.write_text(json.dumps(weights,separators=(',',':'))+'\n')


def predict(args):
    weight=read(args.model) if args.policy in ('learned','reserve') else {}
    output={}
    for task,base,_,choices,unique,trace in examples(args):
        first,second=key(base['attempt_1']),key(base['attempt_2'])
        def score(g):
            return sum(weight.get(k,0)*value for k,value in choices[g].items())
        if args.policy=='learned':
            order=sorted(choices,key=score,reverse=True)
            first,second=order[0],order[min(1,len(order)-1)]
        elif args.policy=='reserve':
            order=sorted(choices,key=score,reverse=True)
            best=next((g for g in order if g!=first),second)
            if score(best)>score(second):
                second=best
        elif args.policy=='consensus' and trace is not None and trace!=first:
            second=trace
        if args.policy!='learned' and unique is not None and unique!=first:
            second=unique
        output.setdefault(task,[]).append(dict(attempt_1=first,attempt_2=second))
    args.out.write_text(json.dumps(output,separators=(',',':'))+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('blind','baseline','traces','structure','layout','model'):
        p.add_argument(name,type=Path)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--labels',type=Path)
    group.add_argument('--out',type=Path)
    p.add_argument('--policy',choices=('learned','reserve','verified','consensus'),default='consensus')
    a=p.parse_args()
    train(a) if a.labels else predict(a)
