"""Score candidate sources separately after their blind runs."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def matched(expected, guesses):
    return len(expected)==len(guesses) and all(g in row for g,row in zip(expected,guesses))


def audit(labels,original,verified,palette,baseline):
    result={m:[] for m in ('direct','heldout','palette','union','new_vs_rc23')}
    solutions=read(baseline)
    paths=sorted(labels.glob('*.json'))
    for path in paths:
        truth=[q['output'] for q in read(path)['test']]
        direct=[[c['grid'] for c in row] for row in read(original/path.name)]
        subsets=[[c['grid'] for c in row if c['folds']] for row in read(verified/path.name)]
        colors=[[c['grid'] for c in row] for row in read(palette/path.name)]
        prior=[list(row.values()) for row in solutions[path.stem]]
        modes=dict(direct=direct,heldout=subsets,palette=colors,
                   union=[p+a+b+c for p,a,b,c in zip(prior,direct,subsets,colors)])
        for mode,choices in modes.items():
            if matched(truth,choices):
                result[mode].append(path.stem)
        if matched(truth,modes['union']) and not matched(truth,[p+d for p,d in zip(prior,direct)]):
            result['new_vs_rc23'].append(path.stem)
    return dict(tasks=len(paths),counts={k:len(v) for k,v in result.items()},ids=result)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('labels','original','verified','palette','baseline'):
        p.add_argument(key,type=Path)
    a=p.parse_args()
    print(json.dumps(audit(a.labels,a.original,a.verified,a.palette,a.baseline),indent=2))
