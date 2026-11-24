"""Compare exact task hits to RC5 and identify each successful emitted program."""
import json,sys
from pathlib import Path

def read(p):
    return json.loads(Path(p).read_text())

def report(folder):
    report={}
    for n in (1,2):
        baseline=read(f'rc5/results/arc{n}-score.json')
        old={r['id'] for r in baseline['rows'] if all(r['hits'])}
        runs={}
        for mode in ('rc5','fixed','free','full'):
            data=read(folder/f'{mode}{n}-score.json')
            runs[mode]=data['summary']
        new={r['id']:r['hits'] for r in data['rows'] if all(r['hits'])}
        trace=read(folder/f'full{n}.json.trace.json')
        gains={k:[trace[k][i][h-1] for i,h in enumerate(new[k])] for k in sorted(new.keys()-old)}
        report[n]=dict(runs=runs,gains=gains,losses=sorted(old-new.keys()))
    return report

if __name__=='__main__':print(json.dumps(report(Path(sys.argv[1])),indent=2))
