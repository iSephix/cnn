"""Rank exact-demonstration hypotheses into the remaining RC9 attempts."""
import json,sys
from pathlib import Path
from induce import infer


def run(folder, prior, out, family):
    learners = {'induce': infer}
    if family != 'induce':
        from programs import LEARNERS
        learners.update(LEARNERS)
    read = lambda p: json.loads(Path(p).read_text())
    pred = read(prior)
    traces = read(prior+'.trace.json')
    for path in sorted(Path(folder).glob('*.json')):
        task = read(path)
        if any('output' in q for q in task['test']):
            raise ValueError('Test labels present')
        tags = traces[path.stem]
        if all('identity' not in row for row in tags):
            continue
        best = {}
        for cost,tag,grids in learners[family](task):
            key = json.dumps(grids)
            if key not in best or cost<best[key][0]:
                best[key] = cost,tag,grids
        fits = sorted(best.values(),key=lambda row:row[0])
        for i,q in enumerate(task['test']):
            choices = [(pred[path.stem][i][f'attempt_{j}'],t) for j,t in enumerate(tags[i],1) if t!='identity']
            choices.extend((grids[i],tag) for _,tag,grids in fits)
            choices.append((q['input'],'identity'))
            a = choices[0]
            b = next((o for o in choices if o[0]!=a[0]),a)
            pred[path.stem][i] = dict(attempt_1=a[0],attempt_2=b[0])
            tags[i] = [a[1],b[1]]
    for name,data in ((out,pred),(out+'.trace.json',traces)):
        Path(name).write_text(json.dumps(data,separators=(',',':'))+'\n')


if __name__=='__main__':
    run(*sys.argv[1:])
