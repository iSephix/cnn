"""After predictions are frozen, measure per-source recall and regressions."""
import json,sys
from pathlib import Path
from rank import read,rows


def audit(labels,prior,pool,final,natives):
    previous=read(prior)
    tags=read(str(prior)+'.trace.json')
    predictions=read(final)
    wins,losses=[],[]
    search_pass2=[0 for _ in natives]
    failures=[{} for _ in natives]
    counts=dict(selected=0,prior=0,oracle=0)
    for path in sorted(labels.glob('*.json')):
        t=read(path)
        key=path.stem
        before,after,possible=True,True,True
        files=[read(d/path.name) for d in natives]
        for j,records in enumerate(files):
            search_pass2[j]+=all(e['output'] in [v[-1] for v in r['candidates'][:2]]
                                 for e,r in zip(t['test'],records))
            for r in records:
                status=r['status']
                failures[j][status]=failures[j].get(status,0)+1
        sources=read(pool/path.name)
        for i,q in enumerate(t['test']):
            y=q['output']
            before &= y in previous[key][i].values()
            after &= y in predictions[key][i].values()
            groups=rows(t,previous[key][i],tags[key][i],files,sources,i)
            possible &= json.dumps(y,separators=(',',':')) in groups
        counts['selected']+=after
        counts['prior']+=before
        counts['oracle']+=possible
        if after and not before:wins.append(key)
        if before and not after:losses.append(key)
    return dict(tasks=len(predictions),**counts,gains=wins,losses=losses,search_pass2=search_pass2,search_status=failures)


if __name__=='__main__':
    labels,prior,pool,final,*sources=map(Path,sys.argv[1:])
    print(json.dumps(audit(labels,prior,pool,final,sources),indent=2))
