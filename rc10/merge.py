"""Keep prior attempts and rank distinct new grids by independent family agreement."""
import json,sys
from pathlib import Path


def read(p):
    return json.loads(Path(p).read_text())


def merge(prior, out, files):
    base = read(prior)
    trace = read(prior+'.trace.json')
    proposals = [(read(p),read(p+'.trace.json')) for p in files]
    for task,queries in base.items():
        for i,attempts in enumerate(queries):
            saved = [(attempts[f'attempt_{j}'],tag) for j,tag in enumerate(trace[task][i],1)]
            options = [p for p in saved if p[1]!='identity']
            previous = {json.dumps(g) for g,_ in options}
            groups = {}
            for family,(data,tags) in enumerate(proposals):
                for j,tag in enumerate(tags[task][i],1):
                    grid = data[task][i][f'attempt_{j}']
                    key = json.dumps(grid)
                    if tag=='identity' or key in previous:
                        continue
                    if key not in groups:
                        groups[key] = [set(), (j,family), grid, tag]
                    groups[key][0].add(family)
                    if (j,family)<groups[key][1]:
                        groups[key][1:] = [(j,family),grid,tag]
            options.extend((g,t) for _,_,g,t in sorted(groups.values(),key=lambda p:(-len(p[0]),p[1])))
            options.extend(saved)
            a = options[0]
            b = next((v for v in options if v[0]!=a[0]),a)
            queries[i] = dict(attempt_1=a[0],attempt_2=b[0])
            trace[task][i] = [a[1],b[1]]
    for p,obj in ((out,base),(out+'.trace.json',trace)):
        Path(p).write_text(json.dumps(obj,separators=(',',':'))+'\n')


if __name__=='__main__':
    merge(sys.argv[1],sys.argv[2],sys.argv[3:])
