import json,sys
from pathlib import Path
from compose import synth
from regions import infer as regions
from latent import infer as latent
from exemplar import infer as exemplar
from motion import infer as motion
from rewrite import infer as rewrite


LEARNERS = dict(compose=synth, regions=regions, latent=latent, exemplar=exemplar, motion=motion, rewrite=rewrite)

def run(folder,prior,out,family="compose"):
    read = lambda p: json.loads(Path(p).read_text())
    pred = read(prior)
    trace = read(prior+'.trace.json')
    for path in sorted(Path(folder).glob('*.json')):
        t = read(path)
        if any("output" in q for q in t["test"]):
            raise ValueError("Test labels present")
        best = {}
        for cost,tag,grids in LEARNERS[family](t):
            key = json.dumps(grids)
            if key not in best or cost<best[key][0]:
                best[key] = cost,tag,grids
        fits = sorted(best.values(),key=lambda f:f[0])
        for i,row in enumerate(pred[path.stem]):
            tags = trace[path.stem][i]
            saved = [(row[f'attempt_{j}'],tag) for j,tag in enumerate(tags,1)]
            if not fits:
                continue
            strong = [v for v in fits if v[0][0]==0]
            movable = ('induce','resample','pointers')
            options = [saved[0]] if tags[0]!='identity' else []
            if tags[1]!='identity' and not (strong and isinstance(tags[1],list) and tags[1][0] in movable):
                options.append(saved[1])
            options.extend((grids[i],tag) for _,tag,grids in (strong or fits))
            options.extend(saved)
            a = options[0]
            b = next((x for x in options if x[0]!=a[0]),a)
            row.update(attempt_1=a[0],attempt_2=b[0])
            trace[path.stem][i] = [a[1],b[1]]
    for p,data in ((out,pred),(out+'.trace.json',trace)):
        Path(p).write_text(json.dumps(data,separators=(',',':'))+'\n')


if __name__=='__main__':
    run(*sys.argv[1:])
