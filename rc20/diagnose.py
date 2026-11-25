"""Diagnose a completed benchmark using candidate caches and answer labels."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rc12'))
from rank import read, rows


def diagnose(labels, prior_path, pool, output, sources):
    prior=read(prior_path)
    traces=read(str(prior_path)+'.trace.json')
    predictions=read(output)
    buckets={name:[] for name in ('solved','ranking','content','shape')}
    for path in sorted(labels.glob('*.json')):
        task=read(path)
        key=path.stem
        records=[read(folder/path.name) for folder in sources]
        programs=read(pool/path.name)
        shape_match=content_match=solved=True
        for i,query in enumerate(task['test']):
            target=query['output']
            candidates=rows(task,prior[key][i],traces[key][i],records,programs,i)
            size=len(target),len(target[0])
            shape_match &= any((len(g),len(g[0]))==size for g in map(json.loads,candidates))
            content_match &= json.dumps(target,separators=(',',':')) in candidates
            solved &= target in predictions[key][i].values()
        bucket='solved' if solved else 'ranking' if content_match else 'content' if shape_match else 'shape'
        buckets[bucket].append(key)
    return dict(tasks=len(predictions),counts={key:len(ids) for key,ids in buckets.items()},ids=buckets)


if __name__=='__main__':
    labels,prior,pool,output,*sources=map(Path,sys.argv[1:])
    print(json.dumps(diagnose(labels,prior,pool,output,sources),indent=2))
