"""Retrospective candidate coverage. This scorer is separate from prediction."""
import json,sys
from pathlib import Path


def coverage(labels,merged,files):
    read = lambda path: json.loads(Path(path).read_text())
    actual = read(merged)
    variants = [read(p) for p in files]
    solved = 0
    recovered = []
    for key,queries in actual.items():
        target = [e['output'] for e in read(Path(labels)/(key+'.json'))['test']]
        if len(target)!=len(queries):
            raise ValueError('Mismatched query count')
        hit = all(y in q.values() for y,q in zip(target,queries))
        solved += hit
        if not hit and all(any(y in p[key][i].values() for p in variants) for i,y in enumerate(target)):
            recovered.append(key)
    return dict(tasks=len(actual), merged_pass2=solved, pool_oracle=solved+len(recovered), missed_in_pool=recovered)


if __name__=='__main__':
    print(json.dumps(coverage(sys.argv[1],sys.argv[2],sys.argv[3:]),indent=2))
