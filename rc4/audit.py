"""Retrospective paired changes of exact task hits, attributed to emitted guesses."""
import json
from pathlib import Path
import sys


def load(p):
    return json.loads(Path(p).read_text())


def audit(results, n):
    rows=lambda p: {r['id']:r['hits'] for r in load(p)['rows']}
    old=rows(f'rc3/results/arc{n}-score.json')
    new=rows(Path(results)/f'arc{n}-score.json')
    sources=load(Path(results)/f'arc{n}.json.trace.json')
    if new.keys()!=old.keys():
        raise ValueError('Split differs between runs')
    return {'gained':{k:[sources[k][i][hit-1] for i,hit in enumerate(hits)]
                      for k,hits in new.items() if all(hits) and not all(old[k])},
            'lost':[k for k in old if all(old[k]) and not all(new[k])]}


if __name__=='__main__':
    print(json.dumps(audit(*sys.argv[1:]),indent=2))
