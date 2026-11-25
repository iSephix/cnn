"""Allocate attempts across the baseline and the demonstration-verified bank."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def select(rows,candidates,policy):
    result=[]
    for row,bank in zip(rows,candidates):
        original=[row['attempt_1'],row['attempt_2']]
        identity=[c for c in bank if c['identity_votes']]
        if policy in ('reserve','bank-first','bank-only'):
            pool=bank
        else:
            pool=identity
        leader=pool[0]['grid'] if pool else None
        if leader is None:
            pair=original
        elif policy=='bank-only':
            pair=[leader,pool[min(1,len(pool)-1)]['grid']]
        elif leader in original:
            pair=original
        else:
            pair=[original[0],leader]
        if policy=='bank-first' and pair[1]==leader and leader is not None:
            pair=pair[::-1]
        result.append(dict(attempt_1=pair[0],attempt_2=pair[1]))
    return result


def combine(baseline,bank,policy):
    output={}
    for key,rows in baseline.items():
        candidates=read(bank/f'{key}.json')
        if len(candidates)!=len(rows):
            raise ValueError(f'Query count mismatch for {key}')
        output[key]=select(rows,candidates,policy)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('baseline','bank','output'):
        p.add_argument(name,type=Path)
    p.add_argument('--policy',choices=('identity','reserve','bank-first','bank-only'),default='reserve')
    a=p.parse_args()
    pred=combine(read(a.baseline),a.bank,a.policy)
    a.output.write_text(json.dumps(pred,separators=(',',':'))+'\n')
