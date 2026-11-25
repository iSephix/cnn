"""Score the bank and its fusion with RC21 after blind prediction."""
import argparse
import json
from pathlib import Path

from select_attempts import read


def evaluate(labels, baseline, bank, combined):
    hits={k:[] for k in ('baseline','original_orientation_supported','augmented_only','bank_coverage','combined')}
    for path in sorted(labels.glob('*.json')):
        task=path.stem
        target=[e['output'] for e in read(path)['test']]
        rows=read(bank/path.name)
        old=baseline[task]
        new=combined[task]
        if not len(target)==len(rows)==len(old)==len(new):
            raise ValueError(f'Query count mismatch for {task}')
        canonical=[[c['grid'] for c in row if c['identity_votes']][:2] for row in rows]
        augmented=[[c['grid'] for c in row[:2]] for row in rows]
        all_bank=[[c['grid'] for c in row] for row in rows]
        possibilities={'baseline':[v.values() for v in old],'original_orientation_supported':canonical,
                       'augmented_only':augmented,'bank_coverage':all_bank,'combined':[v.values() for v in new]}
        for method,guess in possibilities.items():
            if all(expected in options for expected,options in zip(target,guess)):
                hits[method].append(task)
    before,after=set(hits['baseline']),set(hits['combined'])
    return dict(tasks=len(baseline),scores={method:len(ids) for method,ids in hits.items()},
                gains=sorted(after-before),losses=sorted(before-after),
                bank_coverage_ids=hits['bank_coverage'])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for field in ('labels','baseline','bank','combined'):
        p.add_argument(field,type=Path)
    a=p.parse_args()
    print(json.dumps(evaluate(a.labels,read(a.baseline),a.bank,read(a.combined)),indent=2))
