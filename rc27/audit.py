"""Audit completed predictions with answer keys after inference."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from arbitrate import eligible


def read(path):
    return json.loads(path.read_text())


def audit(labels, baseline, prediction, sources, previous=None):
    families = defaultdict(lambda: dict(proposed=set(), correct=set(), verified=set()))
    old_hits, new_hits = set(), set()
    for path in sorted(labels.glob('*.json')):
        task = read(path)
        target = [query['output'] for query in task['test']]
        key = path.stem
        if all(answer in row.values() for answer, row in zip(target, baseline[key])):
            old_hits.add(key)
        if all(answer in row.values() for answer, row in zip(target, prediction[key])):
            new_hits.add(key)
        for source in sources:
            for record in read(source/path.name)['candidates']:
                data = families[record['family']]
                data['proposed'].add(key)
                if record['grids'] == target:
                    data['correct'].add(key)
                    if eligible(record):
                        data['verified'].add(key)
    report = dict(tasks=len(list(labels.glob('*.json'))),
                  gained=sorted(new_hits-old_hits), lost=sorted(old_hits-new_hits), families={})
    old_coverage = set()
    if previous is not None:
        old_coverage = set(previous['ids']['solved']+previous['ids']['ranking'])
    for name, data in sorted(families.items()):
        item = dict(proposed=len(data['proposed']), correct=len(data['correct']),
                    verified_correct=len(data['verified']), correct_ids=sorted(data['correct']))
        if previous is not None:
            item['new_over_previous_pool'] = sorted(data['correct']-old_coverage)
        report['families'][name] = item
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('labels','baseline','prediction','output'):
        parser.add_argument(name,type=Path)
    parser.add_argument('sources',nargs='+',type=Path)
    parser.add_argument('--previous',type=Path)
    args = parser.parse_args()
    result = audit(args.labels,read(args.baseline),read(args.prediction),args.sources,
                   read(args.previous) if args.previous else None)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
