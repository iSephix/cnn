"""Score candidate programs and compare a bounded release policy with RC25."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def evaluate(tasks, cache, baseline=None):
    candidates = []
    first = []
    second = []
    states = 0
    output = {}
    for path in sorted(tasks.glob('*.json')):
        task = read(path)
        result = read(cache/path.name)
        states += result['states']
        rows = result['candidates']
        answers = [q['output'] for q in task['test']]
        if any(all(g == y for g, y in zip(row['grids'], answers)) for row in rows):
            candidates.append(path.stem)
        if rows and rows[0]['grids'] == answers:
            first.append(path.stem)
        if any(row['grids'] == answers for row in rows[:2]):
            second.append(path.stem)
        if baseline is not None:
            choices = baseline[path.stem]
            # This experimental policy admits only a unique complete program output.
            for index, old in enumerate(choices):
                prediction = dict(old)
                if len(rows) == 1 and rows[0]['grids'][index] != old['attempt_1']:
                    prediction['attempt_2'] = rows[0]['grids'][index]
                output.setdefault(path.stem, []).append(prediction)
    return dict(tasks=len(list(tasks.glob('*.json'))), states=states,
                candidate_coverage=len(candidates), first=len(first), two=len(second),
                covered_ids=candidates, first_ids=first, two_ids=second), output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tasks', type=Path)
    parser.add_argument('cache', type=Path)
    parser.add_argument('report', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    report, output = evaluate(args.tasks, args.cache, read(args.baseline) if args.baseline else None)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    if args.out:
        args.out.write_text(json.dumps(output, separators=(',', ':'))+'\n')
