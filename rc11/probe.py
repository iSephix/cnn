"""Blind candidate audit. Store hashes of full-task predictions from each RC11 family."""
import hashlib,json,sys
from pathlib import Path
from run import LEARNERS


def fingerprint(grids):
    return hashlib.sha256(json.dumps(grids,separators=(',',':')).encode()).hexdigest()


def emit(folder,out):
    tasks = {}
    for path in sorted(Path(folder).glob('*.json')):
        task = json.loads(path.read_text())
        if any('output' in e for e in task['test']):
            raise ValueError('Test labels present')
        tasks[path.stem] = {name:sorted({fingerprint(grids) for _,_,grids in learner(task)})
                            for name,learner in LEARNERS.items()}
    Path(out).write_text(json.dumps(tasks,separators=(',',':'))+'\n')


def score(folder,pool,final):
    candidates = json.loads(Path(pool).read_text())
    rows = {r['id']:all(r['hits']) for r in json.loads(Path(final).read_text())['rows']}
    if candidates.keys()!=rows.keys():
        raise ValueError('Different task sets')
    missed = []
    by_family = {name:0 for name in LEARNERS}
    for task,groups in candidates.items():
        data = json.loads((Path(folder)/(task+'.json')).read_text())
        key = fingerprint([q['output'] for q in data['test']])
        matches = [fam for fam,values in groups.items() if key in values]
        for fam in matches:
            by_family[fam] += 1
        if matches and not rows[task]:
            missed.append(task)
    print(json.dumps(dict(tasks=len(rows),final_solved=sum(rows.values()),missed_in_pool=missed,
                          exact_candidates_by_family=by_family),indent=2))


if __name__=='__main__':
    {'emit':emit,'score':score}[sys.argv[1]](*sys.argv[2:])
