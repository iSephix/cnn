"""Reserve the second attempt for a unanimous object-relation prediction."""
import json
from pathlib import Path
import sys


def apply(predictions, relation_dir):
    for task, attempts in predictions.items():
        relations=json.loads((relation_dir/f'{task}.json').read_text())
        if len(relations)!=len(attempts):
            raise ValueError(f'Missing query in relation result for {task}')
        if not all(len(r['candidates'])==1 for r in relations):
            continue
        for row,rel in zip(attempts,relations):
            grid=rel['candidates'][0][-1]
            if grid!=row['attempt_1']:
                row['attempt_2']=grid
    return predictions


if __name__=='__main__':
    ranked, relations, output=map(Path,sys.argv[1:])
    result=apply(json.loads(ranked.read_text()),relations)
    output.write_text(json.dumps(result,separators=(',',':'))+'\n')
