"""Reserve a guess for unanimous, demonstration-verified concept predictions."""
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def select(predictions, directories):
    for task, attempts in predictions.items():
        claims=[]
        for folder in directories:
            records=read(folder/f'{task}.json')
            if len(records)!=len(attempts):
                raise ValueError(f'Missing query for {task}')
            if not all(len(row['candidates'])==1 for row in records):
                continue
            candidates=[row['candidates'][0] for row in records]
            if folder.name.startswith('route') and any(n-score>.0701 for score,n,_ in candidates):
                continue
            claims.append([grid for _,_,grid in candidates])
        if not claims or any(grids!=claims[0] for grids in claims):
            continue
        for row,grid in zip(attempts,claims[0]):
            if row['attempt_1']!=grid:
                row['attempt_2']=grid
    return predictions


if __name__=='__main__':
    base,output,*sources=map(Path,sys.argv[1:])
    result=select(read(base),sources)
    output.write_text(json.dumps(result,separators=(',',':'))+'\n')
