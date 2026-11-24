"""Write RC8 predictions, including which training task supplied each used card."""
import sys
from pathlib import Path
import json
from memory import read,reuse


def run(folder,prior,bank,out):
    old=read(prior)
    traces=read(prior+'.trace.json')
    bank=read(bank)
    pred={}
    for p in sorted(Path(folder).glob('*.json')):
        task=read(p)
        if any('output' in q for q in task['test']):raise ValueError('Test labels present')
        fits=sorted(reuse(task,bank),key=lambda e:e[0])
        pred[p.stem]=[]
        for i,q in enumerate(task['test']):
            choices=[(old[p.stem][i][f'attempt_{j}'],tag) for j,tag in enumerate(traces[p.stem][i],1) if tag!='identity']
            choices.extend((y[i],['memory',*tag,bank[tag[1]]['origins']]) for _,tag,y in fits)
            choices.append((q['input'],'identity'))
            first=choices[0]
            second=next((v for v in choices if v[0]!=first[0]),first)
            pred[p.stem].append(dict(attempt_1=first[0],attempt_2=second[0]))
            traces[p.stem][i]=[first[1],second[1]]
    for path,data in [(out,pred),(out+'.trace.json',traces)]:
        Path(path).write_text(json.dumps(data,separators=(',',':'))+'\n')
if __name__=='__main__':run(*sys.argv[1:])
