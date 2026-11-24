import json,sys
from pathlib import Path
from learn import solve

def main(blind,prior,trace,out):
    priors=json.loads(Path(prior).read_text())
    traces=json.loads(Path(trace).read_text())
    pred,origins={},{}
    for p in sorted(Path(blind).glob('*.json')):
        pred[p.stem],origins[p.stem]=solve(json.loads(p.read_text()),priors[p.stem],traces[p.stem])
    Path(out).write_text(json.dumps(pred,separators=(',',':'))+'\n')
    Path(out+'.trace.json').write_text(json.dumps(origins,separators=(',',':'))+'\n')

if __name__=='__main__':
    main(*sys.argv[1:])
