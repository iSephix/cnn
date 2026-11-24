"""Regression tests for stripped input, latent decoders and independent evidence."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from native import blind,search
from residual import encode_task
from motifs import encode
from latent import decode
from rank import rows
import numpy as np
import sys


def test(binary):
    task={'train':[{'input':[[1,0]],'output':[[2,0]]},{'input':[[0,1]],'output':[[0,2]]}],'test':[{'input':[[1,1]]}]}
    sub,table=encode_task(task)
    assert table=={1:2} and sub['train'][0]['output']==[[1,0]]
    try:
        blind({'train':task['train'],'test':[{'input':[[1]],'output':[[2]]}]})
    except ValueError:
        pass
    else:
        raise AssertionError('Accepted test labels')
    mosaic={'train':[], 'test':[{'input':[[0,1]]}]}
    for g in ([[0,1,0],[1,1,0]],[[1,0],[0,1]]):
        y=np.repeat(np.repeat(g,2,axis=0),2,axis=1)
        mosaic['train'].append(dict(input=g,output=y.tolist()))
    for latent,tiles,colors,panels in encode(mosaic):
        lut={c:i for i,c in enumerate(colors)}
        for e,orig in zip(latent['train'],mosaic['train']):
            ids=np.vectorize(lut.__getitem__)(e['output'])
            assert decode(ids,tiles,panels).tolist()==orig['output']
        break
    else:raise AssertionError('No repeated motifs')
    with TemporaryDirectory() as d:
        p=Path(d)/'anon.json'
        p.write_text(json.dumps(task))
        ans=search(p,binary,2,0,30)
        assert ans[0]['status']=='ok' and ans[0]['candidates']
    print('Blind adapter, edit mask and motif round trip passed.')


if __name__=='__main__':test(sys.argv[1])
