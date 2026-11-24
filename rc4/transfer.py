"""Select a reproducible supplementary sample excluding known ARC-1 and ARC-2 eval tasks."""
from pathlib import Path
import hashlib
import json
import sys

def digest(p):
    obj=json.loads(p.read_text())
    data=json.dumps(obj,sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(data).hexdigest()

def select(one,two,out):
    excluded=[p for folder in (one/'training',one/'evaluation',two/'evaluation') for p in folder.glob('*.json')]
    seen_ids={p.stem for p in excluded}
    seen_hashes={digest(p) for p in excluded}
    order=lambda p:hashlib.sha256(('rc4-transfer-v1/'+p.stem).encode()).digest()
    out.mkdir(parents=True,exist_ok=False)
    for p in sorted((two/'training').glob('*.json'),key=order):
        key=digest(p)
        if p.stem in seen_ids or key in seen_hashes:
            continue
        seen_hashes.add(key)
        (out/p.name).write_bytes(p.read_bytes())
        if len(list(out.glob('*.json')))==100:
            return
    raise ValueError('Insufficient distinct tasks')

if __name__=='__main__':
    select(*map(Path,sys.argv[1:]))
