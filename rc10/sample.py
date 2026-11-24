"""Select a fresh supplementary ARC-2 training sample after freezing RC10."""
import hashlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rc4'))
from transfer import select,digest


def sample(one,two,out):
    with TemporaryDirectory() as tmp:
        old = Path(tmp)/'previous'
        select(one,two,old)
        folders = (old,one/'training',one/'evaluation',two/'evaluation')
        excluded = [p for d in folders for p in d.glob('*.json')]
        ids = {p.stem for p in excluded}
        hashes = {digest(p) for p in excluded}
    order = lambda p: hashlib.sha256(('rc10-frozen-transfer-v1/'+p.stem).encode()).digest()
    chosen = []
    for p in sorted((two/'training').glob('*.json'),key=order):
        if p.stem in ids:
            continue
        sig = digest(p)
        if sig not in hashes:
            hashes.add(sig)
            chosen.append(p)
        if len(chosen)==100:
            break
    if len(chosen)!=100:
        raise ValueError('Insufficient distinct tasks')
    out.mkdir(parents=True,exist_ok=False)
    for p in chosen:
        (out/p.name).write_bytes(p.read_bytes())


if __name__=='__main__':
    sample(*map(Path,sys.argv[1:]))
