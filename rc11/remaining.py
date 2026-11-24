"""Select all remaining ARC-2 training tasks disjoint by ID and content from earlier splits."""
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rc10'))
import sample as rc10


def select(one,two,out):
    with TemporaryDirectory() as temp:
        earlier = Path(temp)/'rc10'
        first = Path(temp)/'rc4'
        rc10.sample(one,two,earlier)
        rc10.select(one,two,first)
        files = [p for folder in (one/'training',one/'evaluation',two/'evaluation',earlier,first)
                 for p in folder.glob('*.json')]
        ids = {p.stem for p in files}
        hashes = {rc10.digest(p) for p in files}
    ordered = sorted((two/'training').glob('*.json'))
    chosen = []
    for p in ordered:
        if p.stem in ids:
            continue
        digest = rc10.digest(p)
        if digest not in hashes:
            hashes.add(digest)
            chosen.append(p)
    if not chosen:
        raise ValueError('No remaining distinct tasks')
    out.mkdir(parents=True)
    for p in chosen:
        (out/p.name).write_bytes(p.read_bytes())


if __name__=='__main__':
    select(*map(Path,sys.argv[1:]))
