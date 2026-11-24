"""Deterministic grid features shared by object extraction and pixel readouts."""
from collections import Counter, deque
from functools import cached_property
import numpy as np

AXES = ((-1,0),(1,0),(0,-1),(0,1))
DIAGONALS = ((-1,-1),(1,1),(-1,1),(1,-1))


class Object:
    def __init__(self, grid, cells, bg):
        self.points = np.array(cells).T
        r,c = self.points
        a,b = r.min(), c.min()
        self.box = grid[a:r.max()+1,b:c.max()+1]
        self.mask = np.zeros(self.box.shape, bool)
        self.mask[r-a,c-b] = True
        self.patch = np.where(self.mask, self.box, bg)
        h,w = self.box.shape
        colors = grid[r,c]
        self.color = int(np.bincount(colors).argmax())
        self.stats = dict(area=len(cells), width=w, height=h, palette=len(set(colors)),
                          void=h*w-len(cells), top=a, left=b, bottom=r.max(), right=c.max(),
                          border=int(a==0 or b==0 or r.max()+1==len(grid) or c.max()+1==grid.shape[1]),
                          square=int(w==h), solid=int(self.mask.all()))
        self.signature = (self.mask.shape,self.mask.tobytes())


class Scene:
    def __init__(self, grid):
        self.grid = np.array(grid, dtype=int)
        self.bg = int(np.bincount(self.grid.ravel()).argmax())
        self.cache = {}

    def objects(self, mode):
        if mode in self.cache:
            return self.cache[mode]
        mono, connectivity, bg = mode
        bg = self.bg if bg=='mode' else bg
        g=self.grid
        h,w=g.shape
        seen=g==bg
        objects=[]
        for r,c in zip(*np.where(~seen)):
            if seen[r,c]:
                continue
            queue=deque([(r,c)])
            seen[r,c]=True
            cells=[]
            while queue:
                y,x=queue.popleft()
                cells.append((y,x))
                for dy,dx in AXES+(DIAGONALS if connectivity==8 else ()):
                    ny,nx=y+dy,x+dx
                    if 0<=ny<h and 0<=nx<w and not seen[ny,nx] and (not mono or g[ny,nx]==g[r,c]):
                        seen[ny,nx]=True
                        queue.append((ny,nx))
            objects.append(Object(g,cells,bg))
        for key,values in [('shape_count',[o.signature for o in objects]),('color_count',[o.color for o in objects])]:
            freq=Counter(values)
            for o,v in zip(objects,values):
                o.stats[key]=freq[v]
        self.cache[mode]=objects
        return objects

    @cached_property
    def features(self):
        g=self.grid
        h,w=g.shape
        f={'color':g}
        yy,xx=np.indices(g.shape)
        for mono,conn in [(True,4),(False,4),(True,8)]:
            obs=self.objects((mono,conn,self.bg))
            groups={k:np.full_like(g,-1) for k in ('area','width','height','void','border','color_count','shape_count','near_color','largest','smallest')}
            areas=[o.stats['area'] for o in obs]
            for o in obs:
                pos=tuple(o.points)
                for k in o.stats.keys() & groups.keys():
                    groups[k][pos]=o.stats[k]
                groups['largest'][pos]=int(o.stats['area']==max(areas))
                groups['smallest'][pos]=int(o.stats['area']==min(areas))
                others=[p for p in obs if p is not o]
                if others:
                    def gap(p):
                        a,b=o.stats,p.stats
                        return max(0,a['left']-b['right'],b['left']-a['right'])+max(0,a['top']-b['bottom'],b['top']-a['bottom'])
                    groups['near_color'][pos]=min(others,key=gap).color
            for k,v in groups.items():
                f[f'{mono}_{conn}_{k}']=v
        for axis in (0,1):
            occupied=g!=self.bg
            count=occupied.sum(axis=axis,keepdims=True)
            f[f'line_count{axis}']=np.broadcast_to(count,g.shape)
            for direction in (-1,1):
                ray=np.full_like(g,-1)
                line=range(g.shape[axis])
                if direction==1:
                    line=reversed(line)
                last=np.full(g.shape[1-axis],-1)
                for i in line:
                    key=(i,slice(None)) if axis==0 else (slice(None),i)
                    ray[key]=last
                    last=np.where(occupied[key],g[key],last)
                f[f'ray{axis}_{direction}']=ray
        for axis in (0,1):
            a,b=f[f'ray{axis}_-1'],f[f'ray{axis}_1']
            f[f'between{axis}']=np.where(a==b,a,-1)
        pad=np.pad(g,1,constant_values=-1)
        neighbors=[]
        for y,x in AXES+DIAGONALS:
            v=pad[1+y:1+y+h,1+x:1+x+w]
            f[f'neighbor{y}_{x}']=v
            neighbors.append((v!=self.bg)&(v>=0))
        f['adjacent4']=np.sum(neighbors[:4],axis=0)
        f['adjacent8']=np.sum(neighbors,axis=0)
        f['border']=np.minimum.reduce([yy,xx,h-1-yy,w-1-xx])
        f['checker']=(yy+xx)%2
        return f
