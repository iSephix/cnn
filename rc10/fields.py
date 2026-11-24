"""Spatial measurements for learning conditional paint programs from examples."""
from itertools import product
import numpy as np
from scipy import ndimage as nd


def shift(g, dy, dx):
    h,w = g.shape
    p = np.pad(g, ((max(0,-dy),max(0,dy)),(max(0,-dx),max(0,dx))), constant_values=-1)
    return p[max(0,dy):max(0,dy)+h,max(0,dx):max(0,dx)+w]


def local(g):
    bg = np.bincount(g.ravel()).argmax()
    h,w = g.shape
    near = []
    for dy,dx in sorted(product(range(-2,3),repeat=2), key=lambda p: abs(p[0])+abs(p[1])):
        v = shift(g,dy,dx)
        yield v
        if max(abs(dy),abs(dx))==1:
            near.append(v)
    neighbors = np.stack(near)
    for c in range(10):
        yield (neighbors==c).sum(axis=0)
    yield (neighbors==g).sum(axis=0)
    yield (neighbors!=bg).sum(axis=0)
    r,c = np.indices(g.shape)
    yield from (r,c,h-1-r,w-1-c, np.minimum.reduce((r,c,h-1-r,w-1-c)))
    for period in (2,3,4):
        yield from (r%period,c%period,(r+c)%period,(r-c)%period)


def relative(g):
    y,x = np.indices(g.shape)
    for c in range(10):
        mark = g==c
        row,col = np.where(mark)
        missing = np.full_like(g,-32)
        if not len(row):
            yield from [missing]*12
            continue
        yield nd.distance_transform_cdt(~mark,metric='taxicab')
        yield nd.distance_transform_cdt(~mark,metric='chessboard')
        yield from (y-row.min(),row.max()-y,x-col.min(),col.max()-x)
        for axis in (0,1):
            yield np.broadcast_to(mark.sum(axis=axis,keepdims=True),g.shape)
            yield mark.cumsum(axis=axis)
            yield np.flip(np.flip(mark,axis=axis).cumsum(axis=axis),axis=axis)


def components(g):
    y,x = np.indices(g.shape)
    bg = np.bincount(g.ravel()).argmax()
    for diagonal in (False,True):
        kernel = nd.generate_binary_structure(2,2 if diagonal else 1)
        for mono in (True,False):
            # Include background components, which can encode holes and enclosures.
            planes = [np.zeros_like(g) for _ in range(14)]
            masks = [g==c for c in np.unique(g)] if mono else [g!=bg,g==bg]
            for mask in masks:
                labels,_ = nd.label(mask,kernel)
                for i,box in enumerate(nd.find_objects(labels),1):
                    region = labels[box]==i
                    yy,xx = np.where(region)
                    h,w = region.shape
                    vals = g[box][region]
                    boundary = nd.binary_dilation(labels==i,kernel) & (labels!=i)
                    outer = g[boundary]
                    surround = int(np.bincount(outer).argmax()) if outer.size else -1
                    stats = (len(yy),h,w,h*w-len(yy),yy,xx,h-1-yy,w-1-xx,
                             min(box[0].start,box[1].start,g.shape[0]-box[0].stop,g.shape[1]-box[1].stop),
                             len(set(vals)),len(set(outer)),surround,
                             int(np.bincount(vals).argmax()),int(np.array_equal(region,region[::-1])))
                    for p,v in zip(planes,stats):
                        p[box][region] = v
            yield from planes
