"""Classify objects by how often congruent shapes occur in the same scene."""
from collections import Counter
import numpy as np
from scipy import ndimage


def scene(grid, diagonal, rotations):
    background = np.bincount(grid.ravel()).argmax()
    objects = []
    for color in np.unique(grid):
        if color == background:
            continue
        labels, _ = ndimage.label(grid == color, ndimage.generate_binary_structure(2, diagonal))
        for index, box in enumerate(ndimage.find_objects(labels), 1):
            mask = labels[box] == index
            shapes = [mask]
            if rotations:
                shapes = [np.rot90(face,k) for face in (mask,mask[:,::-1]) for k in range(4)]
            key = min((s.shape,s.tobytes()) for s in shapes)
            objects.append((box,mask,key,color))
    return background, objects


def rules(task):
    demos = task['train']
    if any(np.shape(e['input']) != np.shape(e['output']) for e in demos):
        return
    grids = [np.array(e['input']) for e in demos + task['test']]
    targets = [np.array(e['output']) for e in demos]
    for diagonal in (1,2):
        for rotations in (False,True):
            scenes = [scene(g,diagonal,rotations) for g in grids]
            for feature in ('unique','count','shape','size'):
                table = {}
                outputs = []
                for i,(bg,objects) in enumerate(scenes):
                    grid = grids[i]
                    out = grid.copy()
                    counts = Counter(key for _,_,key,_ in objects)
                    for box,mask,key,color in objects:
                        attr = {'unique':counts[key]==1,'count':counts[key],
                                'shape':key,'size':int(mask.sum())}[feature]
                        if i < len(demos):
                            vals = np.unique(targets[i][box][mask])
                            if len(vals)!=1:
                                break
                            options = {int(vals[0])}
                            if vals[0]==color:
                                options.add('keep')
                            if vals[0]==bg:
                                options.add('erase')
                            table[attr] = table.get(attr,options) & options
                        choices = table.get(attr)
                        if not choices:
                            break
                        paint = color if 'keep' in choices else bg if 'erase' in choices else min(choices)
                        out[box][mask] = paint
                    else:
                        if i < len(demos) and np.any(out!=targets[i]):
                            break
                        if i >= len(demos):
                            outputs.append(out)
                        continue
                    break
                else:
                    yield 3 + int(rotations),outputs
