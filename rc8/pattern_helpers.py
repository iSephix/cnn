def views(a,k):
    if k&4:a=a.T
    if k&1:a=a[::-1]
    if k&2:a=a[:,::-1]
    return a
