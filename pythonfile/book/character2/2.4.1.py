import numpy as np
def merge(inter):
    inter.sort(key = lambda x:x[0])
    merged = [inter[0]]
    for cur in inter[1:]:
        pre = merged[-1]
        if cur[0]<=pre[1]:
            merged[-1]=[pre[0],max(pre[1],cur[1])]
        else:
            merged.append(cur)
    return merged
intervals = [[2,6],[1,3],[8,10],[15,18]]
intervals = merge(intervals)
print(intervals)