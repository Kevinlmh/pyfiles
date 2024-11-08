scores = {"chen":[80,90],"zhao":[90,80]}
names = ["chen","zhao"]
scores["wang"] = scores["chen"].copy()
names.append('wang')
scores["chen"][0] = 100
#print(scores)

names.sort(key = lambda x:sum(scores[x]))
for name in names:
    print(scores[name][0],end=" ")
#print(scores)
#print(names)