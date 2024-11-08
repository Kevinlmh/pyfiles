import  numpy as np
N = 4
a = np.zeros([6,6])
for i in range(N):
    a[i][0] = 1
for i in range(N):
    for j in range(i):
        a[i][j] = a[i-1][j-1] + a[i-1][j]
for i in range(N):
    for j in range(i):
        print(a[i][j], end=" ")
    print()