import numpy as np

A = np.array([[1, 2, 3, 4],
              [2, 0, 6, 8],
              [3, 7, 1, 2],
              [8, 1, 1, 2]])

B = np.array([[11, 12, 13, 14],
              [12, 10, 16, 18],
              [13, 17, 11, 12],
              [18, 11, 10, 12]])

y = np.array([[1], [2], [3], [8]])

AT_y = np.dot(A.T, y)

x = np.dot(np.linalg.inv(B), AT_y)

print("计算得到的向量 x 为:")
print(x)
