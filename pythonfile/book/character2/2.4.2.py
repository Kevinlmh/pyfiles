import numpy as np
A = np.array([[3.5,-4,5,6,-7],[1,0,0,0,-5],[0,0,10,11,-12],[0,8,9,1,0],[16,-1,1,1,-7]])
print(A)
B = np.array([8,18,60,1,9])
print(B)
A_inv = np.linalg.inv(A)
x = np.dot(A_inv,B)
print(x)