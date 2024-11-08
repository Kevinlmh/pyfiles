import numpy as np
from bgd_optimizer import bgd_optimizer
import matplotlib.pyplot as plt
def target_function(W, X, Y):
    w0, w1 = W
    return np.sum((w0+w1*X-Y)**2)/(2*len(X))
def grad_function(W, X, Y):
    w0, w1 = W
    w0_grad = np.sum(w0+X*w1-Y)/len(X)
    w1_grad = X.dot(w0+X*w1-Y)/len(X)
    return np.array([w0_grad, w1_grad])
x = np.array([75, 87, 105, 110, 120],dtype=np.float64)
y = np.array([270, 280, 295, 310, 335],dtype=np.float64)
np.random.seed(0)
init_W=np.array([np.random.random(),np.random.random()])
i, W = bgd_optimizer(target_function, grad_function, init_W, x, y)
if W is not None:
    w0, w1 = W
    print("迭代次数: %d, 最优的w0和w1:(%f, %f)" % (i, w0 ,w1))
else :print("达到最大迭代次数，未收敛")