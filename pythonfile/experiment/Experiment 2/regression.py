import numpy as np

def bgd_optimizer(target_fn, grad_fn, init_W, X, Y, lr, max_iter = 100):
    W = init_W
    target_value= target_fn(W, X, Y)
    for i in range(max_iter):
        grad=grad_fn(W, X, Y)
        next_W = W - grad * lr
        next_target_value = target_fn(next_W, X, Y)
        #if abs(next_target_value - target_value) < tolerance:
        #    return i, next_W
        #else : 
        W, target_value = next_W, next_target_value
    return i, next_W

def target_function(W, X, Y):
    w0, w1 = W
    return np.sum((w0+w1*X-Y)**2)/(2*len(X))

def grad_function(W, X, Y):
    w0, w1 = W
    w0_grad = np.sum(w0+X*w1-Y)/len(X)
    w1_grad = X.dot(w0+X*w1-Y)/len(X)
    return np.array([w0_grad, w1_grad])

x = np.array([9.58,8.19,9.50,10.54,8.51,9.42,11.99,12.02,12.97,13.19,12.16,9.92,8.33,11.18,13.75,11.78,9.54,10.48,7.50,9.62],dtype=np.float64)
y = np.array([26.65,23.02,30.25,29.56,23.32,28.84,36.01,38.97,42.57,38.77,40.01,27.70,23.66,36.30,44.88,38.61,26.96,29.88,22.20,30.98],dtype=np.float64)
np.random.seed(42)
init_W=np.array([np.random.random(),np.random.random()])
for alpha in [0.01,0.1,1,10]:
    i, W = bgd_optimizer(target_function, grad_function, init_W, x, y, alpha)
    #if W is not None:
    w0, w1 = W
    print(f"迭代次数: %d, 最优的w0和w1:(%f, %f)" % (i, w0 ,w1))
    #else :print(f"达到最大迭代次数，未收敛")
