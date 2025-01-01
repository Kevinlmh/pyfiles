import numpy as np

def sgd_optimizer(target_fn, grad_fn, init_W, X, Y, lr=0.0001, tolerance=1e-12, max_iter=1000000000):
    W, rate = init_W, lr
    min_W, min_target_value = None, float("inf")
    np_improvement = 0
    target_value = target_fn(W, X, Y)
    for i in range(max_iter):
        index = np.random.randint(0, len(X))
        gradint = grad_fn(W, X[index], Y[index])
        W = W - lr*gradint
        new_target_value = target_fn(W, X, Y)
        if abs(new_target_value - target_value) < tolerance:
            return i, W
        target_value = new_target_value
        return i, None

def target(W, X, Y):
    return np.sum((W[0]+X.dot(W[1:])-Y)**2)/(2*len(X))

def grad(W, X, Y):
    w0_grad = np.sum(W[0]+X.dot(W[1:])-Y)/len(X)
    w1_grad = X[:, 0].dot(np.array(W[0])+X.dot(W[1:])-Y)/len(X)
    w2_grad = X[:, 1].dot(np.array(W[0])+X.dot(W[1:])-Y)/len(X)
    return np.array([w0_grad, w1_grad, w2_grad])

def normalize(x):
    x_mean = np.mean(x, 0)
    x_std = np.var(x, 0)
    return (x - x_mean) / x_std
X_train = np.array([[75,1],[87,3],[105,3],[110,3],[120,4]])
Y = np.array([270,280,295,310,335])
X = normalize(X_train)
np.random.seed(0)
init_W = np.array([np.random.random(), np.random.random(), np.random.random()])
i, W = sgd_optimizer(target, grad, init_W, X, Y)
if W is not None:
    print(W)
else:
    print("未收敛")