import numpy as np

def bgd_optimizer(target_fn, grad_fn, init_W, X, Y, lr = 0.0001, tolerance = 1e-12, max_iter = 100000000):
    W = init_W
    target_value= target_fn(W, X, Y)
    for i in range(max_iter):
        grad=grad_fn(W, X, Y)
        next_W = W - grad * lr
        next_target_value = target_fn(next_W, X, Y)
        if abs(next_target_value - target_value) < tolerance:
            return i, next_W
        else : W, target_value = next_W, next_target_value
    return i, None

def target_function(W, X, Y):
    w0, w1 = W
    return np.sum((w0+w1*X-Y)**2)/(2*len(X))
def grad_function(W, X, Y):
    w0, w1 = W
    w0_grad = np.sum(w0+X*w1-Y)/len(X)
    w1_grad = X.dot(w0+X*w1-Y)/len(X)
    return np.array([w0_grad, w1_grad])
x = np.array([80, 90, 111, 120, 120],dtype=np.float64)
y = np.array([280, 290, 311, 325, 335],dtype=np.float64)
np.random.seed(0)
init_W=np.array([np.random.random(),np.random.random()])
i, W = bgd_optimizer(target_function, grad_function, init_W, x, y)
if W is not None:
    w0, w1 = W
    print(f"迭代次数: %d, 最优的w0和w1:(%f, %f)" % (i, w0 ,w1))
else :print(f"达到最大迭代次数，未收敛")

x_test = np.array([75, 87, 105], dtype=np.float64)
y_test = np.array([270, 280, 295], dtype=np.float64)
y_pred = w0 + w1 * x_test
print(f"测试数据的预测售价: {y_pred}")

# 计算R方
y_true = y_test
y_mean = np.mean(y_true)
ss_total = np.sum((y_true - y_mean) ** 2)
ss_residual = np.sum((y_true - y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)
print(f"R²值: {r2}")