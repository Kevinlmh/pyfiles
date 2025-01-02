import numpy as np
import matplotlib.pyplot as plt

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

def normalize(X, mean, std):
    return (X - mean) / std
def make_ext(x):
    ones = np.ones(1)[:, np.newaxis]
    new_x = np.insert(x, 0, ones, axis=1)
    return new_x
def logistic_fun(z):
    return 1./(1+np.exp(-z))
def cost_fun(w, X, y):
    tmp = logistic_fun(X.dot(w))
    cost = -y.dot(np.log(tmp)-(1-y).dot(np.log(1-tmp)))
    return cost
def grad_fun(w, X, y):
    loss = X.T.dot(logistic_fun(X.dot(w))-y)/len(X)
    return loss
xTrain = np.array([[3.36,78],[2.7,75],[2.9,80],[3.12,100],[2.8,125],[3.32,94],[3.05,120],[3.7,160],[3.52,170],[3.6,155]])
yTrain = np.array([1,1,1,1,1,0,0,0,0,0])
mean = xTrain.mean(axis=0)
std = xTrain.std(axis=0)
xTrain_norm = normalize(xTrain, mean, std)
np.random.seed(0)
init_W = np.random.random(3)
xTrain_ext = make_ext(xTrain_norm)
iter_count, w = bgd_optimizer(cost_fun, grad_fun, init_W, xTrain_ext, yTrain, lr = 0.001, tolerance=1e-5, max_iter=100000000)
w0, w1, w2 = w
print("迭代次数：", iter_count)
print("参数w0， w1， w2的值：", w0, w1, w2)
def initPlot():
    plt.figure()
    plt.title('House Price vs House Area')
    plt.xlabel('House Price')
    plt.ylabel('House Area')
    plt.grid(True)
    return plt
plt = initPlot()
plt.plot(xTrain[:5, 0], xTrain[:5, 1], 'k+')
plt.plot(xTrain[5:, 0], xTrain[5:, 1], 'ro')
x1 = np.array([2.6, 3.3, 4.0])
x1_norm = (x1 - mean[0]) / std[0]
x2_norm = - (w0 + w1 * x1_norm) / w2
x2 = std[1] * x2_norm + mean[1]
plt.plot(x1, x2, 'b-')
plt.show()