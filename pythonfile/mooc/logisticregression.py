import numpy as np
import matplotlib.pyplot as plt

# 批量下降函数
def bgd_optimizer(cost_fun, grad_fun, init_W, X, Y, lr = 0.0001, tolerance = 1e-12, max_iter = 100000000):
    W = init_W
    target_value= cost_fun(W, X, Y)
    for i in range(max_iter):
        grad=grad_fun(W, X, Y)
        next_W = W - grad * lr
        next_target_value = cost_fun(next_W, X, Y)
        if abs(next_target_value - target_value) < tolerance:
            return i, next_W
        else : W, target_value = next_W, next_target_value
    return i, None

# 数据归一化
def normalize(X, mean, std):
    return (X - mean) / std
# 对x进行扩展，加入一个全1的列
def make_ext(x):
    ones = np.ones(1)[:, np.newaxis]
    new_x = np.insert(x, 0, ones, axis=1)
    return new_x
def logistic_fun(z):
    return 1./(1+np.exp(-z))
# 计算损失
def cost_fun(w, X, y):
    tmp = logistic_fun(X.dot(w))
    cost = -y.dot(np.log(tmp)-(1-y).dot(np.log(1-tmp)))
    return cost
# 计算梯度
def grad_fun(w, X, y):
    loss = X.T.dot(logistic_fun(X.dot(w))-y)/len(X)
    return loss
# 导入数据
xTrain = np.array([[78, 3.36], [75, 2.7], [80, 2.9], [100, 3.12], [125, 2.8], [94, 3.32], [120, 3.05], [160, 3.7], [170, 3.52], [155, 3.6]])
xTest = np.array([[100, 3], [93, 3.25], [163, 3.63], [120, 2.82], [89, 3.37]])
yTrain = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
yTest = np.array([1, 0, 1, 1, 1])
# 数据处理及初始化
mean = xTrain.mean(axis=0)
std = xTrain.std(axis=0)
xTrain_norm = normalize(xTrain, mean, std)
xTrain_ext = make_ext(xTrain_norm)
# 初始化权重向量
np.random.seed(0)
init_W = np.random.rand(3)
# 梯度下降计算
iter_count, w = bgd_optimizer(cost_fun, grad_fun, init_W, xTrain_ext, yTrain, lr = 0.001, tolerance = 1e-5, max_iter = 100000000)
w0, w1, w2 = w
print("迭代次数:", iter_count)
print("参数w0, w1, w2的值:", w0, w1, w2)
# 绘制训练数据散点图
plt.figure()
plt.title('TrainData:House Price vs House Area')
plt.xlabel('House Price')
plt.ylabel('House Area')
plt.grid(True)
plt.plot(xTrain[:5, 0], xTrain[:5, 1], 'k+')
plt.plot(xTrain[5:, 0], xTrain[5:, 1], 'ro')
# 绘制分类分割线
x_values = np.linspace(xTrain[:, 0].min(), xTrain[:, 0].max(), 100)  # 取房价范围内的均匀分布点
x_values_norm = (x_values - mean[0]) / std[0]  # 对房价数据归一化
decision_boundary = -(w0 + w1 * x_values_norm) / w2  # 计算对应的面积
# 将归一化后的面积还原为原始值
decision_boundary_original = decision_boundary * std[1] + mean[1]
plt.plot(x_values, decision_boundary_original, 'b-', label='Decision Boundary')
plt.show()

# 对测试数据进行归一化和扩展
xTest_norm = normalize(xTest, mean, std)
xTest_ext = make_ext(xTest_norm)
# 使用训练好的权重预测测试数据的分类
y_pred_probs = logistic_fun(xTest_ext.dot(w))  # 计算预测概率
y_pred = (y_pred_probs >= 0.5).astype(int)  # 根据概率分类
# 绘制测试数据预测结果
plt.figure()
plt.title('TestData: Predictions and Decision Boundary')
plt.xlabel('House Price')
plt.ylabel('House Area')
plt.grid(True)

# 根据预测结果绘制不同的颜色
for i in range(len(xTest)):
    if y_pred[i] == 1:
        plt.plot(xTest[i, 0], xTest[i, 1], 'k+', label='Predicted Class 1' if i == 0 else "")
    else:
        plt.plot(xTest[i, 0], xTest[i, 1], 'ro', label='Predicted Class 0' if i == 0 else "")

# 绘制分类分割线
plt.plot(x_values, decision_boundary_original, 'b-', label='Decision Boundary')

plt.legend()
plt.show()

# 输出预测结果
print("测试数据预测标签:", y_pred)
print("实际标签:", yTest)