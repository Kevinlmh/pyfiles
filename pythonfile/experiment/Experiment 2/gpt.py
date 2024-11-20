import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
np.random.seed(42)

# 数据
X = np.array([9.58, 8.19, 9.50, 10.54, 8.51, 9.42, 11.99, 12.02, 12.97, 
              13.19, 12.16, 9.92, 8.33, 11.18, 13.75, 11.78, 9.54, 10.48, 7.50, 9.62])
y = np.array([26.65, 23.02, 30.25, 29.56, 23.32, 28.84, 36.01, 38.97, 
              42.57, 38.77, 40.01, 27.70, 23.66, 36.30, 44.88, 38.61, 26.96, 29.88, 22.20, 30.98])

# 标准化数据
X_mean, X_std = np.mean(X), np.std(X)
X_norm = (X - X_mean) / X_std

# 梯度下降实现
def gradient_descent(X, y, alpha, iterations):
    m = len(X)
    w_0, w_1 = np.random.random(), np.random.random()  # 初始化参数
    #loss_history = []

    for _ in range(iterations):
        y_pred = w_0 + w_1 * X
        error = y_pred - y

        # 梯度计算
        grad_0 = np.sum(error) / m
        grad_1 = X.dot(error) / m

        # 参数更新
        w_0 -= alpha * grad_0
        w_1 -= alpha * grad_1

        # 记录损失
        #loss = np.sum(error**2) / (2 * m)
        #loss_history.append(loss)

    return w_0, w_1 #loss_history

# 不同学习率设置
learning_rates = [0.01, 0.1, 1, 10]
iterations = 100

# 存储结果
results = {}

for alpha in learning_rates:
    w_0, w_1 = gradient_descent(X_norm, y, alpha, iterations)
    y_pred = w_0 + w_1 * X_norm
    r2 = 1 - np.sum((y_pred - y)**2) / np.sum((y - np.mean(y))**2)
    results[alpha] = {
        "w_0": w_0,
        "w_1": w_1,
        "R^2": r2,
        "pre_y": y_pred
    }


# 打印模型参数和最终损失
for alpha, result in results.items():
    print(f"Learning Rate: {alpha}")
    print(f"  w_0: {result['w_0']:.4f}, w_1: {result['w_1']:.4f}")
    print(f"  R^2: {result['R^2']:.4f}")
    y_pred = result["pre_y"]
    print(y_pred)
    print("-" * 30)

