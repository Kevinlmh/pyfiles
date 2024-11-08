import numpy as np
import matplotlib.pyplot as plt
def linreg_matrix( x, y ) :
    X_X_T = np.matmul( x, x.T )
    X_X_T_1 = np.linalg.inv( X_X_T )
    X_X_T_1_X = np.matmul( X_X_T_1, x )
    X_X_T_1_X_Y = np.matmul( X_X_T_1_X, y )
    return X_X_T_1_X_Y

xTrain1 = np.array([[75], [87], [105], [110], [120]])

xTrain_dict = {i: np.concatenate([xTrain1 ** j for j in range(1, i + 1)], axis=1) for i in range(1, 9)}

yTrain = np.array([270, 280, 295, 310, 335])[:,np.newaxis]

# for i in range(1, 9):
#     xTrain = xTrain_dict[i]
#     print("x=\n", xTrain)
def make_ext(x):
    ones_col = np.ones((x.shape[0], 1))  # 创建一个列向量
    return np.hstack([ones_col, x])      # 水平拼接


# for i in range(1, 9):
#     xTrain = xTrain_dict[i]
#     xTrain = make_ext( xTrain )
#     print("x=", xTrain.T)
#     print("y=", yTrain)
#     w = linreg_matrix(xTrain.T, yTrain)
#     print(("w=\n", w))

#设置阶数
degree = 3
xTrain = xTrain_dict[degree]
xTrain = make_ext(xTrain)

# 计算回归系数
w = linreg_matrix(xTrain.T, yTrain)
print("拟合的系数 w=\n", w)

# 绘制二阶多项式拟合曲线
x_vals = np.linspace(70, 130, 100).reshape(-1, 1)  # 为了绘制曲线生成更平滑的 x 轴点
x_vals_poly = np.concatenate([x_vals ** j for j in range(1, degree + 1)], axis=1)
x_vals_poly = make_ext(x_vals_poly)

# 计算预测值
y_pred = np.matmul(x_vals_poly, w)

# 绘制实际数据点和拟合曲线
plt.scatter(xTrain1, yTrain, color='blue', label='Data Points')  # 实际数据点
plt.plot(x_vals, y_pred, color='red', label=f'Polynomial Degree {degree}')  # 拟合曲线
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'{degree}-degree Polynomial Fit')
plt.legend()
plt.grid(True)
plt.show()