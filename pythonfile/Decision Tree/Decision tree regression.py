from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

# 加载波士顿房价数据集
boston_data = pd.read_csv('Decision Tree\\boston.csv')
col_name = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
boston_data.columns = col_name

X = boston_data[col_name[:-1]]  # 特征
y = boston_data['MEDV']  # 结果

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 设置参数网格
param_grid = {
    'max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6, 8]
}

# 使用 GridSearchCV 进行参数优化
regressor = DecisionTreeRegressor(random_state=30)
grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)  # 训练模型

# 输出训练数据的最佳参数和得分
print("最佳参数:", grid_search.best_params_)
print("最佳交叉验证结果:", -grid_search.best_score_)  # 取负值得到均方误差

# 在测试集上评估性能
best_regressor = grid_search.best_estimator_  # 获取最佳模型
test_score = best_regressor.score(X_test, y_test)  # 在测试集上评估模型
print("测试集得分（R^2）：", test_score)

# 交叉验证
cross = cross_val_score(best_regressor, X, y, cv=10, scoring='neg_mean_squared_error')  # 负均方误差
cross1 = cross_val_score(best_regressor, X, y, cv=10)  # R^2
print("负均方误差交叉验证结果:", cross)
print("R^2交叉验证结果:", cross1)

# 预测
y_pred = best_regressor.predict(X_test)

# 设置中文字体，解决乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号乱码问题

# 可视化实际值与预测值
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', s=100, label='预测值')  # 预测值
plt.scatter(y_test, y_test, color='red', edgecolor='k', s=100, label='实际值')  # 实际值的散点图
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='green', linewidth=2, label='理想预测线')  # 理想预测线
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('实际值与预测值的比较')
plt.legend()
plt.grid()
plt.show()