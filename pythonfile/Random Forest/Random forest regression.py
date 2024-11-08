from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# 读取葡萄酒质量数据集
Wine_data = pd.read_csv('Random Forest\\WineQT.csv')
col_name = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality','Id']
Wine_data.columns = col_name

# 特征和结果
X = Wine_data.drop(['Id', 'quality'], axis=1)  # 特征
y = Wine_data['quality']  # 结果

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 设置超参数网格
param_grid = {
    'n_estimators': list(range(1, 100, 5)),
    'max_depth': [None] + list(range(1, 50, 3))
}

# 网格搜索最佳参数组合
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train) # 训练模型

# 输出最佳参数和训练集得分
print("最佳参数：", grid_search.best_params_)
print("训练集最佳交叉验证均方误差：", -grid_search.best_score_)

# 使用最佳参数训练模型
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test) # 在测试集上评估模型
print("测试集得分（R^2）：", test_score)

# 进行交叉验证
_MSE = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error') # 负均方误差
R_2 = cross_val_score(best_model, X, y, cv=5) # R^2

# 输出交叉验证得分
print("交叉验证得分：", _MSE)
print("交叉验证平均得分：", np.mean(_MSE))
print("R^2交叉验证结果:", R_2)

# 预测
y_pred = best_model.predict(X_test)

# 设置中文字体，解决乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号乱码问题

# 可视化实际值与预测值
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', s=10, label='预测值')  # 预测值
plt.scatter(y_test, y_test, color='red', edgecolor='k', s=10, label='实际值')  # 实际值的散点图
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='green', linewidth=2, label='理想预测线')  # 理想预测线
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('葡萄酒质量实际值与预测值的比较')
plt.legend()
plt.grid()
plt.show()