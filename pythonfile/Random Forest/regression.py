import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体为SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载加州房价数据集
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

# 数据提取
X = df.drop('Target', axis=1)
y = df['Target']

# 检查是否有缺失值并进行众数填充
if X.isnull().sum().sum() > 0:
    for column in X.columns:
        if X[column].isnull().sum() > 0:  # 检查列是否有缺失值
            X[column].fillna(X[column].mean(), inplace=True)  # 用均值填充缺失值
            
# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 构建随机森林回归模型
rfr_model = RandomForestRegressor(random_state=42)

# 设置超参数网格
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 10, 20, 30]
}

# 网格搜索
grid_search = GridSearchCV(rfr_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 使用最佳参数重新构建模型
best_model = grid_search.best_estimator_

# 输出最佳参数及测试集均分方差
print("最佳参数：", grid_search.best_params_)
print("最佳均分方差：", -grid_search.best_score_)

# 十折交叉验证
cv_scores = cross_val_score(best_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
print(f"十折交叉验证平均MSE: {-np.mean(cv_scores)}")

# 在测试集上预测并计算评估指标
y_pred = best_model.predict(X_test)
mse_val = mean_squared_error(y_test, y_pred)
r2_val = r2_score(y_test, y_pred)

print(f"MSE (测试集): {mse_val}")
print(f"R^2 (测试集): {r2_val}")

# 绘制回归图
plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={'s':10}, line_kws={'color':'red'})
plt.xlabel("真实值")
plt.ylabel("预测值")
plt.title("测试集-回归图")
plt.show()

# 特征重要性分析
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

# 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=np.array(data.feature_names)[indices])
plt.title('加州房价特征重要性分析')
plt.show()
