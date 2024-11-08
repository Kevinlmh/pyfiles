import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体为SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取乳腺癌数据集
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

# 数据提取
X = df.drop('Target', axis=1)
y = df['Target']

# 检查是否有缺失值并进行众数填充
if X.isnull().sum().sum() > 0:
    for column in X.columns:
        if X[column].isnull().sum() > 0:  # 检查列是否有缺失值
            X[column] = X[column].fillna(X[column].mode()[0])  # 用众数填充缺失值

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 构建随机森林回归模型
rfc_model = RandomForestClassifier(random_state=42)

# 设置超参数网格
param_grid = {
    'n_estimators': list(range(1, 100, 5)),
    'max_depth': [None] + list(range(1, 21, 1))
}

# 网格搜索
grid_search = GridSearchCV(rfc_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳参数：", grid_search.best_params_)
print("训练集得分：", grid_search.score(X_train, y_train))

# 使用最佳参数重新构建模型并用测试集测试
best_model = grid_search.best_estimator_
print("测试集得分为：", best_model.score(X_test, y_test))

# 特征重要性分析
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

# 检查特征重要性是否有无效值并处理
importances = np.nan_to_num(importances)  # 将 NaN 值替换为 0
indices = np.argsort(importances)[::-1]  # 重新获取排序索引

# 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=np.array(data.feature_names)[indices])
plt.title('乳腺癌特征重要性分析')
plt.show()