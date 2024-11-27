import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# 加载数据
data = pd.read_csv('train.txt',header=None, delimiter="\t")

# 分离数据的特征和目标
X = data.iloc[:, :-1].values  # 前23列为特征
Y = data.iloc[:, -1].values   # 第24列为目标

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 对特征进行标准化

# 分割数据，后20%用于验证
X_train, X_val, y_train, y_val = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# 超参数网格：指定KNN分类器的搜索范围
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],  # 邻居数量
    'weights': ['uniform', 'distance'],  # 权重函数
    'metric': ['euclidean', 'manhattan', 'minkowski']  # 距离度量
}

# 记录KNN模型的训练开始时间
start_time_knn = time.time()

# 使用GridSearchCV进行超参数网格搜索
knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_knn.fit(X_train, y_train)

# 记录KNN模型的训练结束时间
end_time_knn = time.time()

# 输出最佳模型和对应的超参数
print("最佳KNN模型:")
print(grid_search_knn.best_estimator_)

# 使用最佳模型进行预测
best_knn_model = grid_search_knn.best_estimator_
y_pred = best_knn_model.predict(X_val)

# 计算TP, TN, FP, FN
TP = np.sum((y_val == 1) & (y_pred == 1)) 
TN = np.sum((y_val == 0) & (y_pred == 0))
FP = np.sum((y_val == 0) & (y_pred == 1))
FN = np.sum((y_val == 1) & (y_pred == 0))  

# 计算评估指标
accuracy = (TP + TN) / (TP + TN + FP + FN)  # 准确率
precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # 精准率
recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # 召回率

# 输出结果
print(f"准确率: {accuracy:.4f}")
print(f"精准率: {precision:.4f}")
print(f"召回率: {recall:.4f}")
print(f"KNN模型训练时间: {end_time_knn - start_time_knn:.10f}秒\n")