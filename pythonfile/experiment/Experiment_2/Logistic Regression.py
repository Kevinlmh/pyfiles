import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

#加载数据
train_data = pd.read_csv('train.txt',header=None, delimiter="\t")
test_data = pd.read_csv('test.txt',header=None, delimiter="\t")

# 分离数据的特征和目标
X_train = train_data.iloc[:, :-1].values  # 前23列为特征
Y_train = train_data.iloc[:, -1].values   # 第24列为目标

X_test = test_data.iloc[:, :-1].values  # 测试数据
Y_test = test_data.iloc[:, -1].values

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 超参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'max_iter': [100, 200],
    'solver': ['liblinear']
}

# 记录逻辑回归模型训练开始时间
start_time_logreg = time.time()

# 超参数网格搜索最佳模型
logreg = LogisticRegression()
grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, Y_train)

# 记录逻辑回归模型训练结束时间
end_time_logreg = time.time()

# 最佳模型预测
best_model = grid_search.best_estimator_
print(grid_search.best_estimator_)
y_pred = best_model.predict(X_test)

# 计算 F1 分数
f1 = f1_score(Y_test, y_pred)

print(f"F1 Score: {f1:.4f}")
print(f"逻辑回归模型训练时间: {end_time_logreg - start_time_logreg:.10f}秒\n")