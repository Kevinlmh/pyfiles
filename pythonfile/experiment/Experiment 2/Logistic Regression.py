import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


#加载数据
train_data = pd.read_csv('./train.txt',header=None, delimiter="\t")
test_data = pd.read_csv('./test.txt',header=None, delimiter="\t")

# 分离训练数据的特征和目标
X_train = train_data.iloc[:, :-1].values  # 前23列为特征
Y_train = train_data.iloc[:, -1].values   # 第24列为目标

X_test = test_data.values  # 测试数据

# 数据预处理：标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 超参数网格
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],                       # 正则化强度
    "penalty": ["l1", "l2"],                            # 正则化方式
    "solver": ["liblinear", "saga", "lbfgs"],           # 优化算法
    "max_iter": [100, 200]                              # 最大迭代次数
}

# 超参数网格搜索最佳模型
logreg = LogisticRegression()
grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, Y_train)

# 获取最佳模型进行测试
best_model = grid_search.best_estimator_
Y_pred = best_model.predict(X_test_scaled)

# 统计F1分数
Y_test_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]  # 类别为1的概率