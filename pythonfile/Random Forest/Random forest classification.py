from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd

# 读取泰坦尼克人员存活数据集
titanic = pd.read_csv('Random Forest/Titanic.csv')

# 数据的清洗、填充以及预处理
titanic['Age'] = titanic["Age"].fillna(titanic["Age"].median()) # 年龄字段的缺失用均值填充
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0   # 性别男用0替代
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1 # 性别女用1替代
titanic["Embarked"] = titanic["Embarked"].fillna("S") # embarked字段缺失用S填充
titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median()) # Fare字段用均值填充

# 将列Embarked中出现的S、C、Q用0、1、2来代替
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

# 特征和标签
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = titanic[features]  # 特征
y = titanic['Survived']  # 标签

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 设置超参数网格
param_grid = {
    'n_estimators': list(range(1, 100, 5)),
    'max_depth': [None] + list(range(1, 50, 3))
}

# 网格搜索最佳参数组合
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 输出最佳参数和训练集得分
print("最佳参数：", grid_search.best_params_)
print("训练集得分：", grid_search.score(X_train, y_train))

# 使用最佳参数训练模型
best_model = grid_search.best_estimator_

#查看特征的权重
print("特征及权重:", *zip(features, best_model.feature_importances_))

# 测试集分类结果
predicted_class = best_model.predict(X_test)
print("分类结果：", predicted_class)
print("测试集得分：", best_model.score(X_test, y_test))