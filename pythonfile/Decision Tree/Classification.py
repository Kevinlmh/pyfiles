from sklearn.tree import DecisionTreeClassifier, export_graphviz
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import pandas as pd
import graphviz
import seaborn as sns

pd.set_option('future.no_silent_downcasting', True)

# 读取泰坦尼克号生存者数据集
data = pd.read_csv('Decision Tree\\Titanic.csv')

# 特征和标签
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = data[features]  # 特征
y = data['Survived']  # 标签

# 数据预处理
X.loc[:, 'Sex'] = X['Sex'].map({'male': 0, 'female': 1})
X.loc[:, 'Embarked'] = X['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})

# 检查是否有缺失值并进行众数填充
if X.isnull().sum().sum() > 0:
    for column in X.columns:
        if X[column].isnull().sum() > 0:
            X.loc[:, column] = X[column].fillna(X[column].mode()[0])  # 用众数填充缺失值

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 设置参数网格
param_grid = {
    'criterion': ('gini', 'entropy'),
    'splitter': ('best', 'random'),
    'max_depth': list(range(1, 20, 1)),
    'min_samples_leaf': list(range(1, 10, 1)),
    'min_samples_split': list(range(2, 10, 1))  
}

# 网格搜索最佳模型
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')

# 训练模型并输出得分
grid_search.fit(X_train, y_train)
print("最佳参数：", grid_search.best_params_)
print("最佳模型分类得分：", grid_search.best_score_)

# 在测试集上评估
best_model = grid_search.best_estimator_ # 获取最佳模型
print("测试集得分：", best_model.score(X_test, y_test))

#可视化决策树
class_names = y.unique().astype(str)  # 获取唯一类别名称
dot_data = export_graphviz(best_model,feature_names=features,class_names=class_names,filled=True,rounded=True,fontname='SimSun')
graph = graphviz.Source(dot_data)  
graph.view() 

# 特征重要性分析
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

# 设置中文字体为SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=np.array(features)[indices])
plt.title('人员存活特征重要性分析')
plt.show()