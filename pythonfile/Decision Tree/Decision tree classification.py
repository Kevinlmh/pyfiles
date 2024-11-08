from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import graphviz

iris = pd.read_csv('Decision Tree\Iris.csv', header=None)  #加载鸢尾花数据集
print(iris.shape)
col_name = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
iris.columns = col_name

X = iris[col_name[:-1]]  #特征
y = iris['class']  #标签

#分割训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

#设置参数网格
param_grid = {
    'max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6, 8]
}

#使用 GridSearchCV 进行参数优化
clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=30, splitter='random')
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)  # 训练模型

#输出训练数据的最佳参数和得分
print("最佳参数:", grid_search.best_params_)
print("最佳交叉验证得分:", grid_search.best_score_)

#在测试集上评估性能
best_clf = grid_search.best_estimator_  #获取最佳模型
test_score = best_clf.score(X_test, y_test)  # 在测试集上评估模型
print("测试集得分:", test_score)

#可视化决策树
feature_names = col_name[:-1]  # 特征名称
class_names = iris['class'].unique()  # 获取唯一类别名称
dot_data = tree.export_graphviz(best_clf,feature_names=feature_names,class_names=class_names,filled=True,rounded=True,fontname='SimSun')
graph = graphviz.Source(dot_data)  
graph.view()  

#查看特征的权重
print("特征及权重:", *zip(feature_names, best_clf.feature_importances_))

#返回测试样本的分类结果
predicted_classes = best_clf.predict(X_test) 
print("分类结果:", predicted_classes)
