import numpy as np
from sklearn.linear_model import LinearRegression

xTrain = np.array([[80], [90], [111], [120], [120]])
yTrain = np.array([280, 290, 311, 325, 335])

model = LinearRegression()
model.fit(xTrain, yTrain)

xTest = np.array([[75], [87], [105]])
yTest = np.array([270, 280, 295])

yPre = model.predict(xTest)
print("预测的新房源数据总价：", yPre)
print("预测房价的R方：", model.score(xTest, yTest))