import numpy as np
from sklearn.linear_model import LinearRegression

X_train = np.array([[80, 2, 4], [87, 3, 5], [100, 3, 1], [110, 3, 8], [120, 4, 7]])
X_test = np.array([[80, 2, 6], [90, 3, 7], [96, 2, 5], [105, 3, 4]])
Y_train = np.array([270, 280, 295, 330, 335])
Y_test = np.array([280, 290, 300, 315])
model = LinearRegression()
model.fit(X_train, Y_train)
ytestpred = model.predict(X_test)
print(ytestpred)
print(model.coef_)
print(model.intercept_)
