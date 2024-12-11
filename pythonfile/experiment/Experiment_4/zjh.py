# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import optimizers
from matplotlib import rcParams

# 设置中文显示和图表的字体大小
rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
rcParams['axes.unicode_minus'] = False   # 负号显示正常
rcParams['font.size'] = 18  # 字体大小
# 定义列名
columns = ["mpg", "cylinders", "displacement", "horsepower", "weight",
           "acceleration", "model year", "origin", "car name"]
# 读取数据
data = pd.read_csv('data.txt', sep='\s+', names=columns, na_values="?")
# 数据预处理
# 填充缺失值（horsepower列中的缺失值使用均值填充）
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
data['horsepower'] = data['horsepower'].fillna(data['horsepower'].mean())
# 删除 car name 列，因为它对回归预测无意义
data.drop(columns=['car name'], inplace=True)
# 分离特征和目标值
X = data.drop(columns=['mpg'])  # 特征
y = data['mpg']  # 目标值
# 对类别变量 'origin' 进行独热编码
X = pd.get_dummies(X, columns=['origin'], drop_first=True)
# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 划分训练集和测试集（80%训练集，20%测试集）
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# 搭建神经网络模型
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),  # 输入层
    layers.Dense(64, activation='relu'),  # 隐藏层1
    layers.Dense(32, activation='relu'),  # 隐藏层2
    layers.Dense(1)  # 输出层，用于回归任务
])
def fit():
    # 编译模型
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    # 训练模型并记录训练过程
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    return history
def plot():
    history = fit()
    # 评估模型
    y_pred = model.predict(X_test)
    # 计算测试集上的MSE和R2
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'R²: {r2:.4f}')
    # 绘制训练误差和测试误差曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='训练误差', color='blue')
    plt.plot(history.history['val_loss'], label='测试误差', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('均方误差')
    plt.title('训练过程中的训练误差和测试误差')
    plt.legend()
    plt.grid(True)
    plt.show()
plot()