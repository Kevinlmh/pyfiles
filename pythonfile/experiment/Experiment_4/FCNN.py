import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras import models, layers, optimizers

# 读取数据（假设数据集为 CSV 格式）
# 假设数据集文件名为 'auto_mpg.csv'
df = pd.read_csv('auto_mpg.csv')

# 数据预处理：处理缺失值
df = df.dropna()  # 删除缺失值行，也可以考虑使用填充方法

# 特征列（假设模型目标值是“每加仑燃油公里数”）
features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']
target = 'mpg'

# 提取特征和目标变量
X = df[features]
y = df[target]

# 类别特征处理：将'origin'列转换为 one-hot 编码
X = pd.get_dummies(X, columns=['origin'], drop_first=True)

# 数据集拆分：80% 训练，20% 测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化数据（神经网络训练通常需要标准化数据）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 搭建神经网络模型
model = models.Sequential()
model.add(layers.Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))  # 输入层，64个神经元
model.add(layers.Dense(32, activation='relu'))  # 隐藏层，32个神经元
model.add(layers.Dense(1))  # 输出层，1个神经元（回归问题）

# 编译模型
model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# 训练模型并记录误差
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=1)

# 绘制训练过程中的误差图
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss over Epochs')
plt.legend()
plt.show()

# 使用训练好的模型进行预测
y_pred = model.predict(X_test_scaled)

# 计算测试数据的 R² 和 MSE
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R² Score: {r2}')
