import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from keras import models, layers, optimizers

# 加载数据
column_names = [
    "mpg", "cylinders", "displacement", "horsepower", "weight",
    "acceleration", "model_year", "origin", "car_name"
]
data = pd.read_csv('data.txt', delim_whitespace=True, names=column_names)

# 数据预处理
# 将 'horsepower' 列中的 '?' 转换为 NaN
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')

# 填充缺失值（horsepower 列中的 NaN 使用均值填充）
data['horsepower'] = data['horsepower'].fillna(data['horsepower'].mean())

# 删除 'car_name' 列（无意义）
data.drop(columns=['car_name'], inplace=True)

# 提取特征和目标值
target = data["mpg"].values  # 目标值
features = data.drop(columns=["mpg"])  # 去掉目标列

# 对 'origin' 列进行独热编码
features_encoded = pd.get_dummies(features, columns=['origin'], drop_first=True)

# 数据集拆分（后 20% 为测试集）
X_train, X_test, y_train, y_test = train_test_split(
    features_encoded, target, test_size=0.2, random_state=42
)

# 构建神经网络
model = models.Sequential([
    layers.Dense(64, input_dim=X_train.shape[1], activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  
])

model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# 训练模型
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=30, verbose=1)

# 绘制训练误差和验证误差
epoch_list = list(range(0, 100))
train_loss = history.history['loss']
test_loss = history.history['val_loss']

# 绘制训练误差和测试误差曲线
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Testing Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()
plt.grid(True)
plt.show()

# 模型评估
y_pred = model.predict(X_test)

# 计算 R² 和 MSE
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R² Score: {r2:.2f}')
