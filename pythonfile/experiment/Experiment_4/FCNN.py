import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from keras import models, layers, optimizers

# 加载数据
column_names = [
    "mpg", "cylinders", "displacement", "horsepower", "weight",
    "acceleration", "model_year", "origin", "car_name"
]
data = pd.read_csv('data.txt', delim_whitespace=True, names=column_names)

# 数据预处理
data = data.replace('?', np.nan)
data = data.dropna()  

# 提取特征和目标值
target = data["mpg"].values  # 每加仑燃油公里数
features = data.drop(columns=["mpg", "car_name"])  # 去掉目标列和车型名称

# 对类别特征（origin列）进行 One-Hot 编码
categorical_feature = features["origin"].values.reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)
categorical_encoded = encoder.fit_transform(categorical_feature)

# 数值特征处理
numerical_features = features.drop(columns=["origin"])
scaler = StandardScaler()
numerical_features_scaled = scaler.fit_transform(numerical_features)

# 合并数值特征和编码后的类别特征
processed_features = np.hstack((numerical_features_scaled, categorical_encoded))

# 数据集拆分（后 20% 为测试集）
X_train, X_test, y_train, y_test = train_test_split(
    processed_features, target, test_size=0.2, random_state=42
)

# 构建神经网络
model = models.Sequential([
    layers.Dense(64, input_dim=X_train.shape[1], activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # 输出层，回归问题只有一个神经元
])

model.compile(optimizer=optimizers.SGD(learning_rate=0.001), loss='mean_squared_error')

# 训练模型
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=30, verbose=1)

# 绘制训练误差和验证误差
epoch_list = list(range(0, 100))
train_loss = history.history['loss']
test_list = history.history['val_loss']

plt.plot(epoch_list, train_loss, label="Training Loss")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Training Loss per Epoch')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

plt.plot(epoch_list, test_list, label="Testing Loss")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Testing Loss per Epoch')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# 模型评估
y_pred = model.predict(X_test)

# 计算 R² 和 MSE
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R² Score: {r2:.2f}')
