import tensorflow as tf
import numpy as np
from keras import datasets, models, layers
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# 加载数据集
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# 数据归一化
train_images = train_images.reshape((60000, 28*28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28*28)).astype('float32') / 255

# 标签进行one-hot编码
onehot_encoder = OneHotEncoder(sparse_output=False)
train_labels = train_labels.reshape(60000, 1)
test_labels = test_labels.reshape(10000, 1)
train_labels = onehot_encoder.fit_transform(train_labels)
test_labels = onehot_encoder.fit_transform(test_labels)

# 创建神经网络模型
neural_net = models.Sequential()
neural_net.add(layers.Dense(units=512, activation='relu', input_shape=(28*28,)))
neural_net.add(layers.Dense(units=10, activation='softmax'))
neural_net.summary()

# 编译模型
neural_net.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), 
                   loss='categorical_crossentropy', 
                   metrics=['accuracy'])

# 训练模型
history = neural_net.fit(train_images, train_labels, epochs=50, batch_size=512)

# 绘制损失函数和准确率曲线
epoch_list = list(range(0, 50))
loss_list = history.history['loss']
accuracy_list = history.history['accuracy']

plt.plot(epoch_list, loss_list, label="loss")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

plt.plot(epoch_list, accuracy_list, label="accuracy")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# 测试模型
test_loss, test_acc = neural_net.evaluate(test_images, test_labels)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)
