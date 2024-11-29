import tensorflow as tf
import numpy as np
from keras import datasets, models, layers
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28*28))
train_images.astype('float32') / 255

test_images = test_images.reshape((60000, 28*28))
test_images.astype('float32') / 255

train_labels = train_labels.reshape(60000, 1)
test_labels = train_labels.reshape(10000, 1)
onehot_encoder = OneHotEncoder(sparse=False)
train_labels = onehot_encoder.fit_transform(train_labels)
train_labels = tf.squeeze(train_labels)
test_labels = onehot_encoder.fit_transform(test_labels)
test_labels = tf.squeeze(test_images)

neural_net = models.Sequential()
neural_net.add(layers.Dense(units=512, activation='relu', input_shape=(28*28, )))
neural_net.add(layers.Dense(units=10,activation='softmax'))
neural_net.summary()

neural_net.compile(optimizer=tf.keras.optimizer.SGD(learning_rate=0.1))
loss = 'mse'
metrics = ['accuracy']

history = neural_net.fit(train_images, train_labels, epochs=50, batch_size=512)
epoch_list = list(range(0,50))
loss_list = history.history['loss']
accuracy_list = history.history['accuracy']
pre_result = neural_net.predict(test_images)
pre_result = np.argmax(pre_result, axis=1)

plt.plot(epoch_list, loss_list, label = "loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc = 'upper left')
plt.show()

plt.plot(epoch_list, accuracy_list, label = "accuracy")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy'], loc = 'uppper left')
plt.show()

test_loss, test_acc = neural_net.evaluate(test_images, test_labels)
print(test_loss, test_acc)
