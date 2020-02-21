#!/Users/mahe/anaconda3/bin/python

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def image_show(image_data):
    plt.figure()
    plt.imshow(image_data)
    plt.colorbar()
    plt.grid(False)
    plt.show()


fasion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fasion_mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

"""
plt.figure(figsize=[10, 10])
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.grid(False)
    plt.xlabel(y_train[i])
plt.show()
"""

"""
# create keras model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(10, activation='softmax')
])
print(model.summary())

# if y_train is 'one-hot' encoding, loss should use non-sparse version
# 'from_logits=True' will convert output to softmax, since we alreasy use
# 'softmax' activation when creating models, so we don't need to set here
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    # loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# start training
model.fit(x_train, y_train, epochs=10)

# save models so don't need to train it again and again
model.save('./mnist_classifier.h5')
"""

# load model
loaded_model = keras.models.load_model('./mnist_classifier.h5')

# evaluate on test set
test_loss, test_acc = loaded_model.evaluate(x_test, y_test, verbose=2)

# make predictions
prob_model = keras.Sequential([
    loaded_model,
    keras.layers.Softmax()
])

prediction = prob_model.predict(x_test)
print('predicted result:', np.argmax(prediction[0]),
      'actual result: ', y_test[0])
print("\nTest accuracy: ", test_acc)
