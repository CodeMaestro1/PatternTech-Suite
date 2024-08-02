from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from plots import plot_some_data, plot_some_predictions

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Scale these values to a range of 0 to 1 before feeding them to the neural network model.

train_images = train_images / 255.0
test_images = test_images / 255.0

num_train_images = train_images.shape[0]
num_test_images = test_images.shape[0]
height = 28
width = 28
channels = 1  # grayscale images have 1 channel

train_images_reshaped = train_images.reshape(num_train_images, height, width, channels)
test_images_reshaped = test_images.reshape(num_test_images, height, width, channels)

# Build the model
# Building the neural network requires configuring the layers of the model, then compiling the model.
model = Sequential([
    Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
    BatchNormalization(),
    keras.layers.ReLU(),

    Conv2D(32, (3, 3), padding='same'),
    BatchNormalization(),
    keras.layers.ReLU(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), padding='same'),
    BatchNormalization(),
    keras.layers.ReLU(),

    Conv2D(64, (3, 3), padding='same'),
    BatchNormalization(),
    keras.layers.ReLU(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(),
    keras.layers.ReLU(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(200),
    BatchNormalization(),
    keras.layers.ReLU(),
    Dropout(0.5),

    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(train_images_reshaped, train_labels, epochs=50, validation_data=(test_images_reshaped, test_labels))

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images_reshaped, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# Make predictions
# With the model trained, you can use it to make predictions about some images.
# The model's linear outputs, logits.
# Attach a softmax layer to convert the logits to probabilities, which are easier to interpret.
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images_reshaped)

plot_some_predictions(test_images, test_labels, predictions, class_names)
