from __future__ import absolute_import, division, print_function, unicode_literals


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Input, Dense, Flatten
import pandas as pd


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from plots import plot_some_data, plot_some_predictions


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Scale these values to a range of 0 to 1 before feeding them to the neural network model. 
    ### YOUR CODE HERE
train_images = train_images / 255.0
test_images = test_images / 255.0

plot_some_data(train_images, train_labels, class_names)

# Build the model of dense neural network
# Building the neural network requires configuring the layers of the model, then compiling the model.
# Define the input layer based on the shape of the images
# Then define two dense layers. 
# The hidden layer with 128 neurons and RELU activation 
# The output layer with 10 neurons and linear activation.

model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Assuming the images are 28x28 pixels
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # Assuming there are 10 classes
])

#optimizer_list = ['adam', 'sgd', 'rmsprop', 'adamax', 'nadam', 'ftrl']
optimizer_list = ['adam']


# Compile the model
# Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:
# Loss function —This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
# Optimizer —This is how the model is updated based on the data it sees and its loss function.


# Loop over each optimizer
for optimizer in optimizer_list:
    # Compile the model with the current optimizer
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    #Train the model
    # Training the neural network model requires the following steps:

    #   1. Feed the training data to the model. In this example, the training data is in the train_images and train_labels arrays.
    #   2. The model learns to associate images and labels.
    #   3. You ask the model to make predictions about a test set—in this example, the test_images array.
    #   4. Verify that the predictions match the labels from the test_labels array.
    
    # Train the model and store the history
    model.fit(train_images, train_labels, epochs=400, 
                        validation_data=(test_images, test_labels))
    
    # Evaluate accuracy
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    # Print an appropriate message
    print(f"Training with {optimizer} finished. Test loss: {test_loss}, Test accuracy: {test_acc}")
    print("###############################")


# Make predictions
# With the model trained, you can use it to make predictions about some images. 
# The model's linear outputs, logits. 
# Attach a softmax layer to convert the logits to probabilities, which are easier to interpret. 
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

plot_some_predictions(test_images, test_labels, predictions, class_names, num_rows=5, num_cols=3)





