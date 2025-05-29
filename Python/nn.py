import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Custom Layer (reLU and layer unga)
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer




# Constants
inputMax = 100

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images to the range [0, inputMax]
x_train = x_train.astype('float32') / 255.0
x_train = (x_train * inputMax).astype('uint8')
x_test = x_test.astype('float32') / 255.0
x_test = (x_test * inputMax).astype('uint8')

#Print Information about the dataset
print("Feature matrix (x_train):", x_train.shape)
print("Target matrix (y_train):", y_train.shape)
print("Feature matrix (x_test):", x_test.shape)
print("Target matrix (y_test):", y_test.shape)

# Neural Network Model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images to a vector
    FixedPointDense(16),   # Hidden layer with 16 neurons and ReLU activation
    Dense(10, activation='softmax')   # Output layer for 10 classes (digits 0-9)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
mod = model.fit(x_train, y_train, epochs=10, 
          batch_size=2000, 
          validation_split=0.2)
          
print(mod)

# Evaluate the model
results = model.evaluate(x_test, y_test, verbose=0)
print('Test loss, Test accuracy:', results)


# Print 4 images in a row
plt.figure(figsize=(10, 5))
for i in range(4):
    #print(x_train[i])
    plt.subplot(1, 4, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

