import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import math
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Custom Layer (reLU and layer unga)
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

# Function to perform fake quantization
def fake_quantize(x, scale):
        return x + tf.stop_gradient(tf.round(x * scale) / scale - x)

#Function for custom layer
class FPGADenseReLU(Layer):
    def __init__(self, units, scale_factor=128, **kwargs):
        super(FPGADenseReLU, self).__init__(**kwargs)
        self.units = units
        self.scale = scale_factor  # Q1.7 → scale = 2^7 = 128

    def build(self, input_shape):
        # Store weights and biases as float32 for training
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='he_normal',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)

        # Fake quantize inputs, weights, and biases
        inputs_q = fake_quantize(inputs, self.scale)
        weights_q = fake_quantize(self.w, self.scale)
        biases_q = fake_quantize(self.b, self.scale)

        # Perform pseudo-fixed-point multiply and accumulate
        x = tf.matmul(inputs_q, weights_q) + biases_q

        # ReLU + saturation
        x = tf.maximum(x, 0.0)
        x = tf.clip_by_value(x, 0.0, (self.scale - 1) / self.scale)  # max is 127/128

        return x



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
    Flatten(input_shape = (28, 28)),  # Flatten the 28x28 images to a vector
    FPGADenseReLU(16),   # Hidden layer with 16 neurons and ReLU activation
    Dense(10, activation = 'softmax')   # Output layer for 10 classes (digits 0-9)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
mod = model.fit(x_train, y_train, epochs = 15, batch_size = 128, validation_split = 0.2)
          
print(mod)

# Evaluate the model
results = model.evaluate(x_test, y_test, verbose=0)
print('Test loss, Test accuracy:', results)

# Visualize the training 
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(mod.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(mod.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Training and Validation Accuracy', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(mod.history['loss'], label='Training Loss', color='blue')
plt.plot(mod.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Training and Validation Loss', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True)

plt.suptitle("Model Training Performance", fontsize=16)
plt.tight_layout()
plt.show()



# Get the custom FPGA layer
def float_to_q17_bin(x):
    """
    Converts a float to 8-bit signed binary in Q1.7 format.
    Clamps to [-1.0, 0.9921875] since 127/128 ≈ 0.9922.
    """
    # Clamp
    x = max(-1.0, min(0.9921875, x))
    # Scale
    val = int(round(x * 128))  # Q1.7 scale factor
    # Convert to 8-bit two's complement
    return format(val & 0xFF, '08b')



output_root = "Weight_Biases"

for layer_num, model_layer_idx in enumerate([1, 2]):  # Layer0=custom, Layer1=output
    # Create directory for this layer
    layer_dir = os.path.join(output_root, f"Layer{layer_num}")
    os.makedirs(layer_dir, exist_ok=True)
    
    fpga_layer = model.layers[model_layer_idx]
    weights, biases = fpga_layer.get_weights()
    
    num_inputs = weights.shape[0]
    num_neurons = weights.shape[1]
    
    for neuron_idx in range(num_neurons):
        # Save weights for this neuron
        weight_filename = os.path.join(layer_dir, f"weight_L{layer_num}_N{neuron_idx}")
        with open(weight_filename, "w") as wf:
            for input_idx in range(num_inputs):
                q17_bin = float_to_q17_bin(weights[input_idx][neuron_idx])
                wf.write(q17_bin + "\n")

        # Save bias for this neuron
        bias_filename = os.path.join(layer_dir, f"bias_L{layer_num}_N{neuron_idx}")
        with open(bias_filename, "w") as bf:
            q17_bin = float_to_q17_bin(biases[neuron_idx])
            bf.write(q17_bin + "\n")
