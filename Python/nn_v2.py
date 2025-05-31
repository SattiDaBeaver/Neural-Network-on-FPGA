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

# Custom argmax layer to replace softmax (mimics your FPGA's "largest value" function)
class ArgMaxOutput(Layer):
    def __init__(self, **kwargs):
        super(ArgMaxOutput, self).__init__(**kwargs)
    
    def call(self, inputs, training=None):
        # During training, use softmax for gradient flow
        if training:
            # Scale up the logits to make softmax more decisive
            scaled_inputs = inputs * 10.0  # Increase temperature
            return tf.nn.softmax(scaled_inputs)
        else:
            # During inference, return one-hot of argmax (like your FPGA)
            argmax_indices = tf.argmax(inputs, axis=-1)
            return tf.one_hot(argmax_indices, depth=tf.shape(inputs)[-1], dtype=inputs.dtype)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super().get_config()
        return config

# Constants
inputMax = 127  # Changed to match Q1.7 range better

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images to proper Q1.7 range [0, 127/128]
x_train = x_train.astype('float32') / 255.0
x_train = x_train * (127.0/128.0)  # Proper Q1.7 positive range
x_test = x_test.astype('float32') / 255.0
x_test = x_test * (127.0/128.0)

#Print Information about the dataset
print("Feature matrix (x_train):", x_train.shape)
print("Target matrix (y_train):", y_train.shape)
print("Feature matrix (x_test):", x_test.shape)
print("Target matrix (y_test):", y_test.shape)
print("Input range:", np.min(x_train), "to", np.max(x_train))

# Neural Network Model - Both layers are now quantized like your FPGA
model = Sequential([
    Flatten(input_shape = (28, 28)),  # Flatten the 28x28 images to a vector
    FPGADenseReLU(16),   # Hidden layer with 16 neurons and ReLU activation
    FPGADenseReLU(10),   # Output layer with 10 neurons and ReLU activation (like your FPGA)
    ArgMaxOutput()       # Custom layer that mimics your "largest value" function
])

# Custom accuracy metric that works with our argmax output
def fpga_accuracy(y_true, y_pred):
    # Convert predictions to class indices
    pred_classes = tf.argmax(y_pred, axis=-1)
    true_classes = tf.cast(y_true, tf.int64)
    return tf.reduce_mean(tf.cast(tf.equal(pred_classes, true_classes), tf.float32))

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='sparse_categorical_crossentropy', 
    metrics=[fpga_accuracy]
)

model.summary()

# Train the model
print("Training FPGA-accurate model...")
mod = model.fit(
    x_train, y_train, 
    epochs=25,  # More epochs since quantized training is harder
    batch_size=64,  # Smaller batch size
    validation_split=0.2,
    verbose=1
)

print(mod)

# Evaluate the model
results = model.evaluate(x_test, y_test, verbose=0)
print('Test loss, Test accuracy:', results)

# Test individual predictions to see argmax behavior
print("\nTesting argmax behavior on first 10 test samples:")
test_samples = x_test[:10]
predictions = model.predict(test_samples, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = y_test[:10]

for i in range(10):
    print(f"Sample {i}: Predicted={predicted_classes[i]}, Actual={actual_classes[i]}, Match={predicted_classes[i]==actual_classes[i]}")

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

# Create separate directories for weights and biases
weights_dir = os.path.join(output_root, "Weights")
biases_dir = os.path.join(output_root, "Biases")
os.makedirs(weights_dir, exist_ok=True)
os.makedirs(biases_dir, exist_ok=True)

# Export weights and biases from both FPGA layers (indices 1 and 2)
for layer_num, model_layer_idx in enumerate([1, 2]):  # Both are now FPGADenseReLU layers
    fpga_layer = model.layers[model_layer_idx]
    weights, biases = fpga_layer.get_weights()
    
    num_inputs = weights.shape[0]
    num_neurons = weights.shape[1]
    
    print(f"Exporting Layer {layer_num}: {num_inputs} inputs -> {num_neurons} neurons")
    
    for neuron_idx in range(num_neurons):
        # Save weights for this neuron with layer-specific naming
        weight_filename = os.path.join(weights_dir, f"weight_L{layer_num}_N{neuron_idx}.mif")
        with open(weight_filename, "w") as wf:
            for input_idx in range(num_inputs):
                q17_bin = float_to_q17_bin(weights[input_idx][neuron_idx])
                wf.write(q17_bin + "\n")

        # Save bias for this neuron with layer-specific naming
        bias_filename = os.path.join(biases_dir, f"bias_L{layer_num}_N{neuron_idx}.mif")
        with open(bias_filename, "w") as bf:
            q17_bin = float_to_q17_bin(biases[neuron_idx])
            bf.write(q17_bin + "\n")

print("Weight and bias export completed!")
print(f"Weights saved to: {weights_dir}")
print(f"Biases saved to: {biases_dir}")

# Visualize the training 
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(mod.history['fpga_accuracy'], label='Training Accuracy', color='blue')
plt.plot(mod.history['val_fpga_accuracy'], label='Validation Accuracy', color='orange')
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

plt.suptitle("FPGA Model Training Performance", fontsize=16)
plt.tight_layout()
plt.show()