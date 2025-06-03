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
    x = tf.clip_by_value(x, -1.0, (scale - 1) / scale)
    x_int = tf.round(x * scale)
    return x_int / scale  # Still float32, but quantized

# Function for custom layer
class FPGADenseReLU(Layer):
    def __init__(self, units, **kwargs):
        super(FPGADenseReLU, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # 8-bit signed integers for weights and biases
        self.w = self.add_weight(
            shape=(input_dim, self.units),
            initializer='random_uniform',
            trainable=True,
            dtype=tf.float32,
            name="float_weights"
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            dtype=tf.float32,
            name="float_biases"
        )

    def call(self, inputs):
        inputs_q = fake_quantize(inputs, self.scale)
        weights_q = fake_quantize(self.w, self.scale)
        biases_q = fake_quantize(self.b, self.scale)

        x = tf.matmul(inputs_q, weights_q) + biases_q
        x = tf.maximum(x, 0.0)
        x = tf.clip_by_value(x, 0.0, (self.scale - 1) / self.scale)

        return x


# Custom argmax layer
class ArgMaxOutput(Layer):
    def __init__(self, **kwargs):
        super(ArgMaxOutput, self).__init__(**kwargs)
    
    def call(self, inputs, training=None):
        if training:
            scaled_inputs = inputs * 10.0
            return tf.nn.softmax(scaled_inputs)
        else:
            argmax_indices = tf.argmax(inputs, axis=-1)
            return tf.one_hot(argmax_indices, depth=tf.shape(inputs)[-1], dtype=inputs.dtype)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        return super().get_config()

# Constants
inputMax = 127

# Load and normalize MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_train = x_train * (127.0/128.0)
x_test = x_test.astype('float32') / 255.0
x_test = x_test * (127.0/128.0)

print("Feature matrix (x_train):", x_train.shape)
print("Target matrix (y_train):", y_train.shape)
print("Feature matrix (x_test):", x_test.shape)
print("Target matrix (y_test):", y_test.shape)
print("Input range:", np.min(x_train), "to", np.max(x_train))

# Define model
model = Sequential([
    Flatten(input_shape = (28, 28)),
    FPGADenseReLU(16),
    FPGADenseReLU(10),
    ArgMaxOutput()
])

# Custom accuracy
def fpga_accuracy(y_true, y_pred):
    pred_classes = tf.argmax(y_pred, axis=-1)
    true_classes = tf.cast(y_true, tf.int64)
    return tf.reduce_mean(tf.cast(tf.equal(pred_classes, true_classes), tf.float32))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='sparse_categorical_crossentropy', 
    metrics=[fpga_accuracy]
)

model.summary()

# Train
print("Training FPGA-accurate model...")
mod = model.fit(
    x_train, y_train, 
    epochs=25,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

print(mod)

# Evaluate
results = model.evaluate(x_test, y_test, verbose=0)
print('Test loss, Test accuracy:', results)

# Test individual predictions
print("\nTesting argmax behavior on first 10 test samples:")
test_samples = x_test[:10]
predictions = model.predict(test_samples, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = y_test[:10]

for i in range(10):
    print(f"Sample {i}: Predicted={predicted_classes[i]}, Actual={actual_classes[i]}, Match={predicted_classes[i]==actual_classes[i]}")

# Helper: Float to Q1.7
def float_to_q17_bin(x):
    x = max(-1.0, min(0.9921875, x))
    val = int(round(x * 128))
    return format(val & 0xFF, '08b')

# Export weights/biases
output_root = "Weight_Biases"
weights_dir = os.path.join(output_root, "weights")
biases_dir = os.path.join(output_root, "bias")
os.makedirs(weights_dir, exist_ok=True)
os.makedirs(biases_dir, exist_ok=True)

for layer_num, model_layer_idx in enumerate([1, 2]):
    fpga_layer = model.layers[model_layer_idx]
    weights, biases = fpga_layer.get_weights()
    
    num_inputs = weights.shape[0]
    num_neurons = weights.shape[1]
    
    print(f"Exporting Layer {layer_num}: {num_inputs} inputs -> {num_neurons} neurons")
    
    for neuron_idx in range(num_neurons):
        weight_filename = os.path.join(weights_dir, f"weight_L{layer_num}_N{neuron_idx}.mif")
        with open(weight_filename, "w") as wf:
            for input_idx in range(num_inputs):
                q17_bin = float_to_q17_bin(weights[input_idx][neuron_idx])
                wf.write(q17_bin + "\n")

        bias_filename = os.path.join(biases_dir, f"bias_L{layer_num}_N{neuron_idx}.mif")
        with open(bias_filename, "w") as bf:
            q17_bin = float_to_q17_bin(biases[neuron_idx])
            bf.write(q17_bin + "\n")

print("Weight and bias export completed!")
print(f"Weights saved to: {weights_dir}")
print(f"Biases saved to: {biases_dir}")

# Plot training results
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
