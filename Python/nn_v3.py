import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# ========== Load and preprocess MNIST ==========
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = np.clip(x_train, 0, 1)
x_test = np.clip(x_test, 0, 1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ========== Define simple model ==========
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(16, activation='relu', name='L0'),
    Dense(10, activation='softmax', name='L1')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ========== Train model ==========
print("Training model...")
model.fit(x_train, y_train, epochs=15, batch_size=80, validation_data=(x_test, y_test))

# ========== Helper function to convert float to Q1.7 and save as binary ==========
def save_q17_binary_file(data, filename):
    """
    Convert float data to Q1.7 format and save as binary string representation.
    Q1.7: 1 sign bit + 7 fractional bits
    Range: -1.0 to +0.9921875 (127/128)
    """
    # Clip data to Q1.7 range
    clipped_data = np.clip(data, -1.0, 127/128)
    
    # Convert to Q1.7 fixed-point representation
    # For Q1.7: multiply by 2^7 = 128 and round to nearest integer
    q17_int = np.round(clipped_data * 128).astype(np.int16)  # Use int16 to avoid overflow
    
    # Convert each value to 8-bit binary string
    binary_strings = []
    for val in q17_int:
        # Convert to 8-bit representation (handle two's complement)
        if val < 0:
            # Two's complement for negative numbers
            unsigned_val = val & 0xFF  # Mask to 8 bits
        else:
            unsigned_val = val
        
        # Convert to 8-bit binary string
        binary_str = format(unsigned_val, '08b')
        binary_strings.append(binary_str)
    
    # Save as .mif file with binary strings
    with open(filename + '.mif', 'w') as f:
        for binary_str in binary_strings:
            f.write(binary_str + '\n')
    
    print(f"Saved {len(binary_strings)} values to {filename}.mif")
    if len(binary_strings) <= 5:  # Print first few for verification
        print(f"Sample binary values: {binary_strings}")
        print(f"Sample Q1.7 integers: {q17_int.tolist()}")
        # Handle both numpy arrays and Python lists
        if hasattr(data, 'tolist'):
            print(f"Original float values: {data.tolist()}")
        else:
            print(f"Original float values: {data}")

# ========== Export Weights and Biases in Q1.7 ==========
os.makedirs("Weights_Biases/weights", exist_ok=True)
os.makedirs("Weights_Biases/bias", exist_ok=True)

for layer_idx, layer_name in enumerate(['L0', 'L1']):
    layer = model.get_layer(name=layer_name)
    weights, biases = layer.get_weights()
    
    print(f"\nProcessing Layer {layer_name}:")
    print(f"Weight shape: {weights.shape}, Bias shape: {biases.shape}")
    
    # Save weights per neuron
    for neuron_idx in range(weights.shape[1]):
        neuron_weights = weights[:, neuron_idx]
        filename = f"Weights_Biases/weights/weight_L{layer_idx}_N{neuron_idx}"
        print(f"\nSaving weights for Layer {layer_idx}, Neuron {neuron_idx}:")
        save_q17_binary_file(neuron_weights, filename)
    
    # Save biases per neuron
    for neuron_idx, bias in enumerate(biases):
        filename = f"Weights_Biases/bias/bias_L{layer_idx}_N{neuron_idx}"
        print(f"\nSaving bias for Layer {layer_idx}, Neuron {neuron_idx}:")
        save_q17_binary_file([bias], filename)

print("\n✅ All weights and biases exported in Q1.7 format to 'Weights_Biases/' folders.")
print("📁 Files saved with .mif extension containing binary strings (01010010 format)")