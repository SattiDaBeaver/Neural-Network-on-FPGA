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

# ========== Helper function to convert float to Q6.10 and save as binary ==========
def save_q610_binary_file(data, filename):
    """
    Convert float data to Q6.10 format and save as binary string representation.
    Q6.10: 6 bits for integer part (including sign) + 10 bits for fractional part = 16 bits total
    Range: -32.0 to +31.999023438 (32767/1024)
    """
    # Clip data to Q6.10 range
    clipped_data = np.clip(data, -32.0, 32767/1024)
    
    # Convert to Q6.10 fixed-point representation
    # For Q6.10: multiply by 2^10 = 1024 and round to nearest integer
    q610_int = np.round(clipped_data * 1024).astype(np.int32)  # Use int32 to avoid overflow
    
    # Convert each value to 16-bit binary string
    binary_strings = []
    for val in q610_int:
        # Convert to 16-bit representation (handle two's complement)
        if val < 0:
            # Two's complement for negative numbers (16-bit)
            unsigned_val = val & 0xFFFF  # Mask to 16 bits
        else:
            unsigned_val = val
        
        # Convert to 16-bit binary string
        binary_str = format(unsigned_val, '016b')
        binary_strings.append(binary_str)
    
    # Save as .mif file with binary strings
    with open(filename + '.mif', 'w') as f:
        for binary_str in binary_strings:
            f.write(binary_str + '\n')
    
    print(f"Saved {len(binary_strings)} values to {filename}.mif")
    if len(binary_strings) <= 5:  # Print first few for verification
        print(f"Sample binary values: {binary_strings}")
        print(f"Sample Q6.10 integers: {q610_int.tolist()}")
        # Handle both numpy arrays and Python lists
        if hasattr(data, 'tolist'):
            print(f"Original float values: {data.tolist()}")
        else:
            print(f"Original float values: {data}")

# ========== Export Weights and Biases in Q6.10 ==========
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
        save_q610_binary_file(neuron_weights, filename)
    
    # Save biases per neuron
    for neuron_idx, bias in enumerate(biases):
        filename = f"Weights_Biases/bias/bias_L{layer_idx}_N{neuron_idx}"
        print(f"\nSaving bias for Layer {layer_idx}, Neuron {neuron_idx}:")
        save_q610_binary_file([bias], filename)

print("\n✅ All weights and biases exported in Q6.10 format to 'Weights_Biases/' folders.")
print("📁 Files saved with .mif extension containing binary strings (16-bit format)")


# Testing
# ========== Test 10 samples and export neuron outputs ==========
print("\n" + "="*60)
print("TESTING 10 SAMPLES AND EXPORTING NEURON OUTPUTS")
print("="*60)

# Create directories for test outputs
os.makedirs("Test_Outputs/inputs", exist_ok=True)
os.makedirs("Test_Outputs/layer_outputs", exist_ok=True)
os.makedirs("Test_Outputs/final_outputs", exist_ok=True)

# Helper function to get intermediate layer outputs
def get_layer_outputs(model, x_input):
    """Get outputs from each layer"""
    layer_outputs = []
    
    # Method 1: Use a separate model to get L0 output
    # Create input layer explicitly
    input_layer = tf.keras.Input(shape=(28, 28))
    flattened = tf.keras.layers.Flatten()(input_layer)
    l0_output = model.get_layer('L0')(flattened)
    l0_model = tf.keras.Model(inputs=input_layer, outputs=l0_output)
    
    l0_result = l0_model.predict(x_input, verbose=0)
    layer_outputs.append(l0_result)
    
    # Method 2: Get final output from complete model
    l1_result = model.predict(x_input, verbose=0)
    layer_outputs.append(l1_result)
    
    return layer_outputs

# Helper function to save decimal values
def save_decimal_file(data, filename):
    """Save decimal values to file"""
    with open(filename + '.txt', 'w') as f:
        if data.ndim == 1:
            for val in data:
                f.write(f"{val:.6f}\n")
        else:
            for sample_idx in range(data.shape[0]):
                f.write(f"Sample {sample_idx}:\n")
                for val in data[sample_idx]:
                    f.write(f"{val:.6f}\n")
                f.write("\n")
    print(f"Saved decimal values to {filename}.txt")

# Helper function to convert float to Q1.7 hex (keeping inputs in Q1.7 as in original)
def float_to_q17_hex(val):
    val = np.clip(val, -1.0, 127/128)
    q_val = int(round(val * 128))
    if q_val < 0:
        q_val = 256 + q_val
    return format(q_val, '02x')

# Helper function to convert float to Q6.10 hex
def float_to_q610_hex(val):
    val = np.clip(val, -32.0, 32767/1024)
    q_val = int(round(val * 1024))
    if q_val < 0:
        q_val = 65536 + q_val  # 2^16 for 16-bit representation
    return format(q_val, '04x')

# Select 10 test samples
num_test_samples = 10
test_indices = np.arange(num_test_samples)
x_test_samples = x_test[test_indices]
y_test_samples = y_test[test_indices]
y_test_labels = np.argmax(y_test_samples, axis=1)

print(f"Selected {num_test_samples} test samples")
print(f"True labels: {y_test_labels}")

# ========== Export Test Inputs (keeping Q1.7 as in original) ==========
print("\n--- Exporting Test Inputs ---")
for i in range(num_test_samples):
    # Save input image in Q1.7 hex format (keeping original format for inputs)
    image = x_test_samples[i].flatten()
    hex_vals = [float_to_q17_hex(pix) for pix in image]
    hex_string = ''.join(hex_vals)
    
    # Save as hex string
    with open(f"Test_Outputs/inputs/input_sample_{i}_hex.txt", "w") as f:
        f.write(f"6272'h{hex_string}\n")
    
    # Save as binary for verification (Q1.7)
    save_q17_binary_file(image, f"Test_Outputs/inputs/input_sample_{i}_binary")
    
    # Save as decimal for verification
    save_decimal_file(image, f"Test_Outputs/inputs/input_sample_{i}_decimal")

print(f"✅ Exported {num_test_samples} test inputs")

# Helper function to convert float to Q1.7 and save as binary (for inputs/outputs)
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

# ========== Get Layer Outputs ==========
print("\n--- Getting Layer Outputs ---")
layer_outputs = get_layer_outputs(model, x_test_samples)

# ========== Export Layer Outputs ==========
print("\n--- Exporting Layer Outputs ---")

# Export L0 outputs (hidden layer)
l0_outputs = layer_outputs[0]  # Shape: (10, 16)
print(f"L0 output shape: {l0_outputs.shape}")

for sample_idx in range(num_test_samples):
    sample_l0_output = l0_outputs[sample_idx]
    
    # Save decimal
    save_decimal_file(sample_l0_output, f"Test_Outputs/layer_outputs/L0_sample_{sample_idx}_decimal")
    
    # Save binary Q6.10 (now using Q6.10 for outputs)
    save_q610_binary_file(sample_l0_output, f"Test_Outputs/layer_outputs/L0_sample_{sample_idx}_binary")
    
    # Save individual neuron outputs
    for neuron_idx in range(16):
        neuron_output = sample_l0_output[neuron_idx]
        with open(f"Test_Outputs/layer_outputs/L0_sample_{sample_idx}_neuron_{neuron_idx}.txt", "w") as f:
            f.write(f"Decimal: {neuron_output:.6f}\n")
            f.write(f"Q6.10 binary: {format(int(round(np.clip(neuron_output, -32.0, 32767/1024) * 1024)) & 0xFFFF, '016b')}\n")
            f.write(f"Q6.10 hex: {float_to_q610_hex(neuron_output)}\n")

# Export L1 outputs (final layer)
l1_outputs = layer_outputs[1]  # Shape: (10, 10)
print(f"L1 output shape: {l1_outputs.shape}")

for sample_idx in range(num_test_samples):
    sample_l1_output = l1_outputs[sample_idx]
    
    # Save decimal
    save_decimal_file(sample_l1_output, f"Test_Outputs/final_outputs/L1_sample_{sample_idx}_decimal")
    
    # Save binary Q6.10 (now using Q6.10 for outputs)
    save_q610_binary_file(sample_l1_output, f"Test_Outputs/final_outputs/L1_sample_{sample_idx}_binary")
    
    # Save individual neuron outputs
    for neuron_idx in range(10):
        neuron_output = sample_l1_output[neuron_idx]
        with open(f"Test_Outputs/final_outputs/L1_sample_{sample_idx}_neuron_{neuron_idx}.txt", "w") as f:
            f.write(f"Decimal: {neuron_output:.6f}\n")
            f.write(f"Q6.10 binary: {format(int(round(np.clip(neuron_output, -32.0, 32767/1024) * 1024)) & 0xFFFF, '016b')}\n")
            f.write(f"Q6.10 hex: {float_to_q610_hex(neuron_output)}\n")

# ========== Create Summary File ==========
print("\n--- Creating Summary File ---")
with open("Test_Outputs/test_summary.txt", "w") as f:
    f.write("MNIST Neural Network Test Summary\n")
    f.write("="*40 + "\n\n")
    
    f.write("Network Architecture:\n")
    f.write("- Input: 784 pixels (28x28 flattened) - Q1.7 format\n")
    f.write("- Layer L0: 16 neurons (ReLU) - Weights/Biases in Q6.10 format\n")
    f.write("- Layer L1: 10 neurons (Softmax) - Weights/Biases in Q6.10 format\n")
    f.write("- Outputs: Q1.7 format\n\n")
    
    f.write("Fixed-Point Formats:\n")
    f.write("- Q1.7: 8-bit, range -1.0 to +0.9921875 (inputs/outputs)\n")
    f.write("- Q6.10: 16-bit, range -32.0 to +31.999023438 (weights/biases)\n\n")
    
    f.write("Test Results:\n")
    predictions = np.argmax(l1_outputs, axis=1)
    accuracy = np.mean(predictions == y_test_labels)
    f.write(f"Accuracy on {num_test_samples} samples: {accuracy*100:.1f}%\n\n")
    
    f.write("Sample-by-Sample Results:\n")
    for i in range(num_test_samples):
        f.write(f"Sample {i}: True={y_test_labels[i]}, Predicted={predictions[i]}, ")
        f.write(f"Confidence={l1_outputs[i][predictions[i]]:.4f}\n")
    
    f.write("\nFile Structure:\n")
    f.write("- Weights_Biases/: Neural network parameters in Q6.10 format (16-bit)\n")
    f.write("- inputs/: Test input images in Q1.7 format (8-bit)\n")
    f.write("- layer_outputs/: L0 hidden layer outputs in Q1.7 format\n")
    f.write("- final_outputs/: L1 output layer (softmax) outputs in Q1.7 format\n")
    f.write("- Individual neuron files contain decimal, binary, and hex values\n")

print("✅ Test complete! Files saved with Q6.10 weights and Q1.7 inputs/outputs")
print("\nSummary:")
print(f"- Tested {num_test_samples} samples")
print(f"- Accuracy: {np.mean(np.argmax(l1_outputs, axis=1) == y_test_labels)*100:.1f}%")
print(f"- Weights and biases saved in Q6.10 format (16-bit)")
print(f"- Inputs and outputs saved in Q1.7 format (8-bit)")
print(f"- Individual neuron outputs saved for verification")