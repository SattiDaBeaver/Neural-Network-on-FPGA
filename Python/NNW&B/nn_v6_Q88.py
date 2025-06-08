import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# ========== 1. Load & preprocess ==========
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Binarize: if pixel > threshold → 32, else → 0
threshold = 128
x_train = ((x_train > threshold).astype(np.float32)) * 32.0
x_test = ((x_test > threshold).astype(np.float32)) * 32.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# ========== 2. Model ==========
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(16, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=15, batch_size=100, verbose=1)
test_loss, test_accuracy = model.evaluate(x_test, y_test)

# ========== 3. Fixed-point conversion: Q8.8 ==========
def float_to_q8_8(val):
    return int(np.round(val * 256.0))  # Signed 16-bit Q8.8

def to_bin(val):
    return format(val & 0xFFFF, '016b')

def to_hex(val):
    return format(val & 0xFFFF, '04X')  # Uppercase 4-digit hex

# ========== 4. Save weights & biases ==========
def save_layer_weights_biases(weights, biases, layer_id):
    os.makedirs(f'Weights_Biases/weights', exist_ok=True)
    os.makedirs(f'Weights_Biases/bias', exist_ok=True)

    weights = weights.T  # shape: (neurons, inputs)

    for neuron_idx in range(weights.shape[0]):
        with open(f'Weights_Biases/weights/weight_L{layer_id}_N{neuron_idx}.mif', 'w') as wf:
            for val in weights[neuron_idx]:
                q = float_to_q8_8(val)
                wf.write(f'{to_bin(q)}\n')
        with open(f'Weights_Biases/bias/bias_L{layer_id}_N{neuron_idx}.mif', 'w') as bf:
            q = float_to_q8_8(biases[neuron_idx])
            bf.write(f'{to_bin(q)}\n')

weights0, biases0 = model.layers[1].get_weights()
weights1, biases1 = model.layers[2].get_weights()

save_layer_weights_biases(weights0, biases0, layer_id=0)
save_layer_weights_biases(weights1, biases1, layer_id=1)

# ========== 5. Test Sample Inputs & Log ==========
os.makedirs("Inputs", exist_ok=True)
os.makedirs("LayerOutputs", exist_ok=True)
os.makedirs("Test_Results", exist_ok=True)

with open("Test_Results/test_results.txt", "w") as log:
    log.write(f"Overall Test Accuracy: {test_accuracy:.4f}, Loss: {test_loss:.4f}\n\n")

    for sample_idx in range(10):
        x_sample = x_test[sample_idx].flatten()
        y_true = np.argmax(y_test[sample_idx])

        # Save input: single line, hex Q8.8, no spaces
        with open(f"Inputs/input_{sample_idx}_q8_8.txt", "w") as f:
            hex_vals = ''.join([to_hex(float_to_q8_8(val)) for val in x_sample])
            f.write(hex_vals + '\n')

        # Forward pass
        layer0_out = np.maximum(0, np.dot(x_sample, weights0) + biases0)
        layer1_out = np.dot(layer0_out, weights1) + biases1
        prediction = np.argmax(layer1_out)

        # Save Layer 0 output (one neuron per line)
        with open(f"LayerOutputs/layer0_output_{sample_idx}.mif", "w") as f0:
            for i, val in enumerate(layer0_out):
                q = float_to_q8_8(val)
                f0.write(f'N{i}: DEC={val:.6f}, BIN={to_bin(q)}\n')

        # Save Layer 1 output (one neuron per line)
        with open(f"LayerOutputs/layer1_output_{sample_idx}.mif", "w") as f1:
            for i, val in enumerate(layer1_out):
                q = float_to_q8_8(val)
                f1.write(f'N{i}: DEC={val:.6f}, BIN={to_bin(q)}\n')

        # Log results
        with open("Test_Results/test_results.txt", "a") as log:
            log.write(f"Sample {sample_idx}:\n")
            log.write(f"  True Label:      {y_true}\n")
            log.write(f"  Predicted Label: {prediction}\n")
            log.write(f"  Correct:         {prediction == y_true}\n\n")
