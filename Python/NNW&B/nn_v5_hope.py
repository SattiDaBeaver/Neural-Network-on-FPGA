import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# ========== 1. Load & preprocess ==========
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.clip(x_train.astype(np.float32) / 255.0 * 32.0, -32, 31.999)
x_test = np.clip(x_test.astype(np.float32) / 255.0 * 32.0, -32, 31.999)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ========== 2. Model ==========
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(16, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=128, verbose=1)
model.evaluate(x_test, y_test)

# ========== 3. Convert float to Q6.10 ==========
def float_to_q6_10(val):
    return int(np.round(val * 1024.0))  # Signed 16-bit

def to_bin(val):
    return format(val & 0xFFFF, '016b')

# ========== 4. Save to per-neuron files ==========
def save_layer_weights_biases(weights, biases, layer_id):
    os.makedirs(f'Weights_Biases/weights', exist_ok=True)
    os.makedirs(f'Weights_Biases/bias', exist_ok=True)

    weights = weights.T  # So each row = weights for one neuron

    for neuron_idx in range(weights.shape[0]):
        # Save weights
        with open(f'Weights_Biases/weights/weight_L{layer_id}_N{neuron_idx}.mif', 'w') as wf:
            for val in weights[neuron_idx]:
                q = float_to_q6_10(val)
                wf.write(f'{to_bin(q)}\n')

        # Save bias
        with open(f'Weights_Biases/bias/bias_L{layer_id}_N{neuron_idx}.mif', 'w') as bf:
            q = float_to_q6_10(biases[neuron_idx])
            bf.write(f'{to_bin(q)}\n')

# ========== 5. Export L0 and L1 ==========
weights0, biases0 = model.layers[1].get_weights()
weights1, biases1 = model.layers[2].get_weights()

save_layer_weights_biases(weights0, biases0, layer_id=0)
save_layer_weights_biases(weights1, biases1, layer_id=1)

# ========== 6. Export input image ==========
sample_idx = 0
x_sample = x_test[sample_idx].flatten()

os.makedirs("Inputs", exist_ok=True)
with open("Inputs/input_image_q6_10.txt", "w") as f:
    for val in x_sample:
        q = float_to_q6_10(val)
        f.write(f'{to_bin(q)}\n')
