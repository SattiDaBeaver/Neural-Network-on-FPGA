import os
import numpy as np
from tensorflow.keras.datasets import mnist

# Create output directory
output_dir = "GetInputs"
os.makedirs(output_dir, exist_ok=True)

# Load MNIST test set
(_, _), (x_test, y_test) = mnist.load_data()

# Choose image index
index = 0
image = x_test[index]

# Binarize: threshold at 128
binary_image = (image > 128).astype(np.uint8)
flattened = binary_image.flatten()
bit_string = ''.join(str(bit) for bit in flattened)

# Save flat 784-bit string
with open(os.path.join(output_dir, "mnist_binary.txt"), "w") as f:
    f.write(bit_string)

# Save readable 28x28 format
with open(os.path.join(output_dir, "mnist_readable.txt"), "w") as f:
    for i in range(28):
        row = ''.join(str(bit) for bit in binary_image[i])
        f.write(row + '\n')

# Convert to Q8.8 hex (0 -> 0x0000, 1 -> 0x0100)
hex_vals = ['0100' if b else '0000' for b in flattened]
hex_string = ''.join(hex_vals)

# Save Q8.8 hex string
with open(os.path.join(output_dir, "mnist_q88_hex.txt"), "w") as f:
    f.write(hex_string)

print(f"Saved 784-bit binary to '{output_dir}/mnist_binary.txt'")
print(f"Saved readable 28x28 image to '{output_dir}/mnist_readable.txt'")
print(f"Saved Q8.8 hex string to '{output_dir}/mnist_q88_hex.txt'")
print(f"Label: {y_test[index]}")
