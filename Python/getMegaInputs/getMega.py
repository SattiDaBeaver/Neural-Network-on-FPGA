import numpy as np
import os
from tensorflow.keras.datasets import mnist

def float_to_q8_8_hex(val):
    val = np.clip(val, -128.0, 127.99609375)
    q_val = int(round(val * 256))
    if q_val < 0:
        q_val += 65536
    return format(q_val & 0xFFFF, '04x')

# Load MNIST test data
(_, _), (x_test, y_test) = mnist.load_data()

# Create output directory
output_dir = "GetInputs"
os.makedirs(output_dir, exist_ok=True)

# File paths
hex_path = os.path.join(output_dir, "q8_8_hex_inputs.txt")
bin_path = os.path.join(output_dir, "binary_inputs.txt")
label_path = os.path.join(output_dir, "labels.txt")

# Number of test images to convert
num_images = 10

with open(hex_path, "w") as hex_file, \
     open(bin_path, "w") as bin_file, \
     open(label_path, "w") as label_file:

    for idx in range(num_images):
        image = x_test[idx].astype(np.float32) / 255.0
        flattened = image.flatten()

        # Q8.8 hex string
        hex_vals = [float_to_q8_8_hex(pix) for pix in flattened]
        hex_file.write(''.join(hex_vals) + '\n')

        # Binary thresholded string
        bin_vals = ['1' if pix >= 0.5 else '0' for pix in flattened]
        bin_file.write(''.join(bin_vals) + '\n')

        # Label
        label_file.write(f"{y_test[idx]}\n")

print(f"✅ Saved files to folder: '{output_dir}'")
print(" - q8_8_hex_inputs.txt")
print(" - binary_inputs.txt")
print(" - labels.txt")

# Sample preview
print("\nSample conversion for first image (first 5 pixels):")
image = x_test[0].astype(np.float32) / 255.0
flattened = image.flatten()
for i in range(5):
    pix = flattened[i]
    hex_val = float_to_q8_8_hex(pix)
    bin_val = '1' if pix >= 0.5 else '0'
    int_val = int(hex_val, 16)
    if int_val >= 32768:
        int_val -= 65536
    recon = int_val / 256.0
    print(f"  Pixel {i}: {pix:.6f} -> 0x{hex_val} -> {recon:.6f} (binary: {bin_val})")
