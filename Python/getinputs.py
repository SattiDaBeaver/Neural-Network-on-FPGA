import numpy as np
from tensorflow.keras.datasets import mnist

def float_to_q17_hex(val):
    # Clip to valid range for Q1.7 format: -1.0 to 127/128
    val = np.clip(val, -1.0, 127/128)
    
    # Convert to Q1.7 fixed point
    # Q1.7 uses 1 sign bit + 7 fractional bits
    q_val = int(round(val * 128))  # Scale by 2^7
    
    # Handle negative values in two's complement
    if q_val < 0:
        q_val = 256 + q_val  # Convert to unsigned 8-bit representation
    
    return format(q_val, '02x')  # 2-digit hex

# Load MNIST
(_, _), (x_test, y_test) = mnist.load_data()

# Parameters
num_images = 8  # Choose how many images
bit_length = 784 * 8  # 8 bits per pixel × 784 pixels = 6272 bits

with open("flattened_inputs_hex.txt", "w") as data_file, open("labels.txt", "w") as label_file:
    for idx in range(num_images):
        # Normalize to [0, 1] range
        image = x_test[idx].astype(np.float32) / 255.0
        flattened = image.flatten()
        
        # Convert each pixel to Q1.7 hex
        hex_vals = [float_to_q17_hex(pix) for pix in flattened]
        hex_string = ''.join(hex_vals)
        
        # Write in the format: 6272'h{hex_string}
        data_file.write(f"{bit_length}'h{hex_string}\n")
        label_file.write(f"{y_test[idx]}\n")

print(f"Saved {num_images} images to 'flattened_inputs_hex.txt' and labels to 'labels.txt'")
print(f"Each image is in Q1.7 fixed-point format: {bit_length}'h...")