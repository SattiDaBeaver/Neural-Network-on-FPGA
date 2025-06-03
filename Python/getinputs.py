import numpy as np
from tensorflow.keras.datasets import mnist

def float_to_q6_10_hex(val):
    # Clip to valid range for Q6.10 format: -32.0 to 31.9990234375
    val = np.clip(val, -32.0, 31.9990234375)
    
    # Convert to Q6.10 fixed point
    # Q6.10 uses 6 integer bits (including sign) + 10 fractional bits = 16 total bits
    q_val = int(round(val * 1024))  # Scale by 2^10
    
    # Handle negative values in two's complement for 16-bit
    if q_val < 0:
        q_val = 65536 + q_val  # Convert to unsigned 16-bit representation
    
    # Ensure we don't exceed 16-bit range
    q_val = q_val & 0xFFFF
    
    return format(q_val, '04x')  # 4-digit hex for 16 bits

# Load MNIST
(_, _), (x_test, y_test) = mnist.load_data()

# Parameters
num_images = 8  # Choose how many images

with open("flattened_inputs_hex.txt", "w") as data_file, open("labels.txt", "w") as label_file:
    for idx in range(num_images):
        # Normalize to [0, 1] range
        image = x_test[idx].astype(np.float32) / 255.0
        flattened = image.flatten()
        
        # Convert each pixel to Q6.10 hex (4 hex digits per pixel)
        hex_vals = [float_to_q6_10_hex(pix) for pix in flattened]
        hex_string = ''.join(hex_vals)
        
        # Write just the hex string (no prefix)
        data_file.write(f"{hex_string}\n")
        label_file.write(f"{y_test[idx]}\n")

print(f"Saved {num_images} images to 'flattened_inputs_hex.txt' and labels to 'labels.txt'")
print(f"Each image is in Q6.10 fixed-point format (16-bit per pixel)")
print(f"Format: 6 integer bits (including sign) + 10 fractional bits")
print(f"Range: -32.0 to +31.9990234375")
print(f"Resolution: 1/1024 = 0.0009765625")
print(f"Each pixel: 4 hex digits, Total per image: {784 * 4} hex characters")

# Display some sample conversions for verification
print("\nSample pixel conversions (first 5 pixels of first image):")
image = x_test[0].astype(np.float32) / 255.0
flattened = image.flatten()
for i in range(5):
    pixel_val = flattened[i]
    hex_val = float_to_q6_10_hex(pixel_val)
    # Convert back to verify
    int_val = int(hex_val, 16)
    if int_val >= 32768:  # Handle two's complement
        int_val -= 65536
    reconstructed = int_val / 1024.0
    print(f"  Pixel {i}: {pixel_val:.6f} -> 0x{hex_val} -> {reconstructed:.6f}")