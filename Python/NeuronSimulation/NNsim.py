import numpy as np

def bin_to_int16(bstr):
    bstr = bstr.strip()
    if len(bstr) != 16 or any(c not in '01' for c in bstr):
        raise ValueError(f"Invalid binary string: '{bstr}'")
    val = int(bstr, 2)
    if val & 0x8000:
        val -= 0x10000
    return np.int16(val)

def read_weights(file_path):
    with open(file_path, 'r') as f:
        return [bin_to_int16(line) for line in f if line.strip()]

def read_bias(file_path):
    with open(file_path, 'r') as f:
        line = f.readline().strip()
        return bin_to_int16(line)

def read_inputs(file_path):
    with open(file_path, 'r') as f:
        hexline = f.readline().strip()
    if len(hexline) % 4 != 0:
        raise ValueError("Input hex line length not divisible by 4 characters (16 bits)")
    inputs = []
    for i in range(0, len(hexline), 4):
        hex_word = hexline[i:i+4]
        val = int(hex_word, 16)
        if val & 0x8000:
            val -= 0x10000
        inputs.append(np.int16(val))
    return inputs

def emulate_neuron(input_file, weight_file, bias_file):
    inputs = read_inputs(input_file)
    weights = read_weights(weight_file)
    bias = read_bias(bias_file)

    if len(inputs) != len(weights):
        raise ValueError(f"Mismatch: {len(inputs)} inputs vs {len(weights)} weights")

    acc = 0
    print("\n===== Multiply-Accumulate Trace =====")
    for i in range(len(inputs)):
        input_val = int(inputs[i])
        weight_val = int(weights[i])
        product = input_val * weight_val  # Q8.8 * Q8.8 = Q16.16
        acc += product
        print(f"[{i:03d}] Input = {input_val:6d} ({input_val & 0xFFFF:04X}), "
              f"Weight = {weight_val:6d} ({weight_val & 0xFFFF:04X}) → "
              f"Product = {product:11d} ({product & 0xFFFFFFFF:08X}) → "
              f"Acc = {acc:11d} ({acc & 0xFFFFFFFF:08X})")

    bias_shifted = int(bias) << 8  # Q8.8 → Q16.16
    acc += bias_shifted
    print("\n===== Bias Addition =====")
    print(f"Bias       = {int(bias):6d} ({int(bias) & 0xFFFF:04X}) → Shifted = {bias_shifted:11d} ({bias_shifted & 0xFFFFFFFF:08X})")
    print(f"Acc + Bias = {acc:11d} ({acc & 0xFFFFFFFF:08X})")

    # Convert back to Q8.8
    result_q8_8 = acc >> 8
    relu_q8_8 = max(result_q8_8, 0)
    float_output = relu_q8_8 / 256.0

    print("\n===== Final Output =====")
    print(f"Q8.8 (int)   = {relu_q8_8:6d} ({relu_q8_8 & 0xFFFF:04X})")
    print(f"Float value  = {float_output:.6f}")

# ====== Example usage ======
if __name__ == "__main__":
    input_file = "C:/Projects/Neural Network on FPGA/Inputs/input_0_q8_8.txt"
    weight_file = "C:/Projects/Neural Network on FPGA/Weights_Biases/weights/weight_L0_N0.mif"
    bias_file   = "C:/Projects/Neural Network on FPGA/Weights_Biases/bias/bias_L0_N0.mif"

    emulate_neuron(input_file, weight_file, bias_file)
