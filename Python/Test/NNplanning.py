import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

class NeuralNetworkAnalyzer:
    def __init__(self, layer_sizes, activation='relu'):
        """
        Initialize the neural network analyzer
        
        Args:
            layer_sizes: List of integers representing neurons per layer (excluding input)
                        e.g., [128, 64, 10] for 128->64->10 architecture
            activation: Activation function (default: 'relu')
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.model = None
        self.history = None
        self.stats = {}
        
    def build_model(self, input_shape=(28, 28)):
        """Build the neural network model"""
        self.model = models.Sequential()
        
        # Flatten input
        self.model.add(layers.Flatten(input_shape=input_shape))
        
        # Add hidden layers
        for i, size in enumerate(self.layer_sizes[:-1]):
            self.model.add(layers.Dense(size, activation=self.activation, 
                                      name=f'hidden_{i+1}'))
        
        # Output layer (softmax for classification)
        self.model.add(layers.Dense(self.layer_sizes[-1], activation='softmax', 
                                   name='output'))
        
        self.model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        
        print("Model Architecture:")
        self.model.summary()
        
    def load_and_preprocess_data(self):
        """Load and preprocess MNIST data"""
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Normalize pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Convert labels to categorical
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        
        return (x_train, y_train), (x_test, y_test)
    
    def train_model(self, epochs=5, batch_size=128, validation_split=0.1):
        """Train the model"""
        (x_train, y_train), (x_test, y_test) = self.load_and_preprocess_data()
        
        print(f"\nTraining model for {epochs} epochs...")
        self.history = self.model.fit(x_train, y_train,
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     validation_split=validation_split,
                                     verbose=1)
        
        # Evaluate on test set
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"\nTest accuracy: {test_acc:.4f}")
        
        return (x_train, y_train), (x_test, y_test)
    
    def analyze_weights_and_biases(self):
        """Analyze weight and bias distributions"""
        print("\n" + "="*60)
        print("WEIGHT AND BIAS ANALYSIS")
        print("="*60)
        
        self.stats['weights'] = {}
        self.stats['biases'] = {}
        
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'get_weights') and layer.get_weights():
                weights, biases = layer.get_weights()
                layer_name = layer.name
                
                # Weight statistics
                w_min, w_max = np.min(weights), np.max(weights)
                w_mean, w_std = np.mean(weights), np.std(weights)
                w_abs_max = np.max(np.abs(weights))
                
                # Bias statistics
                b_min, b_max = np.min(biases), np.max(biases)
                b_mean, b_std = np.mean(biases), np.std(biases)
                b_abs_max = np.max(np.abs(biases))
                
                self.stats['weights'][layer_name] = {
                    'min': w_min, 'max': w_max, 'mean': w_mean, 'std': w_std,
                    'abs_max': w_abs_max, 'shape': weights.shape
                }
                
                self.stats['biases'][layer_name] = {
                    'min': b_min, 'max': b_max, 'mean': b_mean, 'std': b_std,
                    'abs_max': b_abs_max, 'shape': biases.shape
                }
                
                print(f"\nLayer: {layer_name}")
                print(f"  Weights - Min: {w_min:.6f}, Max: {w_max:.6f}, "
                      f"Mean: {w_mean:.6f}, Std: {w_std:.6f}")
                print(f"  Weights - Abs Max: {w_abs_max:.6f}, Shape: {weights.shape}")
                print(f"  Biases  - Min: {b_min:.6f}, Max: {b_max:.6f}, "
                      f"Mean: {b_mean:.6f}, Std: {b_std:.6f}")
                print(f"  Biases  - Abs Max: {b_abs_max:.6f}, Shape: {biases.shape}")
    
    def analyze_neuron_outputs(self, x_sample, sample_size=1000):
        """Analyze neuron output distributions"""
        print("\n" + "="*60)
        print("NEURON OUTPUT ANALYSIS")
        print("="*60)
        
        # Use a subset for analysis to speed things up
        if len(x_sample) > sample_size:
            indices = np.random.choice(len(x_sample), sample_size, replace=False)
            x_analysis = x_sample[indices]
        else:
            x_analysis = x_sample
            
        self.stats['neuron_outputs'] = {}
        
        # First, make sure the model has been called by doing a prediction
        _ = self.model.predict(x_analysis[:1], verbose=0)
        
        # Create intermediate models to get layer outputs
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'activation') and ('dense' in layer.name.lower() or 'hidden' in layer.name.lower()):
                # Create model that outputs up to this layer
                try:
                    intermediate_model = models.Model(inputs=self.model.inputs,
                                                    outputs=layer.output)
                    
                    # Get outputs for sample data
                    outputs = intermediate_model.predict(x_analysis, verbose=0)
                    
                    # Calculate statistics
                    out_min = np.min(outputs)
                    out_max = np.max(outputs)
                    out_mean = np.mean(outputs)
                    out_std = np.std(outputs)
                    out_abs_max = np.max(np.abs(outputs))
                    
                    # Calculate percentage of zero outputs (ReLU clipping)
                    zero_percentage = np.mean(outputs == 0) * 100
                    
                    self.stats['neuron_outputs'][layer.name] = {
                        'min': out_min, 'max': out_max, 'mean': out_mean, 'std': out_std,
                        'abs_max': out_abs_max, 'zero_percentage': zero_percentage,
                        'shape': outputs.shape
                    }
                    
                    print(f"\nLayer: {layer.name}")
                    print(f"  Output Min: {out_min:.6f}, Max: {out_max:.6f}")
                    print(f"  Output Mean: {out_mean:.6f}, Std: {out_std:.6f}")
                    print(f"  Output Abs Max: {out_abs_max:.6f}")
                    print(f"  Zero outputs (ReLU clipping): {zero_percentage:.2f}%")
                    print(f"  Output shape: {outputs.shape}")
                    
                except Exception as e:
                    print(f"Could not analyze layer {layer.name}: {e}")
                    continue
    
    def suggest_fpga_resolution(self):
        """Suggest bit widths for FPGA implementation"""
        print("\n" + "="*60)
        print("FPGA RESOLUTION SUGGESTIONS")
        print("="*60)
        
        # Find maximum absolute values across all weights, biases, and outputs
        max_weight = max([stats['abs_max'] for stats in self.stats['weights'].values()])
        max_bias = max([stats['abs_max'] for stats in self.stats['biases'].values()])
        
        # Check if we have neuron output stats
        if self.stats['neuron_outputs']:
            max_output = max([stats['abs_max'] for stats in self.stats['neuron_outputs'].values()])
            overall_max = max(max_weight, max_bias, max_output)
            print(f"Maximum absolute values:")
            print(f"  Weights: {max_weight:.6f}")
            print(f"  Biases: {max_bias:.6f}")
            print(f"  Neuron outputs: {max_output:.6f}")
            print(f"  Overall maximum: {overall_max:.6f}")
        else:
            overall_max = max(max_weight, max_bias)
            print(f"Maximum absolute values:")
            print(f"  Weights: {max_weight:.6f}")
            print(f"  Biases: {max_bias:.6f}")
            print(f"  Overall maximum: {overall_max:.6f}")
            print("  Note: Neuron outputs not analyzed due to model structure")
        
        # Calculate required integer bits
        if overall_max == 0:
            int_bits = 1
        else:
            int_bits = int(np.ceil(np.log2(overall_max))) + 1  # +1 for sign bit
        
        # Suggest different precision options
        print(f"\nSuggested bit widths:")
        print(f"  Minimum integer bits needed: {int_bits} (including sign)")
        
        for total_bits in [8, 16, 32]:
            frac_bits = total_bits - int_bits
            if frac_bits > 0:
                resolution = 2**(-frac_bits)
                print(f"  {total_bits}-bit fixed point: {int_bits} integer + {frac_bits} fractional bits")
                print(f"    Resolution: {resolution:.8f}")
                print(f"    Range: [{-2**(int_bits-1):.1f}, {2**(int_bits-1)-resolution:.1f}]")
            else:
                print(f"  {total_bits}-bit: Not sufficient (need {int_bits} integer bits)")
        
        # Custom recommendation
        recommended_total = max(16, int_bits + 8)  # At least 8 fractional bits
        recommended_frac = recommended_total - int_bits
        print(f"\n  RECOMMENDED: {recommended_total}-bit fixed point")
        print(f"    {int_bits} integer + {recommended_frac} fractional bits")
        print(f"    Resolution: {2**(-recommended_frac):.8f}")
    
    def plot_distributions(self, x_sample=None, sample_size=500):
        """Plot weight, bias, and output distributions in separate figures"""
        
        # Figure 1: Parameter Distributions (2x2 layout)
        fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
        fig1.suptitle('Neural Network Parameter Analysis', fontsize=16)
        
        # Collect all weights and biases
        all_weights = []
        all_biases = []
        
        for layer_name in self.stats['weights'].keys():
            layer = None
            for l in self.model.layers:
                if l.name == layer_name:
                    layer = l
                    break
            if layer and hasattr(layer, 'get_weights'):
                weights, biases = layer.get_weights()
                all_weights.extend(weights.flatten())
                all_biases.extend(biases.flatten())
        
        # Plot weight distribution
        axes1[0, 0].hist(all_weights, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes1[0, 0].set_title('Weight Distribution (All Layers)')
        axes1[0, 0].set_xlabel('Weight Value')
        axes1[0, 0].set_ylabel('Frequency')
        axes1[0, 0].grid(True, alpha=0.3)
        
        # Plot bias distribution
        axes1[0, 1].hist(all_biases, bins=30, alpha=0.7, color='red', edgecolor='black')
        axes1[0, 1].set_title('Bias Distribution (All Layers)')
        axes1[0, 1].set_xlabel('Bias Value')
        axes1[0, 1].set_ylabel('Frequency')
        axes1[0, 1].grid(True, alpha=0.3)
        
        # Plot training history
        if self.history:
            axes1[1, 0].plot(self.history.history['accuracy'], label='Training', marker='o', linewidth=2)
            if 'val_accuracy' in self.history.history:
                axes1[1, 0].plot(self.history.history['val_accuracy'], label='Validation', marker='s', linewidth=2)
            axes1[1, 0].set_title('Training History')
            axes1[1, 0].set_xlabel('Epoch')
            axes1[1, 0].set_ylabel('Accuracy')
            axes1[1, 0].legend()
            axes1[1, 0].grid(True, alpha=0.3)
        
        # Plot per-layer max values
        layer_names = list(self.stats['weights'].keys())
        weight_maxes = [self.stats['weights'][name]['abs_max'] for name in layer_names]
        bias_maxes = [self.stats['biases'][name]['abs_max'] for name in layer_names]
        
        x_pos = np.arange(len(layer_names))
        width = 0.35
        axes1[1, 1].bar(x_pos - width/2, weight_maxes, width, label='Weights', alpha=0.8, color='blue')
        axes1[1, 1].bar(x_pos + width/2, bias_maxes, width, label='Biases', alpha=0.8, color='red')
        axes1[1, 1].set_title('Max Absolute Values by Layer')
        axes1[1, 1].set_xlabel('Layer')
        axes1[1, 1].set_ylabel('Max Absolute Value')
        axes1[1, 1].set_xticks(x_pos)
        axes1[1, 1].set_xticklabels([name.replace('hidden_', 'H').replace('output', 'Out') for name in layer_names])
        axes1[1, 1].legend()
        axes1[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Figure 2: Neuron Output Analysis
        if self.stats['neuron_outputs']:
            output_layer_names = list(self.stats['neuron_outputs'].keys())
            
            fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
            fig2.suptitle('Neuron Output Analysis', fontsize=16)
            
            output_maxes = [self.stats['neuron_outputs'][name]['abs_max'] for name in output_layer_names]
            zero_percentages = [self.stats['neuron_outputs'][name]['zero_percentage'] for name in output_layer_names]
            
            x_pos_out = np.arange(len(output_layer_names))
            
            # Max outputs
            axes2[0].bar(x_pos_out, output_maxes, alpha=0.8, color='green')
            axes2[0].set_title('Max Neuron Outputs by Layer')
            axes2[0].set_xlabel('Layer')
            axes2[0].set_ylabel('Max Output Value')
            axes2[0].set_xticks(x_pos_out)
            axes2[0].set_xticklabels([name.replace('hidden_', 'H').replace('output', 'Out') for name in output_layer_names])
            axes2[0].grid(True, alpha=0.3)
            
            # ReLU clipping
            bars = axes2[1].bar(x_pos_out, zero_percentages, alpha=0.8, color='orange')
            axes2[1].set_title('ReLU Clipping (% Zero Outputs)')
            axes2[1].set_xlabel('Layer')
            axes2[1].set_ylabel('Percentage of Zero Outputs')
            axes2[1].set_xticks(x_pos_out)
            axes2[1].set_xticklabels([name.replace('hidden_', 'H').replace('output', 'Out') for name in output_layer_names])
            axes2[1].grid(True, alpha=0.3)
            
            # Add percentage labels on bars
            for bar, pct in zip(bars, zero_percentages):
                height = bar.get_height()
                axes2[1].text(bar.get_x() + bar.get_width()/2., height + 1,
                             f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
            # Figure 3: Output Distributions
            if x_sample is not None:
                self.plot_output_distributions(x_sample, sample_size)
    
    def plot_output_distributions(self, x_sample, sample_size=500):
        """Plot individual layer output distributions"""
        if not self.stats['neuron_outputs']:
            return
            
        # Get sample data
        if len(x_sample) > sample_size:
            indices = np.random.choice(len(x_sample), sample_size, replace=False)
            x_plot = x_sample[indices]
        else:
            x_plot = x_sample
        
        # Force model building
        _ = self.model.predict(x_plot[:1], verbose=0)
        
        output_layer_names = list(self.stats['neuron_outputs'].keys())
        n_layers = len(output_layer_names)
        
        # Create figure with appropriate size
        cols = min(3, n_layers)
        rows = (n_layers + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        fig.suptitle('Layer Output Distributions', fontsize=16)
        
        # Handle single subplot case
        if n_layers == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if n_layers == 1 else axes
        else:
            axes = axes.flatten()
        
        colors = ['purple', 'brown', 'teal', 'navy', 'maroon', 'darkgreen']
        
        for i, layer_name in enumerate(output_layer_names):
            try:
                # Find the layer
                layer = None
                for l in self.model.layers:
                    if l.name == layer_name:
                        layer = l
                        break
                
                if layer:
                    # Create intermediate model
                    intermediate_model = models.Model(inputs=self.model.inputs,
                                                    outputs=layer.output)
                    outputs = intermediate_model.predict(x_plot, verbose=0)
                    
                    # Plot distribution
                    color = colors[i % len(colors)]
                    axes[i].hist(outputs.flatten(), bins=50, alpha=0.7, 
                               color=color, edgecolor='black')
                    axes[i].set_title(f'{layer_name.replace("hidden_", "Hidden ").replace("output", "Output")}')
                    axes[i].set_xlabel('Output Value')
                    axes[i].set_ylabel('Frequency')
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add statistics as text
                    mean_val = np.mean(outputs)
                    std_val = np.std(outputs)
                    zero_pct = np.mean(outputs == 0) * 100
                    max_val = np.max(outputs)
                    
                    stats_text = f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nMax: {max_val:.3f}\nZeros: {zero_pct:.1f}%'
                    axes[i].text(0.98, 0.98, stats_text,
                               transform=axes[i].transAxes,
                               verticalalignment='top',
                               horizontalalignment='right',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.5),
                               fontsize=9)
                    
            except Exception as e:
                print(f"Could not plot distribution for {layer_name}: {e}")
                axes[i].text(0.5, 0.5, f'Error plotting\n{layer_name}', 
                           transform=axes[i].transAxes, ha='center', va='center')
        
        # Hide unused subplots
        for i in range(n_layers, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()

# Example usage
def main():
    # Configuration - modify these parameters
    LAYER_SIZES = [16, 10]  # Hidden layers + output layer
    EPOCHS = 5
    
    print("Neural Network Resolution Analyzer for FPGA Implementation")
    print("="*60)
    print(f"Architecture: Input(784) -> {' -> '.join(map(str, LAYER_SIZES))}")
    
    # Create and configure the analyzer
    analyzer = NeuralNetworkAnalyzer(LAYER_SIZES)
    analyzer.build_model()
    
    # Train the model
    (x_train, y_train), (x_test, y_test) = analyzer.train_model(epochs=EPOCHS)
    
    # Analyze weights and biases
    analyzer.analyze_weights_and_biases()
    
    # Analyze neuron outputs
    analyzer.analyze_neuron_outputs(x_test)
    
    # Get FPGA resolution suggestions
    analyzer.suggest_fpga_resolution()
    
    # Plot distributions
    analyzer.plot_distributions(x_test)
    
    return analyzer

if __name__ == "__main__":
    # Run the analysis
    analyzer = main()
    
    # You can also experiment with different architectures:
    print("\n" + "="*60)
    print("To experiment with different architectures, modify LAYER_SIZES:")
    print("Examples:")
    print("  [64, 32, 10] - Smaller network")
    print("  [256, 128, 64, 10] - Deeper network") 
    print("  [512, 256, 10] - Wider network")