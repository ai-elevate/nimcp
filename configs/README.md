# NIMCP Configuration Files

This directory contains brain configuration files for NIMCP. Configuration files allow you to define brain architecture, training parameters, plasticity settings, and other options without hardcoding them in your application.

## Supported Formats

- **YAML** (`.yaml`, `.yml`) - Human-friendly, recommended
- **JSON** (`.json`) - Machine-friendly, programmatic generation

## Configuration Schema

### Basic Structure

```yaml
brain:
  # Basic identification
  name: "my_brain"              # Brain identifier
  size: small                   # tiny, small, medium, large
  task: classification          # classification, regression, pattern_matching, sequence, association

  # Network architecture
  architecture:
    num_inputs: 784             # Input dimensions
    num_outputs: 10             # Output dimensions
    num_hidden: 256             # Hidden layer size
    learning_rate: 0.01         # Learning rate (0.001 - 0.1)

  # Training parameters
  training:
    max_epochs: 100             # Maximum training epochs
    batch_size: 32              # Mini-batch size
    validation_split: 0.2       # Validation data ratio
    early_stopping: true        # Enable early stopping
    patience: 10                # Early stopping patience

  # Plasticity mechanisms
  plasticity:
    enable_bcm: true            # Bienenstock-Cooper-Munro rule
    bcm_tau: 1000.0             # BCM time constant
    enable_stdp: false          # Spike-timing dependent plasticity
    stdp_window: 20.0           # STDP time window (ms)

  # Ethics configuration
  ethics:
    enabled: false              # Enable ethical constraints
    golden_rule_threshold: 0.0  # Golden Rule threshold
    empathy_weight: 0.5         # Empathy consideration weight

  # Model persistence
  model_path: "/tmp/brain.model"  # Default save path
  checkpoint_interval: 10         # Save every N epochs
```

## Brain Sizes

| Size   | Neurons | Memory  | Inference Time | Use Case                |
|--------|---------|---------|----------------|-------------------------|
| tiny   | 100     | <1MB    | ~0.1ms         | Prototyping, testing    |
| small  | 1K      | ~10MB   | ~0.5ms         | Small datasets, mobile  |
| medium | 10K     | ~50MB   | ~5ms           | General purpose         |
| large  | 100K    | ~500MB  | ~50ms          | Large datasets, complex |

## Task Types

- **classification** - Multi-class classification (e.g., image recognition)
- **regression** - Continuous value prediction (e.g., price estimation)
- **pattern_matching** - Pattern recognition (e.g., anomaly detection)
- **sequence** - Temporal sequence learning (e.g., time series)
- **association** - Association learning (e.g., recommendation)

## Usage Examples

### C

```c
#include "nimcp.h"

int main() {
    nimcp_init();

    // Load brain from config file
    nimcp_brain_t brain = nimcp_brain_create_from_config("configs/brain_classifier_small.yaml");

    // Use brain...
    float features[784] = { /* ... */ };
    char label[64];
    float confidence;

    nimcp_brain_predict(brain, features, 784, label, &confidence);

    nimcp_brain_destroy(brain);
    nimcp_shutdown();
}
```

### Python

```python
import nimcp

nimcp.init()

# Load from config
brain = nimcp.Brain.from_config("configs/brain_classifier_small.yaml")

# Use brain...
label, confidence = brain.predict(features)

brain.destroy()
nimcp.shutdown()
```

### Ruby

```ruby
require 'nimcp'

NIMCP.init

# Load from config
brain = NIMCP::Brain.from_config('configs/brain_classifier_small.yaml')

# Use brain...
result = brain.predict(features)

brain.destroy
NIMCP.shutdown
```

## Example Configurations

See the example files in this directory:

- `brain_simple.json` - Minimal configuration for quick testing
- `brain_classifier_small.yaml` - Image classification setup
- `brain_regression_medium.yaml` - Regression task with medium brain

## Customization Tips

1. **Start Small**: Begin with `size: tiny` for prototyping
2. **Tune Learning Rate**: Typical range is 0.001 to 0.1
3. **Enable BCM**: Recommended for most tasks
4. **Use Early Stopping**: Prevents overfitting
5. **Adjust Batch Size**: Larger for faster training, smaller for better convergence

## Validation

Configuration files are validated when loaded. Common errors:

- Invalid size or task value
- Missing required fields (name, num_inputs, num_outputs)
- Out-of-range parameters (e.g., negative learning rate)
- File not found or unreadable

Error messages will indicate the specific problem.

## Best Practices

1. **Version Control**: Keep configs in git with your project
2. **Documentation**: Add comments explaining non-obvious choices
3. **Naming**: Use descriptive names (e.g., `brain_mnist_classifier_v2.yaml`)
4. **Separation**: One config per distinct task/model
5. **Environment**: Use environment variables for paths if needed

## Advanced Features

Future additions will include:

- Multi-layer architecture specification
- Custom activation functions
- Regularization options (dropout, L1/L2)
- Advanced ethics policies
- Distributed training settings
- Hyperparameter search configurations
