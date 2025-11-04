# NIMCP Configuration Templates

This directory contains production-ready YAML configuration templates for common use cases. Each template is extensively documented with usage examples, best practices, and deployment guidelines.

## Available Templates

### 1. Image Classification (MNIST)
**File:** `image_classification_mnist.yaml`

**Use Case:** Handwritten digit recognition, visual pattern classification

**Specifications:**
- **Size:** Medium (10K neurons, ~50MB)
- **Task:** Classification
- **Inputs:** 784 (28×28 pixel images)
- **Outputs:** 10 (digit classes 0-9)
- **Expected Accuracy:** 95-98%
- **Training Time:** 10-30 minutes

**Applications:**
- Document OCR
- Form processing
- Postal code recognition
- Check digit verification

**Start Here If You Need:** Computer vision, image recognition, CNN-alternative

---

### 2. Time Series Forecasting
**File:** `time_series_forecasting.yaml`

**Use Case:** Predicting future values from historical sequences

**Specifications:**
- **Size:** Small (1K neurons, ~10MB)
- **Task:** Sequence learning
- **Inputs:** 100 (time window)
- **Outputs:** 10 (forecast horizon)
- **Inference:** <1ms per prediction
- **Training Time:** 5-15 minutes

**Applications:**
- Stock price prediction
- Weather forecasting
- Energy demand forecasting
- IoT sensor prediction
- Sales forecasting

**Start Here If You Need:** Temporal patterns, sequential data, LSTM-alternative

---

### 3. Anomaly Detection (IoT)
**File:** `anomaly_detection_iot.yaml`

**Use Case:** Detecting abnormal patterns in sensor data

**Specifications:**
- **Size:** Tiny (100 neurons, <1MB)
- **Task:** Pattern matching
- **Inputs:** 20 (sensor features)
- **Outputs:** 2 (normal/anomaly)
- **Inference:** ~0.1ms (edge-deployable)
- **Training Time:** <5 minutes

**Applications:**
- Predictive maintenance
- Security monitoring
- Quality control
- Network intrusion detection
- Health monitoring

**Start Here If You Need:** Edge deployment, real-time monitoring, outlier detection

---

### 4. Recommendation System
**File:** `recommendation_system.yaml`

**Use Case:** Personalized content/product recommendations

**Specifications:**
- **Size:** Medium (10K neurons, ~50MB)
- **Task:** Association learning
- **Inputs:** 200 (user + item features)
- **Outputs:** 5 (recommendation scores)
- **Ethics:** Enabled (prevents harmful recommendations)
- **Training Time:** 20-40 minutes

**Applications:**
- E-commerce product recommendations
- Content recommendations (articles, videos)
- Music/playlist generation
- Job/candidate matching
- Friend suggestions

**Start Here If You Need:** Collaborative filtering, personalization, ethical AI

---

### 5. Sentiment Analysis (NLP)
**File:** `sentiment_analysis_nlp.yaml`

**Use Case:** Analyzing text sentiment and emotions

**Specifications:**
- **Size:** Small (1K neurons, ~10MB)
- **Task:** Classification
- **Inputs:** 300 (text embedding)
- **Outputs:** 3 (positive/negative/neutral)
- **Ethics:** Enabled (bias prevention)
- **Training Time:** 10-20 minutes

**Applications:**
- Customer review analysis
- Social media monitoring
- Brand sentiment tracking
- Chatbot emotion detection
- Content moderation

**Start Here If You Need:** Text classification, NLP, opinion mining

---

## Quick Start Guide

### 1. Choose Your Template

Select based on your task type:

| Task Type | Template |
|-----------|----------|
| Images, photos | Image Classification |
| Numbers over time | Time Series Forecasting |
| Detect unusual patterns | Anomaly Detection |
| Suggest items to users | Recommendation System |
| Analyze text opinions | Sentiment Analysis |

### 2. Copy and Customize

```bash
# Copy template to your project
cp configs/templates/image_classification_mnist.yaml my_project/brain_config.yaml

# Edit configuration
nano my_project/brain_config.yaml
```

### 3. Modify Key Parameters

```yaml
brain:
  name: "my_custom_brain"  # Your brain name

  architecture:
    num_inputs: ???         # YOUR input dimension
    num_outputs: ???        # YOUR output classes
    num_hidden: ???         # Adjust for complexity
```

### 4. Load and Use

**C:**
```c
nimcp_brain_t brain = nimcp_brain_create_from_config("my_project/brain_config.yaml");
```

**Python:**
```python
brain = nimcp.Brain.from_config("my_project/brain_config.yaml")
```

**Ruby:**
```ruby
brain = NIMCP::Brain.from_config('my_project/brain_config.yaml')
```

## Template Selection Guide

### By Performance Requirements

| Requirement | Recommended Template |
|-------------|---------------------|
| **Edge/Mobile** (<1MB) | Anomaly Detection (Tiny) |
| **Fast Inference** (<1ms) | Time Series, Sentiment Analysis |
| **High Accuracy** | Image Classification (Medium/Large) |
| **Low Latency** | Anomaly Detection |
| **Scalability** | Recommendation System |

### By Data Type

| Data Type | Recommended Template |
|-----------|---------------------|
| **Images** | Image Classification |
| **Time-ordered numbers** | Time Series Forecasting |
| **Sensor readings** | Anomaly Detection |
| **User-item pairs** | Recommendation System |
| **Text/documents** | Sentiment Analysis |

### By Deployment Target

| Target | Recommended Template |
|--------|---------------------|
| **IoT Device** (ESP32, Arduino) | Anomaly Detection |
| **Mobile App** (iOS, Android) | Sentiment Analysis |
| **Web Server** | Recommendation System |
| **Desktop** | Image Classification |
| **Cloud** | Any (all supported) |

## Customization Tips

### Adjusting Brain Size

```yaml
# For faster inference but lower accuracy
size: tiny    # 100 neurons, <1MB

# Balanced performance
size: small   # 1K neurons, ~10MB

# Better accuracy
size: medium  # 10K neurons, ~50MB

# Maximum accuracy (slower)
size: large   # 100K neurons, ~500MB
```

### Tuning Learning Rate

```yaml
architecture:
  # Conservative (stable but slow)
  learning_rate: 0.001

  # Moderate (recommended)
  learning_rate: 0.01

  # Aggressive (fast but may be unstable)
  learning_rate: 0.1
```

### Enabling Advanced Features

```yaml
# Synaptic plasticity for adaptation
plasticity:
  enable_bcm: true     # Adaptive thresholds
  enable_stdp: true    # Temporal learning (for sequences)

# Ethical constraints
ethics:
  enabled: true        # Prevent harmful predictions
  empathy_weight: 0.7  # High empathy
```

## Common Modifications

### Changing Task Type

All templates can be adapted for different tasks:

```yaml
# Original: Classification
task: classification

# Modify to:
task: regression          # Continuous values
task: pattern_matching    # Binary anomaly
task: sequence           # Temporal patterns
task: association        # Relationships
```

### Multi-Class vs Binary

```yaml
# Binary classification (2 classes)
architecture:
  num_outputs: 2

# Multi-class (N classes)
architecture:
  num_outputs: 10  # Adjust to your number of classes
```

### Extending to Multi-Output

```yaml
# Single output task
architecture:
  num_outputs: 1

# Multi-task learning (predict multiple things)
architecture:
  num_outputs: 20  # E.g., 10 emotions + 10 topics
```

## Performance Comparison

| Template | Size | Memory | Inference | Training | Accuracy |
|----------|------|--------|-----------|----------|----------|
| MNIST | Medium | 50MB | ~5ms | 20min | 95-98% |
| Time Series | Small | 10MB | <1ms | 10min | 75-85% |
| Anomaly | Tiny | <1MB | 0.1ms | 3min | 90-95% |
| Recommendation | Medium | 50MB | ~5ms | 30min | 80-90% |
| Sentiment | Small | 10MB | <1ms | 15min | 85-90% |

## Best Practices

1. **Start Simple**
   - Begin with default parameters
   - Use tiny/small brain for prototyping
   - Scale up once working

2. **Validate Thoroughly**
   - Always use validation_split
   - Enable early_stopping
   - Test on unseen data

3. **Monitor Training**
   - Check for overfitting
   - Adjust patience if needed
   - Save checkpoints regularly

4. **Document Changes**
   - Add YAML comments
   - Version your configs
   - Track what worked

5. **Ethics Consideration**
   - Enable ethics for user-facing apps
   - Test for bias
   - Consider societal impact

## Troubleshooting

### Brain Not Learning
- **Decrease** learning_rate (try 0.001)
- **Increase** max_epochs
- **Check** data normalization (0-1 range)
- **Verify** labels are correct

### Overfitting
- **Enable** early_stopping
- **Increase** validation_split (0.2-0.3)
- **Decrease** num_hidden
- **Add** more training data

### Slow Training
- **Increase** batch_size
- **Decrease** brain size
- **Reduce** num_hidden
- **Use** smaller dataset for testing

### Poor Accuracy
- **Increase** brain size
- **Increase** num_hidden
- **Tune** learning_rate
- **Check** feature engineering
- **Add** more training data

## Advanced Topics

### Ensemble Methods
Combine multiple brains:
```bash
# Train 3 different configurations
brain1 = create_from_config("config_small.yaml")
brain2 = create_from_config("config_medium.yaml")
brain3 = create_from_config("config_large.yaml")

# Vote or average predictions
final_prediction = majority_vote([brain1, brain2, brain3])
```

### Transfer Learning
Use pre-trained brain:
```yaml
# In new config
brain:
  # Start from pre-trained model
  model_path: "/models/pretrained_base.brain"

  # Fine-tune with new data
  training:
    max_epochs: 20  # Fewer epochs for fine-tuning
    learning_rate: 0.0001  # Lower learning rate
```

### Hyperparameter Search
Automated tuning:
```python
# Generate multiple configs
for lr in [0.001, 0.01, 0.1]:
    for hidden in [100, 200, 400]:
        config = generate_config(learning_rate=lr, num_hidden=hidden)
        brain = train_and_evaluate(config)
        if brain.accuracy > best_accuracy:
            best_config = config
```

## Contributing

Have a useful template? Submit a pull request!

**Template Requirements:**
- Comprehensive documentation
- Real-world use case
- Performance benchmarks
- Usage example
- Hyperparameter guidance

## Support

- **Documentation:** See `docs/CONFIG_SYSTEM.md`
- **Examples:** See `examples/config_based_brain.c`
- **Issues:** https://github.com/nimcp/nimcp/issues
- **Discussions:** https://github.com/nimcp/nimcp/discussions

## License

Same as main NIMCP project (MIT License)
