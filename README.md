# Neural Architecture Search using Genetic Algorithm

## Project Overview
This project implements Neural Architecture Search (NAS) using a Genetic Algorithm to optimize Convolutional Neural Network architectures for image classification tasks. The implementation features roulette-wheel selection and weighted complexity penalties for different layer types.

## Key Features
- **Genetic Algorithm-based Search**
  - Population-based evolution of CNN architectures
  - Roulette-wheel selection mechanism
  - Crossover and mutation operators
  - Elitism to preserve best solutions

- **Advanced Fitness Evaluation**
  - Weighted complexity penalties
  - Separate weights for convolutional and fully connected layers
  - Kernel size consideration in complexity calculations
  - Normalized parameter penalties

- **Search Space**
  - Convolutional layers: 1-4 layers
  - Filter sizes: 16, 32, 64, 128
  - Kernel sizes: 3x3, 5x5, 7x7
  - Pooling types: Max, Average
  - Activation functions: ReLU, Leaky ReLU
  - Fully connected units: 64, 128, 256, 512

## Implementation Details

### Complexity Penalties
The fitness function incorporates sophisticated complexity penalties:
```python
# Convolutional layers: Higher weight due to sliding window operations
conv_weight = 0.1

# Fully connected layers: Base weight for matrix multiplication
fc_weight = 0.01

# Complexity calculation includes kernel operations
complexity_penalty = (conv_weight * conv_params + fc_weight * fc_params) / total_params
```

### Selection Mechanism
Roulette-wheel selection implemented for parent selection:
- Probability proportional to fitness
- Normalized selection weights
- Fallback mechanism for zero fitness cases
``` markdown
## Project Structure
```
├── model_ga.py # Genetic Algorithm implementation ├── model_cnn.py # CNN model architecture ├── nas_run.py # Main execution script └── outputs/ # Generated outputs and results └── run_*/ # Results for each run
``` 

## Requirements
- Python 3.12.7
- PyTorch
- NumPy
- Other dependencies listed in requirements.txt

## Usage
```bash
python nas_run.py --population 20 --generations 10 --mutation_rate 0.2 --crossover_rate 0.7
```

### Parameters
- `--population`: Size of population (default: 20)
- `--generations`: Number of generations (default: 10)
- `--mutation_rate`: Mutation probability (default: 0.2)
- `--crossover_rate`: Crossover probability (default: 0.7)

## Output
The algorithm generates:
- Architecture configurations for each generation
- Fitness scores and accuracy metrics
- Best architecture found during the search
- JSON files containing architecture details

## Results Format
Results are saved in JSONL format in the `outputs` directory:
```json
{
    "num_conv": 3,
    "conv_configs": [
        {"filters": 32, "kernel_size": 3},
        {"filters": 64, "kernel_size": 5},
        {"filters": 128, "kernel_size": 3}
    ],
    "pool_type": "max",
    "activation": "relu",
    "fc_units": 256
}
```

## Performance Considerations
- Complexity penalties are normalized by total parameters
- Convolutional operations weighted higher due to sliding window computations
- Kernel size included in complexity calculations
- Early stopping implemented during training
```
