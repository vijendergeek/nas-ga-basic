# Assignment-2: NAS with Genetic Algorithm (CIFAR-10)

This repository contains the starter code for Neural Architecture Search (NAS) on CIFAR-10 using a Genetic Algorithm (GA), along with the **minimal modifications** required for Assignment-2:
---

## 1. Environment Setup

### 1.1 Prerequisites

- Python **3.8+** (tested with ≥3.8)
- `pip` (or `pip3`)
- (Optional) CUDA-capable GPU + drivers if you want to run on GPU  
  The code automatically uses `cuda` if available; otherwise it falls back to `cpu`.

### 1.1 Install dependencies

Install the minimal packages manually:

```bash
pip install torch torchvision torchaudio tqdm numpy
```

(PyTorch installation command may vary depending on your CUDA version; use the official PyTorch site if needed.)

---

## 2. Dataset

The code uses **CIFAR-10** via `torchvision.datasets.CIFAR10`.

- On the first run, CIFAR-10 will be **automatically downloaded** to `./data` (you will see log messages like `Downloading ... cifar-10-python.tar.gz` and `Extracting ...`).
- Subsequent runs will reuse the local copy and print `Files already downloaded and verified`.

No manual dataset download is required.

---
## Hardware-Specific Configurations

### CPU-Only Mode Adjustments
When running on CPU-only systems, the following configurations are automatically applied to reduce computation time and resource usage:

#### nas_run.py changes:
- Training samples reduced from 5000 to 1000
- Validation subset reduced from 1000 to 200
- GeneticAlgorithm configurations adjusted:
  - Population size reduced from 10 to 5
  - Number of generations reduced from 5 to 3

#### model_ga.py changes:
- Early stopping patience reduced from 10 to 5 in `evaluate_fitness` method
---
## 3. Running the Code

The main entry point is `nas_run.py`.

From the project root:

```bash
python nas_run.py
```

Typical console output will look like:

```text
Using device: cpu
Files already downloaded and verified
Starting with 5 Population:
[Arch(conv=1, acc=0.0000), Arch(conv=2, acc=0.0000), ...]

============================================================
Generation 1/3
============================================================
Evaluating architecture 1/5... Fitness: ...
...
Roulette-Wheel Selection Probabilities:
Architecture 1: Original Fitness = ..., Shifted Fitness = ..., Probability = ...
...
FINAL BEST ARCHITECTURE
============================================================
Genes: {...}
Accuracy: ...
Fitness: ...
Total parameters: ...
Model architecture:
CNN(
  ...
)
```

---

## 4. Code Structure

- `model_cnn.py`  
  Defines the CNN architecture (`CNN` class) used for evaluation.

- `model_ga.py`  
  Implements the Genetic Algorithm: population initialization, selection, crossover, mutation, and fitness evaluation.

- `nas_run.py`  
  Orchestrates the NAS experiment: sets hyperparameters (population size, generations, etc.), runs GA, and prints the best architecture.
- `nas_run_rw_weighted_penality.log`
A log file for reference generated from latest run on CPU.


---

## 6. Summary

To reproduce the assignment:

1. Set up the Python environment (Section 1).
2. Run `python nas_run.py` (Section 3).
3. It will create an outputs/run_1 folder which will contain nas_run.log file showcasing following:
   - Roulette-wheel selection probabilities.
   - Fitness analysis with original vs. weighted complexity penalties.
   - Final best architecture (genes, accuracy, fitness, and parameter count).


Only **minimal changes** were made to `model_ga.py`:

- `selection` (tournament → roulette-wheel)
- `evaluate_fitness` (new weighted complexity penalty and fitness definition)
