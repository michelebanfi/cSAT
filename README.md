# cSAT: Causal Structure and SAT Solvers

This project implements causal discovery algorithms using SAT (Boolean satisfiability) solvers, including both classical and quantum approaches. It provides tools for discovering causal relationships in data using algorithms such as PC and FCI.

> `venv\Scripts\activate` to activate the virtual environment on Windows
> `source venv/bin/activate` to activate the virtual environment on Linux

## Overview

cSAT combines causal inference techniques with SAT solving to discover and validate causal structures in data. The project supports:

- PC (Peter-Clark) algorithm for causal discovery
- FCI (Fast Causal Inference) algorithm for causal discovery with latent variables
- Classical SAT solving
- Quantum SAT solving using Qiskit

## Installation

### Prerequisites

- Python 3.8+
- numpy
- pandas
- matplotlib
- causallearn
- pydot
- qiskit (for quantum SAT solving)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd cSAT

# Install dependencies
pip install numpy pandas matplotlib causallearn pydot qiskit
```

## Project Structure

```
cSAT/
├── PC.py                # PC algorithm implementation
├── FCI.py               # FCI algorithm implementation
├── utils.py             # Utility functions for the project
├── SAT/
│   ├── classical.py     # Classical SAT solver
│   └── quantum.py       # Quantum SAT solver implementation
└── output/              # Generated visualizations and results
    ├── PC/              # PC algorithm outputs
    └── FCI/             # FCI algorithm outputs
```

## Usage

### PC Algorithm

The PC algorithm discovers causal relationships assuming no latent confounders:

```python
python PC.py
```

This will:

1. Generate synthetic causal data
2. Run the PC algorithm
3. Convert the discovered structure to SAT constraints
4. Solve using both classical and quantum SAT solvers
5. Visualize the results in the `output/PC/` directory

### FCI Algorithm

The FCI algorithm extends PC for cases with latent confounders:

```python
python FCI.py
```

This will perform similar steps to PC but accounting for possible hidden common causes.

## Outputs

The algorithms generate several visualizations:

- `PC_output.png` / `FCI_output.png` - Direct output from the causal discovery algorithm
- `classical_PC_output.png` / `classical_FCI_output.png` - Final graph from classical SAT solver
- `quantum_PC_outputs.png` / `quantum_FCI_outputs.png` - Grid of possible solutions from quantum SAT solver

## Implementation Details

### Causal Discovery

The project uses the `causallearn` library to implement PC and FCI algorithms, which discover causal structures from observational data.

### SAT Solvers

- **Classical SAT**: Deterministic approach using traditional SAT solving techniques
- **Quantum SAT**: Quantum computing approach using Grover's algorithm via Qiskit

### Visualization

The project uses `pydot` and `matplotlib` to visualize causal graphs and compare solutions from different solvers.

## License

[Your License Information]
