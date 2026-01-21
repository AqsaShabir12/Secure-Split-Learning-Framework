# Encrypted Split Learning Framework 

#### Forked from https://github.com/PaluMacil/gophernet.

## Overview

This repository contains the implementation of the Split Learning system, leveraging homomorphic encryption (HE) to secure the model without compromising training efficiency. 
 
## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/Encrypted Split Learning Framework.git 
   ```
2. **Setup Environment**:
   Ensure you have Go installed on your machine, as the main implementation is in Go. Python is required for some utility scripts.

3. **Run Experiments**:
   Navigate to the project directory and execute:
   ```bash
   go run main.go train digits -layers=4 -hidden=128,32 -epochs=10 -rate=.1
   ```

## Python Simulations

The `python_simulation` code trains simple neural networks on the MNIST, BCW (Breast Cancer Wisconsin), and CREDIT datasets using a configurable architecture. This section allows for quick experimentation with different configurations and fast accuracy assessments.

To run any of the main scripts, use:
```bash
python main.py --input 784 --hidden "128,32" --output 10 --epochs 10 --rate 0.01 --batch 60
```



