# Info-IV Project

This repository contains the **Info-IV** project.

## Setup

Follow these steps to set up the project environment:

### 1. Create the Conda environment
```bash
conda env create -f environment.yml
conda activate info-iv
```
### 2. Install local packages 
```bash
pip install -e .
```

### 3. Run experiment (example)
```bash
python cli/train_aux_imca.py
```
