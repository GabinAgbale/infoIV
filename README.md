# InfoIV

Implementation of [Recovering Causal Features for Instrumental Variable Regression with Contrastive Learning](https://openreview.net/forum?id=3qOdzEUKHZ&noteId=3qOdzEUKHZ)


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
