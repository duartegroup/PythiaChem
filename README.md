# Pythia
Simple ML tool

# Supported functionality
## Tasks:
- Classification
- Regression

## Data types
- SMILES strings

## Modules:
- to be completed

We are working on populating this package with more models and other building blocks.

## Installation
To use Pythia, you need to install several third-party softwares including rdkit, sklearn, mordred, etc.
```Bash
conda create -n pythia
conda activate pythia
conda install -c rdkit rdkit
pip install -U imbalanced-learn
pip install -U scikit-learn
pip install mlxtend
pip install 'mordred[full]'
pip install seaborn
pip install scikit-plot
pip install notebook
```
To make environment available in jupyter notebook
```Bash
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=pythia
```

### Installation by pip
```Bash
git clone https://github.com/duartegroup/Pythia.git
pip install .
```
