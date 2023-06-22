# Pythia
A modular toolkit, implemented in Python and organized in Jupyter Notebooks. It employs fingerprints Mordred descriptors and precalculated QM descriptors as input features for shallow learners and ensemble models for regression and classification tasks. 

# Supported functionality
## Tasks:
- Classification
- Regression

## Data types
- SMILES strings
- Precalculated descriptors

## Modules:
- classification metrics: calculation of confusion matrix, accuracy, g-mean, precision, recall, generalized f, MCC, AUC
- fingerprints: generation of Morgan, rdkit, atom pair, torsion fingerprints and MACCS keys with rd
- molecules and images: SMILES to molecules and images with rdkit
- plot sklearn: plot of parity plots, ROC curves, confusion matrix with matplotlib
- scaling: z, min-max, logarithmic scaling
- workflow functions: correlation tests, training for regression and classification, ensemble learning with sklearn

We are working on populating this package with more models and other building blocks.

## Notebooks:
- data analysis: data exploration, visualization, scaling
- regression-fingerprints: regression with fingerprints, data set split, ensemble models
- regression-Mordred: regression with Mordred, feature elimination techniques, data set split
- regression-DFT: regression with DFT descriptors, PCA, data set split
- classification-fingerprints: classification with fingerprints, feature exploration, synthetic data, data set split
- classification-Mordred: classification with Mordred, feature elimination and exploration, synthetic data, data set split
- classification-DFT: classification with DFT descriptors, synthetic data, data set split, interpretability

Please mix and match Notebook cells and Modules. The world is your oyster, the sky is the limit.

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
