# PythiaCHEM (PYthon Toolkit for macHIne leArning in CHEMistry)
PythiaChem is a modular toolkit, implemented in Python and organized in Jupyter Notebooks. It employs fingerprints Mordred descriptors and precalculated QM descriptors as input features for shallow learners and ensemble models for regression and classification tasks. 

Paper: https://chemrxiv.org/engage/chemrxiv/article-details/6504251fb338ec988a83e057

<img width="1002" alt="Pythia" src="https://github.com/duartegroup/PythiaChem/assets/46780283/be55d979-6bee-41c6-9923-c5223bd4233d">


## Supported functionality
### Tasks:
- Classification
- Regression

### Data types
- SMILES strings
- Precalculated descriptors

### Modules:
- classification metrics: calculation of confusion matrix, accuracy, g-mean, precision, recall, generalized f, MCC, AUC
- fingerprints generation: generation of Morgan, rdkit, atom pair, torsion fingerprints and MACCS keys with rd
- molecules and structures: SMILES to molecules and images with rdkit
- plots: plot of parity plots, ROC curves, confusion matrix with matplotlib
- scaling: z, min-max, logarithmic scaling
- workflow functions: correlation tests, training for regression and classification, ensemble learning with sklearn

We are working on populating this package with more models and other building blocks.

### Notebooks:
*Please refer to 'Example-enantioselective-Strecker-synthesis' for the full desmontration of all notebooks.*
- data analysis: data exploration, visualization, scaling
- regression-fingerprints: regression with fingerprints, data set split, ensemble models
- regression-Mordred: regression with Mordred, feature elimination techniques, data set split
- regression-DFT: regression with DFT descriptors, PCA, data set split
- classification-fingerprints: classification with fingerprints, feature exploration, synthetic data, data set split
- classification-Mordred: classification with Mordred, feature elimination and exploration, synthetic data, data set split
- classification-DFT: classification with DFT descriptors, synthetic data, data set split, interpretability

Please mix and match Notebook cells and Modules. The world is your oyster, the sky is the limit.
Use the .csv files to run the Notebooks and use the comments to assist you.

## Installation
From environment.yml file:
```Bash
git clone https://github.com/duartegroup/PythiaChem.git
cd PythiaChem
conda env create -f environment.yml
conda activate pythiachem
conda install -c anaconda ipykernel -y
python -m ipykernel install --user --name=pythiachem
pip install -e .
```

If installation with environment.yml fails, you can install manually with the following steps:
```Bash
conda create -n pythiachem -y
conda activate pythiachem
pip install rdkit 'mordred[full]' mlxtend imbalanced-learn scikit-learn scikit-plot seaborn notebook matplotlib matplotlib_venn
git clone https://github.com/duartegroup/PythiaChem.git
cd PythiaChem
pip install -e .
conda install -c anaconda ipykernel -y
python -m ipykernel install --user --name=pythiachem
```
If you would like to run the unit tests please make a folder named "test_directory" in the tests folder and in a terminal run
```Bash
python -m unittest *.py
```
