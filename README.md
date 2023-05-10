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
### Install by conda
To use Pythia, you need to install several third-party softwares.
```Bash
conda create -n pythia
conda activate pythia

conda install -c conda-forge -c omnia openmm=7.4.2
conda install -c conda-forge mdanalysis mdtraj parmed
conda install -c conda-forge acpype openmpi mpi4py gromacs
conda install -c conda-forge matplotlib
pip install unigbsa gmx_MMPBSA lickit
```
