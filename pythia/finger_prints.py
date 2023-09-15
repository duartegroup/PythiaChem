#!/usr/bin/env python

# Python packages and utilities
import os
from datetime import datetime
import pandas as pd
import numpy as np
import scipy
from scipy import spatial

#RDKit
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import DataStructs
from rdkit.Chem.AllChem import GetHashedAtomPairFingerprintAsBitVect

# disp
from IPython.display import SVG, Image, display

# Logging
import logging


# own modules
from . import molecules_and_images as mi

def rdkit_fingerprints(smiles):
    """
    Function to get RDKit fingerprints
    :param smiles: list - smiles representation of the molecules to make fingerprints of
    """

    log = logging.getLogger(__name__)

    mols = [mi.smiles_to_molcule(smile) for smile in smiles]
    fps = [Chem.RDKFingerprint(mol) for mol in mols]

    return fps
def maccskeys_fingerprints(smiles):
    """
    Function to get MACCS fingerprints
    :param smiles: list - smiles representation of the molecules to make fingerprints of
    """
    
    log = logging.getLogger(__name__)
    
    mols = [mi.smiles_to_molcule(smile) for smile in smiles]
    fps = [MACCSkeys.GenMACCSKeys(mol) for mol in mols]
    A = np.mat(fps)
    df = pd.DataFrame(data=A)
    return fps,df

def atom_pair_fingerprints(smiles, nBits = 1024, bit_vec=False, return_only_non_zero=False, log_explanation=False):
    """
    Function to get atom pair fingerprints
    :param smiles: list - smiles representation of the molecules to make fingerprints of
    :param nBits: number of bits for the bit vector representation of the fingerprint
    :param bit_vec: true/false - get the bit vector representation of the fingerprint
    :param return_only_non_zero: true/false - get only the non-zero elements of the fingerprint as it is sparse
    :param log_explanation: true/false - print an explanation of the bit see http://www.rdkit.org/docs/GettingStartedInPython.html#atom-pairs-and-topological-torsions
    """
    
    log = logging.getLogger(__name__)
    
    mols = [mi.smiles_to_molcule(smile) for smile in smiles]
    
    if bit_vec is True:
        #fps = [Pairs.GetAtomPairFingerprintAsBitVect(mol) for mol in mols]
        fps = [GetHashedAtomPairFingerprintAsBitVect(mol, nBits = nBits) for mol in mols]
    else:
        fps = [Pairs.GetAtomPairFingerprint(mol) for mol in mols]
    
    if return_only_non_zero is True:
        fps = [fp.GetNonzeroElements() for fp in fps]
        
    if log_explanation is True:
        for inx, fp in enumerate(fps):
            log.info("Atom pair finger print number {}:\n\texplanation: {}".format(inx, Pairs.ExplainPairScore(fp)))
    
    return fps

def torsion_fingerprints(smiles, nBits = 1024, bit_vec=False, return_only_non_zero=False, log_explanation=False):
    """
    Function to get topological fingerprints
    :param smiles: list - smiles representation of the molecules to make fingerprints of
    :param nBits: number of bits for the bit vector representation of the fingerprint
    :param bit_vec: true/false - get the bit vector representation of the fingerprint
    :param return_only_non_zero: true/false - get only the non-zero elements of the fingerprint as it is sparse
    :param log_explanation: true/false - print an explanation of the bit see http://www.rdkit.org/docs/GettingStartedInPython.html#atom-pairs-and-topological-torsions
    """
    
    log = logging.getLogger(__name__)
    
    mols = [mi.smiles_to_molcule(smile) for smile in smiles]
    
    if bit_vec is True:
        fps = [rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits = nBits) for mol in mols]
    else:
        fps = [rdMolDescriptors.GetTopologicalTorsionFingerprint(mol) for mol in mols]
    
    if return_only_non_zero is True:
        fps = [fp.GetNonzeroElements() for fp in fps]
        
    if log_explanation is True:
        for inx, fp in enumerate(fps):
            log.info("Topological torsion finger print number {}:\n\texplanation: {}".format(inx, Pairs.ExplainPairScore(fp)))
    
    return fps

def morgan_fingerprints(smiles, radius=2, n_bits=1024, bit_vec=True, feature_invarients=False, return_only_non_zero=True, log_explanation=False):
    """
    Function to the Morgan/circular fingerprints
    :param smiles: list - smiles representation of the molecules to make fingerprints of
    :param radius: int - radius of the fingerprint
    :param n_bits: int - only for if bit_vec is true the length of the bit representation
    :param bit_vec: true/false - get the bit vector representation of the fingerprint
    :param feature_invarients: true/false - use feature invarients not structure for the fingerprint definition see http://www.rdkit.org/docs/GettingStartedInPython.html#feature-definitions-used-in-the-morgan-fingerprints
    :param return_only_non_zero: true/false - get only the non-zero elements of the fingerprint as it is sparse
    :param log_explanation: true/false - print an explanation of the bit see http://www.rdkit.org/docs/GettingStartedInPython.html#explaining-bits-from-morgan-fingerprints
    """
    
    log = logging.getLogger(__name__)
    
    mols = [mi.smiles_to_molcule(smile) for smile in smiles]

    all_fps = []
    if bit_vec is True:
        fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits, useFeatures=True) for mol in mols]

    else:
        fps = [AllChem.GetMorganFingerprint(mol, radius, useFeatures=True) for mol in mols]
        if return_only_non_zero is True:
            fps = [fp.GetNonzeroElements() for fp in fps]


    A = np.mat(fps)
    df = pd.DataFrame(data=A)

    return fps,df



def draw_molecule(smiles, smarts=None, filename="tmp.png"):
    """
    Function to draw the highligh over the molecule of what matches the substructure
    see https://www.rdkit.org/docs/GettingStartedInPython.html#drawing-molecules
    :param smiles: str - smiles molecualr representation
    :param smarts: str - smarts of the substructure to display
    :param filename: str - filename to save the overlaid image too
    """
    
    log = logging.getLogger(__name__)
    
    mol = mi.smiles_to_molcule(smiles)

    if smarts is not None:
        no_sub_structure = False
        substructure = Chem.MolFromSmarts(smarts)
        try:
            match_atoms = list(mol.GetSubstructMatch(substructure))
        except IndexError:
            no_sub_structure = True

        try:
            match_bonds = [mol.GetBondBetweenAtoms(match_atoms[b.GetBeginAtomIdx()],match_atoms[b.GetEndAtomIdx()]).GetIdx() for b in substructure.GetBonds()]    
        except IndexError:
            no_sub_structure = True

        if no_sub_structure is False:
            img = rdMolDraw2D.MolDraw2DCairo(400, 400)
            rdMolDraw2D.PrepareAndDrawMolecule(img, mol, highlightAtoms=match_atoms,
                                    highlightBonds=match_bonds)
        else:
            log.info("No sub-structure matching smarts {} found".format(smarts))
            img = rdMolDraw2D.MolDraw2DCairo(400, 400)
            rdMolDraw2D.PrepareAndDrawMolecule(img, mol)

    else:
        img = rdMolDraw2D.MolDraw2DCairo(400, 400)
        rdMolDraw2D.PrepareAndDrawMolecule(img, mol)
    
    img.drawOptions().addStereoAnnotation = True
    img.FinishDrawing()

    if filename is not None:
        img.WriteDrawingText(filename)

    img_string = img.GetDrawingText()
    
    return img, img_string

def draw_smarts_overlay(smiles, smarts, filename):
    """
    Function to draw the highligh over the molecule of what matches the substructure
    see https://www.rdkit.org/docs/GettingStartedInPython.html#drawing-molecules
    :param smiles: str - smiles molecualr representation
    :param smarts: str - smarts of the substructure to display
    :param filename: str - filename to save the overlaid image too
    """
    
    log = logging.getLogger(__name__)
    
    mol = mi.smiles_to_molcule(smiles)
    substructure = Chem.MolFromSmarts(smarts)
    match_atoms = list(mol.GetSubstructMatch(substructure))
    match_bonds = [mol.GetBondBetweenAtoms(match_atoms[b.GetBeginAtomIdx()],match_atoms[b.GetEndAtomIdx()]).GetIdx() for b in substructure.GetBonds()]
    img = rdMolDraw2D.MolDraw2DCairo(400, 400)
    rdMolDraw2D.PrepareAndDrawMolecule(img, mol, highlightAtoms=match_atoms,
                                    highlightBonds=match_bonds)
    
    img.drawOptions().addStereoAnnotation = True
    img.FinishDrawing()
    img.WriteDrawingText(filename)
    img_string = img.GetDrawingText()
    
    return img_string



def substructure_checker(smiles, substructure=None):
    """ 
    Function to find a substructure using SMARTS
    :param smi: str - smiles
    :param substructure: str - SMARTS defining the substructure to search for
    :return: tuple - smiles substructure and True/False for looking for the substructure
    """

    log = logging.getLogger(__name__)
        
    mol = mi.smiles_to_molcule(smiles)

    substruct = Chem.MolFromSmarts(substructure)

    has_substructure = 0

    if mol.HasSubstructMatch(substruct):
        has_substructure = 1 

    return has_substructure


def fingerprint_similarity(fps1, fps2, dice=False):
    """
    Function to calculate fingerprint similarity
    :param fps1: RDKit fingerprint - fingerprint of molecule 1
    :param fps2: RDKit fingerprint - fingerprint of molecule 2
    :param dice: true/false - Use dice similarity
    """
    
    log = logging.getLogger(__name__)
    
    if dice is True:
        similarity = DataStructs.DiceSimilarity(fps[0],fps[1])
    else:
        similarity = DataStructs.FingerprintSimilarity(fps[0],fps[1])
    return similarity

def bits_to_text(fp):
    """
    Function to convert bit vec to text 0s and 1s
    :param fp: RDKit bit fingerprint - RDKit bit fingerprint to be set to 1s and 0s
    """
    
    log = logging.getLogger(__name__)
    
    text = DataStructs.cDataStructs.BitVectToText(fp)
    
    return text

def bulk_similarity(fp, fp_targets, test=False, thresh=0.5):
    """
    Function to compare one fp with a list of others and get all the scores
    """
    
    log = logging.getLogger(__name__)
    
    tani_similarity = DataStructs.BulkTanimotoSimilarity(fp, fp_targets)
    data = np.array([[i for i in range(0, len(fp_targets))],[fp]*len(fp_targets), fp_targets, tani_similarity]).T
    df = pd.DataFrame(data=data, columns=["number", "fp_reference", "fp_target", "tanimoto_similarity"])
    
    if test is True:
        df = df[df["tanimoto_similarity"] >= thresh]
    return df
