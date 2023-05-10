# Python packages and utilities
from datetime import datetime
import pandas as pd
import numpy as np

#RDKit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
from rdkit.Chem import Descriptors, rdMolDescriptors

# Image libraries
try:
    import Image
except ImportError:
    from PIL import Image
from io import BytesIO

# Logging
import logging


def smiles_to_molcule(s, addH=True, canonicalize=True, threed=True, add_stereo=False, remove_stereo=False, random_seed=10459, verbose=False):
    """ 
    :param s: str - smiles string
    :param addH: bool - Add Hydrogens or not
    """

    log = logging.getLogger(__name__)

    mol = get_mol_from_smiles(s, canonicalize=canonicalize)
    Chem.rdmolops.Cleanup(mol)
    Chem.rdmolops.SanitizeMol(mol)
    
    if remove_stereo is True:
        non_isosmiles = Chem.rdmolfiles.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=False)
        mol = get_mol_from_smiles(non_isosmiles, canonicalize=canonicalize)
        Chem.rdmolops.Cleanup(mol)
        Chem.rdmolops.SanitizeMol(mol)
        
        if verbose is True:
            for atom in mol.GetAtoms():
                log.info("Atom {} {} in molecule from smiles {} tag will be cleared. "
                        "Set properties {}.".format(atom.GetIdx(), atom.GetSymbol(), s, atom.GetPropsAsDict(includePrivate=True, includeComputed=True)))

    if addH is True:
        mol = Chem.rdmolops.AddHs(mol)

    if add_stereo is True:
        rdCIPLabeler.AssignCIPLabels(mol)


    if threed:
        AllChem.EmbedMolecule(mol, randomSeed=random_seed)

    return mol 
    
def get_mol_from_smiles(smiles, canonicalize=True):
    """ 
    Function to make a mol object based on smiles
    :param smiles: str - SMILES string
    :param canonicalize: True/False - use RDKit canonicalized smile or the input resprectively
    """

    log = logging.getLogger(__name__)

    if canonicalize is True:
        s = Chem.CanonSmiles(smiles, useChiral=1)
    else:
        s = smiles
    mol = Chem.MolFromSmiles(s)
    log.debug("Input smiles: {} RDKit Canonicalized smiles {} (Note RDKit does not use "
              "general canon smiles rules https://github.com/rdkit/rdkit/issues/2747)".format(smiles, s)) 
    Chem.rdmolops.SanitizeMol(mol)
    Chem.rdmolops.Cleanup(mol)

    return mol 

def check_stereo(mol, clear_props=False):
    """
    Function to check the status of the molecules CIP codes for isomers
    """
    
    log = logging.getLogger(__name__)

    update_smiles = False

    for atom in mol.GetAtoms():
        try:
            cip_code = atom.GetProp('_CIPCode')
            log.info("CIP Code: {} Atom {} {} in molecule from smiles {}. "
                      "Set properties {}.".format(cip_code, atom.GetIdx(), atom.GetSymbol(), s, atom.GetPropsAsDict(includePrivate=True, includeComputed=True)))
            
            if clear_props is True:
                atom.ClearProp("_CIPCode")
                log.info("CIP Code: {} Atom {} {} in molecule from smiles {} tag will be cleared. "
                     "Set properties {}.".format(cip_code, atom.GetIdx(), atom.GetSymbol(), s, atom.GetPropsAsDict(includePrivate=True, includeComputed=True)))
                log.info("NEW: non-isomeric smiles: {}".format(Chem.MolToSmiles(Chem.rdmolops.RemoveHs(mol), isomericSmiles=False, allHsExplicit=False)))
                non_isosmiles = Chem.rdmolfiles.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=False)
                update_smiles = True
        except KeyError:
            pass

        if update_smiles is True:
            return mol, non_isosmiles
        else:
            return mol

def molecule_image(mol, smile, fnam=None, label_with_num=True):
    """ 
    Save an image 2D of the molecule
    :param mol: object molecule 
    :param smile: smiles string
    :param fnam: file name
    :return:
    """

    log = logging.getLogger(__name__)

    if label_with_num is True:
        for atom in mol.GetAtoms():
            atom.SetProp('atomLabel',"{}{}".format(atom.GetSymbol(),str(atom.GetIdx())))
    
    if fnam is None:
        mf = rdMolDescriptors.CalcMolFormula(mol)
        if os.path.isfile("molecule_{}.png".format(mf)):
            fnam = "molecule_{}.png".format(smile)
        else:
            fnam = "molecule_{}.png".format(mf)

    DrawingOptions.atomLabelFontSize = 55
    DrawingOptions.dotsPerAngstrom = 300 
    DrawingOptions.bondLineWidth = 3.5 

    Draw.MolToImageFile(mol, fnam, size=(800,800))

def get_mols(smiles, individual_image=True, label_with_num=True):
    """
    function to generate a list of molecule objects
    :param smiles: list of smiles
    """

    log = logging.getLogger(__name__)

    list_mols = []
    for n, smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile)
        
        if label_with_num is True:
            for atom in mol.GetAtoms():
                atom.SetProp('atomLabel',"{}{}".format(atom.GetSymbol(),str(atom.GetIdx())))
                
        if individual_image is True:
            molecule_image(mol, smile, fnam=str(n) + ".png")
        list_mols.append(mol)
    
    return list_mols
    
def calc_rows(mols, mol_rows):
    """
    Calculate the number of rows needed to neatly represent molecule images on a grid
    :param mols: list of molecules
    :param mol_rows: number of molecules on a row
    """

    log = logging.getLogger(__name__)

    n_mols = len(mols)
    m_rows = n_mols / mol_rows
    n_rows = int(np.floor(m_rows))
    if n_mols % mol_rows:
        n_rows = n_rows + 1
        
    log.info("Using {} rows".format(n_rows))
    
    return n_rows

def row_col_off_grid(n, mol_row, subimage_size):
    """
    calculate the row index column index and the offset in the grid from [0,0]
    :param n: molecule number in list of molecules 7th, 9th etc
    :param mol_row: number of molecules on a row
    :param subimage_size: sub image pixel size
    """

    log = logging.getLogger(__name__)

    row_index = int(np.floor(n / mol_row))
    column_index = n % mol_row
    offx = column_index * subimage_size[0]
    offy = row_index * subimage_size[1]
    log.info("Molecule {}: off set x {} off set y {}".format(n, offx, offy))
    grid_offset = [column_index * subimage_size[0], row_index * subimage_size[1]]
    
    return row_index, column_index, grid_offset

def mol_grid(smiles=None, mols=None, mol_row=3, subimage_size=(200, 200), labels=None, filename=None, max_mols=None):
    """
    :param mols: list of molecules
    :param mol_rows: number of molecules on a row
    """
    
    log = logging.getLogger(__name__)

    if mols is None:
        if smiles is None:
            log.error("Error - Need either list of smiles or list of molecule objects from RDKit")
        else:
            mols = get_mols(smiles)
    

    if max_mols is None:
        max_mols = len(mols)

    if filename is None:
        if labels is None:
            grid = Chem.Draw.MolsToGridImage(mols, molsPerRow=mol_row, subImgSize=(400,400), maxMols=max_mols)
        else:
            grid = Chem.Draw.MolsToGridImage(mols, molsPerRow=mol_row, subImgSize=(400,400), legends=labels, maxMols=max_mols)
    else:
        if labels is None:
            grid = Chem.Draw.MolsToGridImage(mols, molsPerRow=mol_row, returnPNG=True,  subImgSize=(400,400), maxMols=max_mols)
        else:
            grid = Chem.Draw.MolsToGridImage(mols, molsPerRow=mol_row, returnPNG=True, subImgSize=(400,400), legends=labels, maxMols=max_mols)

        log.info("Image saved as {}".format(filename))
        with open("{}".format(filename, "wb")) as img_png:
            img_png.write(grid.data)

    return grid

def twod_mol_align(molecules, template_smarts=None, template_smiles=None):
    """
    Function to align 2D RDkit molecules.
    :param molecules: list - list of RDKit molecules
    :param template_smarts: str - smarts string to use as a template
    :param template_smiles: str - smiles string to use as a template
    """
    if template_smarts is not None:
        temp_mol = Chem.MolFromSmarts(template_smarts)
    elif template_smiles is not None:
        temp_mol = Chem.MolFromSmiles(template_smarts)

    AllChem.Compute2DCoords(temp_mol)
    temp_atom_inxes = temp_mol.GetSubstructMatch(temp_mol)


    for m in molecules:
        #Chem.TemplateAlign.AlignMolToTemplate2D(m, macss_mol, clearConfs=True)
        AllChem.Compute2DCoords(m)
        mol_atom_inxes = m.GetSubstructMatch(temp_mol)
        rms = AllChem.AlignMol(m, temp_mol, atomMap=list(zip(mol_atom_inxes, temp_atom_inxes)))

    return molecules
