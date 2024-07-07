import unittest
import numpy as np
import os
import pandas as pd
import sys
from rdkit import Chem
from rdkit.Chem import AllChem

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from molecules_and_structures import (smiles_to_molcule,get_mol_from_smiles,check_stereo,
     molecule_image,get_mols,calc_rows,row_col_off_grid, mol_grid, twod_mol_align,substructure_match,
     overlap_venn,tanimoto_plot,tanimoto_similarity_comparison,get_fingerprints_bit_importance,plot_fingerprints_bit_importance)

class TestChemistryFunctions(unittest.TestCase):

    def test_smiles_to_molcule(self):
        smiles = "CCO"
        mol = smiles_to_molcule(smiles)
        self.assertIsInstance(mol, Chem.Mol)
        self.assertEqual(Chem.MolToSmiles(mol), "CCO")

    def test_get_mol_from_smiles(self):
        smiles = "CCO"
        mol = get_mol_from_smiles(smiles)
        self.assertIsInstance(mol, Chem.Mol)
        self.assertEqual(Chem.MolToSmiles(mol), "CCO")

    def test_check_stereo(self):
        smiles = "F[C@H](Cl)Br"
        mol = get_mol_from_smiles(smiles)
        mol, non_isosmiles = check_stereo(mol, clear_props=True)
        self.assertIsInstance(mol, Chem.Mol)
        self.assertEqual(non_isosmiles, "FC(Cl)Br")

    def test_molecule_image(self):
        smiles = "CCO"
        mol = get_mol_from_smiles(smiles)
        molecule_image(mol, smiles, fnam="test_image.png")
        self.assertTrue(os.path.isfile("test_image.png"))
        os.remove("test_image.png")

    def test_get_mols(self):
        smiles_list = ["CCO", "CCC"]
        mols = get_mols(smiles_list)
        self.assertEqual(len(mols), 2)
        self.assertIsInstance(mols[0], Chem.Mol)

    def test_calc_rows(self):
        mols = [get_mol_from_smiles(smiles) for smiles in ["CCO", "CCC", "CCCC"]]
        rows = calc_rows(mols, 2)
        self.assertEqual(rows, 2)

    def test_row_col_off_grid(self):
        row, col, offset = row_col_off_grid(5, 3, (100, 100))
        self.assertEqual(row, 1)
        self.assertEqual(col, 2)
        self.assertEqual(offset, [200, 100])

    def test_mol_grid(self):
        smiles_list = ["CCO", "CCC", "CCCC"]
        grid = mol_grid(smiles=smiles_list, mol_row=2)
        self.assertIsNotNone(grid)

    def test_twod_mol_align(self):
        smiles_list = ["CCO", "CCC"]
        mols = [get_mol_from_smiles(smiles) for smiles in smiles_list]
        aligned_mols = twod_mol_align(mols, template_smiles="CCO")
        self.assertEqual(len(aligned_mols), 2)

    def test_substructure_match(self):
        smiles_list = ["CCO", "CCC", "CCCC"]
        mols = [get_mol_from_smiles(smiles) for smiles in smiles_list]
        matches, indices = substructure_match("C", mols)
        self.assertEqual(len(matches), 3)

    def test_overlap_venn(self):
        smiles_dict = {
            "sub1": [get_mol_from_smiles(smiles) for smiles in ["CCO", "CCC"]],
            "sub2": [get_mol_from_smiles(smiles) for smiles in ["CCC", "CCCC"]]
        }
        overlap_venn(smiles_dict)
        # Manual inspection needed for the output plots

    def test_tanimoto_plot(self):
        smiles_list = ["CCO", "CCC", "CCCC"]
        df = tanimoto_plot(smiles_list, title="Test Plot", filename="test_tanimoto")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(os.path.isfile("fig_similarity_test_tanimoto.png"))
        os.remove("fig_similarity_test_tanimoto.png")

    def test_tanimoto_similarity_comparison(self):
        smiles_list1 = ["CCO", "CCC"]
        smiles_list2 = ["CCCC", "CC"]
        fps1 = [AllChem.GetMorganFingerprintAsBitVect(get_mol_from_smiles(smiles), 2) for smiles in smiles_list1]
        fps2 = [AllChem.GetMorganFingerprintAsBitVect(get_mol_from_smiles(smiles), 2) for smiles in smiles_list2]
        tanimoto_similarity_comparison(fps1, fps2, title="Comparison Plot", filename="test_comparison")
        self.assertTrue(os.path.isfile("fig_similarity_test_comparison.png"))
        os.remove("fig_similarity_test_comparison.png")



if __name__ == '__main__':
    unittest.main()

