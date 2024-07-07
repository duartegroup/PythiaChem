import unittest
import pandas as pd
import os
from pythia import molecules_and_structures as mi
import sys
from rdkit import Chem
from rdkit.DataStructs import ExplicitBitVect
# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Assuming the previous functions are in a module named "fingerprint_module"
from fingerprints_generation import (
    rdkit_fingerprints,
    maccskeys_fingerprints,
    atom_pair_fingerprints,
    torsion_fingerprints,
    morgan_fingerprints,
    draw_molecule,
    draw_smarts_overlay,
    substructure_checker,
    fingerprint_similarity,
    bits_to_text,
    bulk_similarity
)

class TestFingerprintFunctions(unittest.TestCase):

    def test_rdkit_fingerprints(self):
        smiles = ["CCO", "CCC"]
        fps = rdkit_fingerprints(smiles)
        self.assertEqual(len(fps), 2)
        self.assertIsInstance(fps[0], ExplicitBitVect)

    def test_maccskeys_fingerprints(self):
        smiles = ["CCO", "CCC"]
        fps, df = maccskeys_fingerprints(smiles)
        self.assertEqual(len(fps), 2)
        self.assertIsInstance(fps[0], ExplicitBitVect)
        self.assertIsInstance(df, pd.DataFrame)

    def test_atom_pair_fingerprints(self):
        smiles = ["CCO", "CCC"]
        fps = atom_pair_fingerprints(smiles)
        self.assertEqual(len(fps), 2)

    def test_torsion_fingerprints(self):
        smiles = ["CCO", "CCC"]
        fps = torsion_fingerprints(smiles)
        self.assertEqual(len(fps), 2)

    def test_morgan_fingerprints(self):
        smiles = ["CCO", "CCC"]
        fps, df = morgan_fingerprints(smiles)
        self.assertEqual(len(fps), 2)
        self.assertIsInstance(df, pd.DataFrame)

    def test_draw_molecule(self):
        smiles = "CCO"
        img, img_string = draw_molecule(smiles)
        self.assertIsNotNone(img)
        self.assertIsInstance(img_string, bytes)


    def test_draw_smarts_overlay(self):
        smiles = "CCO"
        smarts = "CC"
        img_string = draw_smarts_overlay(smiles, smarts, "tmp.png")
        self.assertIsInstance(img_string, bytes)

    def test_substructure_checker(self):
        smiles = "CCO"
        smarts = "CC"
        result = substructure_checker(smiles, smarts)
        self.assertEqual(result, 1)

    def test_fingerprint_similarity(self):
        smiles = ["CCO", "CCC"]
        fps = rdkit_fingerprints(smiles)
        similarity = fingerprint_similarity(fps[0], fps[1])
        self.assertIsInstance(similarity, float)

    def test_bits_to_text(self):
        smiles = ["CCO"]
        fps = rdkit_fingerprints(smiles)
        text = bits_to_text(fps[0])
        self.assertIsInstance(text, str)

    def test_bulk_similarity(self):
        smiles = ["CCO", "CCC"]
        fps = rdkit_fingerprints(smiles)
        df = bulk_similarity(fps[0], fps)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("tanimoto_similarity", df.columns)

if __name__ == '__main__':
    unittest.main()

