#!/usr/bin/env python

"""
Module of functions for graph hopping
"""

import os
import json
from datetime import datetime
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdMolHash
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import Crippen
from rdkit.ML.Cluster import Butina
from rdkit import DataStructs
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdMolHash
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import DataStructs
from rdkit.Chem import rdCIPLabeler
from IPython.display import SVG, Image, display

import matplotlib.pyplot as plt

import logging


def smiles_to_molcule(
    s,
    addH=True,
    canonicalize=True,
    threed=True,
    add_stereo=False,
    remove_stereo=False,
    random_seed=10459,
    verbose=False,
):
    """
    :param s: str - smiles string
    :param addH: bool - Add Hydrogens or not
    """

    log = logging.getLogger(__name__)
    mol = get_mol_from_smiles(s, canonicalize=canonicalize)
    Chem.rdmolops.Cleanup(mol)
    Chem.rdmolops.SanitizeMol(mol)

    if remove_stereo is True:
        non_isosmiles = Chem.rdmolfiles.MolToSmiles(
            mol, isomericSmiles=False, allHsExplicit=False
        )
        mol = get_mol_from_smiles(non_isosmiles, canonicalize=canonicalize)
        Chem.rdmolops.Cleanup(mol)
        Chem.rdmolops.SanitizeMol(mol)

        if verbose is True:
            for atom in mol.GetAtoms():
                log.info(
                    "Atom {} {} in molecule from smiles {} tag will be cleared. "
                    "Set properties {}.".format(
                        atom.GetIdx(),
                        atom.GetSymbol(),
                        s,
                        atom.GetPropsAsDict(includePrivate=True, includeComputed=True),
                    )
                )

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
    log.debug(
        "Input smiles: {} RDKit Canonicalized smiles {} (Note RDKit does not use "
        "general canon smiles rules https://github.com/rdkit/rdkit/issues/2747)".format(
            smiles, s
        )
    )
    Chem.rdmolops.SanitizeMol(mol)
    Chem.rdmolops.Cleanup(mol)

    return mol


def check_stereo(mol, clear_props=False):
    """
    Function to check the status of the molecules CIP codes for isomers
    """
    update_smiles = False

    for atom in mol.GetAtoms():
        try:
            cip_code = atom.GetProp("_CIPCode")
            log.info(
                "CIP Code: {} Atom {} {} in molecule from smiles {}. "
                "Set properties {}.".format(
                    cip_code,
                    atom.GetIdx(),
                    atom.GetSymbol(),
                    s,
                    atom.GetPropsAsDict(includePrivate=True, includeComputed=True),
                )
            )

            if clear_props is True:
                atom.ClearProp("_CIPCode")
                log.info(
                    "CIP Code: {} Atom {} {} in molecule from smiles {} tag will be cleared. "
                    "Set properties {}.".format(
                        cip_code,
                        atom.GetIdx(),
                        atom.GetSymbol(),
                        s,
                        atom.GetPropsAsDict(includePrivate=True, includeComputed=True),
                    )
                )
                log.info(
                    "NEW: non-isomeric smiles: {}".format(
                        Chem.MolToSmiles(
                            Chem.rdmolops.RemoveHs(mol),
                            isomericSmiles=False,
                            allHsExplicit=False,
                        )
                    )
                )
                non_isosmiles = Chem.rdmolfiles.MolToSmiles(
                    mol, isomericSmiles=False, allHsExplicit=False
                )
                update_smiles = True
        except KeyError:
            pass

        if update_smiles is True:
            return mol, non_isosmiles
        else:
            return mol


def max_common_sub_graph(mols):
    """
    Function to find the maximum common sub-graph in a list of RDKit molecules
    :param: mols: list - list of RDKit molecule instances or smiles strings
    """

    log = logging.getLogger(__name__)

    if len(mols) <= 1:
        return None

    if isinstance(mols[0], str):
        mols = mols = [smiles_to_molcule(smi) for smi in mols]

    mcs = Chem.rdFMCS.FindMCS(mols)
    log.info(f"MCS cancelled? {mcs.canceled}")
    if not mcs.canceled:
        log.info(f"MCS SMARTS string = {mcs.smartsString}")
        log.info(f"MCS number of atoms = {mcs.numAtoms}")
        log.info(f"MCS number of bonds = {mcs.numBonds}")
        log.info(f"MCS query molecule = {mcs.queryMol}")
        mcs_obj_m = Chem.MolFromSmarts(mcs.smartsString)
        log.info(
            f"suggest visiting https://smarts.plus/smartsview to get the meaning of the SMARTS string"
        )
        log.info("\nQuery molecule:")
        display(mcs.queryMol)
        log.info("\nmcs:")
        display(mcs_obj_m)
    else:
        log.warning("MCS did not converge")

    return mcs


def compare_murcko_hashes(known_set, screening_set):
    """
    Compare two sets of murko hashes
    :param known_set: list - list of Murcko hashes from a known set
    :param screening_set: list - list of Murcko hashes you want to screen
    """

    log = logging.getLogger(__name__)

    matches = []
    for ith, hsh in enumerate(known_set):
        log.info("----- {} '{}'' -----".format(ith, hsh))
        tmp = []
        for inx, th in enumerate(screening_set):
            if hsh in th:
                log.info("{} {}".format(hsh, th))
                log.info("Scafold {} matches screening hashes {}\n".format(ith, inx))
                tmp.append(inx)
        matches.append(tmp)
        # matches.append([inx for inx, th in enumerate(test_murcko_hashes) if hsh in th])

    return matches


def compare_hashes(
    known_set,
    screening_set,
    exact=True,
    within=False,
    verbose=False,
    substrutcure_search=False,
):
    """
    Compare two sets of hashes
    :param known_set: list - list of Murcko hashes from a known set
    :param screening_set: list - list of Murcko hashes you want to screen
    """

    log = logging.getLogger(__name__)

    if verbose:
        if within is True:
            matches = []
            for ith, hsh in enumerate(known_set):
                log.info("----- {} '{}'' -----".format(ith, hsh))
                tmp = []
                log.info("Checking for matches to known structure {}".format(ith))
                for inx, th in enumerate(screening_set):
                    if hsh in th:
                        # log.info("{} {}".format(hsh,th))
                        # log.info("Scafold {} matches screening hashes {}\n".format(ith, inx))
                        tmp.append(inx)
                matches.append(tmp)
                # matches.append([inx for inx, th in enumerate(test_murcko_hashes) if hsh in th])

        if exact is True:
            matches = []
            for ith, hsh in enumerate(known_set):
                log.info("----- {} '{}'' -----".format(ith, hsh))
                tmp = []
                log.info("Checking for matches to known structure {}".format(ith))
                for inx, th in enumerate(screening_set):
                    if hsh == th:
                        # log.info("{} {}".format(hsh,th))
                        # log.info("Scafold {} matches screening hashes {}\n".format(ith, inx))
                        tmp.append(inx)
                matches.append(tmp)

        if substrutcure_search:
            matches = []
            for ith, hsh in enumerate(known_set):
                tmp = []
                log.info("Checking for matches to known structure {}".format(ith))
                for inx, th in enumerate(screening_set):
                    try:
                        m = Chem.MolFromSmarts(hsh)
                    except Exception as err:
                        log.info("Invalid SMARTS: {}".format(hsh))
                    if m is not None:
                        if th.HasSubstructMatch(m):
                            tmp.append(inx)
                    else:
                        log.error("Cannot parse smarts {}".format(hsh))
                        break

                matches.append(tmp)

    else:

        if within is True:
            matches = []
            for ith, hsh in enumerate(known_set):
                tmp = [inx for inx, th in enumerate(screening_set) if hsh in th]
                matches.append(tmp)

        if exact is True:
            matches = []
            for ith, hsh in enumerate(known_set):
                tmp = [inx for inx, th in enumerate(screening_set) if hsh == th]
                matches.append(tmp)

        if substrutcure_search:
            matches = []
            for ith, hsh in enumerate(known_set):
                tmp = []
                for inx, th in enumerate(screening_set):
                    m = Chem.MolFromSmarts(hsh)
                    if m is not None:
                        if th.HasSubstructMatch(m):
                            tmp.append(inx)
                    else:
                        log.error("Cannot parse smarts {}".format(hsh))
                        break

                matches.append(tmp)

    return matches


def restricted_anonymous_hash(smiles, restricted_set="[C]", verbose=False):
    """
    Function for custom hashing replaces an anonymous graph with a graph which contains only a certain sub-set of elements
    :param smiles: list - list of smiles
    :param restricted_set: str - exact string to use to match certain atoms/elements
    :return list
    """
    log = logging.getLogger(__name__)

    log.info("Note: restricted anonymous graphs ignore stereochemical information")
    anon_moles = [
        smiles_to_molcule(smi, threed=False, addH=False, remove_stereo=True)
        for smi in smiles
    ]
    anon_hashes = [
        rdMolHash.MolHash(
            Chem.rdmolops.RemoveHs(m), rdMolHash.HashFunction.AnonymousGraph
        )
        for m in anon_moles
    ]
    rag_hashes = [anon_hash.replace("*", restricted_set) for anon_hash in anon_hashes]

    if verbose is True:
        log.info(
            "\n".join(
                [
                    "{} {} {}".format(smiles[ith], ent, elt)
                    for ith, (ent, elt) in enumerate(zip(anon_hashes, rag_hashes))
                ]
            )
        )

    return rag_hashes


def show_specific(
    index,
    ld_hashes,
    ld_smiles,
    screen_hashes,
    screen_smiles,
    matches,
    dataset=None,
    dataset_label=None,
    dataset_structure=None,
    smarts=False,
    ld_indexes_giving_hashes=None,
):
    """
    Show a specific output of hash matching
    :param index: int - index to look at from the non-screening list
    :param ld_hashes: list - literature data list of hashes
    :param screen_hashes: list - screening molecules list of hashes
    :param screen_smiles: list - screening molecules list of smiles
    :param matches: list of lists - matches index numbers outer list in order of the non-screening list and sub lists are index of hashes matching screening set
    :param dataset: pandas dataframe - raw screening dataset
    :param dataset_label: str - column header for labels
    :param dataset_structure: str - column header for smiles
    """

    log = logging.getLogger(__name__)

    if ld_indexes_giving_hashes is not None:
        smi_index = ld_indexes_giving_hashes[index]
        if smarts is False:
            log.info(ld_smiles[smi_index])
            ld_m = smiles_to_molcule(ld_smiles[smi_index], threed=False)
            mtmp = smiles_to_molcule(ld_hashes[index], threed=False)
            ld_m.GetSubstructMatches(mtmp)
            log.info(
                "----- Known scafold index {} -----\nGenerated hash ({}) from molecule:".format(
                    index, ld_hashes[index]
                )
            )
            log.info("\n{}\n{}\n".format(display(ld_m), display(mtmp)))
            log.info("Matching indexes:\n{}\nImages".format(matches[index]))

            if dataset is not None:
                for j in matches[index]:
                    log.info(
                        "-- {}: {}  {} --".format(
                            j,
                            dataset[dataset_label].values[j],
                            dataset[dataset_structure].values[j],
                        )
                    )
                    mmtmp = smiles_to_molcule(
                        dataset[dataset_structure].values[j], threed=False
                    )
                    mmtmp.GetSubstructMatches(mtmp)
                    log.info("{}\n--\n".format(display(mmtmp)))
            else:
                for j in matches[index]:
                    log.info("-- {}: {} --".format(j, screen_smiles[j]))
                    mmtmp = smiles_to_molcule(screen_smiles[j], threed=False)
                    mmtmp.GetSubstructMatches(mtmp)
                    log.info("{}\n--\n".format(display(mmtmp)))
        else:
            ld_m = smiles_to_molcule(ld_smiles[smi_index], threed=False)
            mtmp = Chem.MolFromSmarts(ld_hashes[index])
            ld_m.GetSubstructMatches(mtmp)
            log.info(
                "----- Known scafold index {} -----\nGenerated hash ({}) from molecule:".format(
                    index, ld_hashes[index]
                )
            )
            log.info("\n{}\n{}\n".format(display(ld_m), display(mtmp)))
            log.info("Matching indexes:\n{}\nImages".format(matches[index]))

            if dataset is not None:
                for j in matches[index]:
                    log.info(
                        "-- {}: {}  {} --".format(
                            j,
                            dataset[dataset_label].values[j],
                            dataset[dataset_structure].values[j],
                        )
                    )
                    mmtmp = smiles_to_molcule(
                        dataset[dataset_structure].values[j], threed=False
                    )
                    mmtmp.GetSubstructMatches(mtmp)
                    log.info("{}\n--\n".format(display(mmtmp)))
            else:
                for j in matches[index]:
                    log.info("-- {}: {} --".format(j, screen_smiles[j]))
                    mmtmp = Chem.MolFromSmarts(oakwood_dataset["smiles"].values[j])
                    mmtmp.GetSubstructMatches(mtmp)
                    log.info("{}\n--\n".format(display(mmtmp)))

    else:

        if smarts is False:
            ld_m = smiles_to_molcule(ld_smiles[index], threed=False)
            mtmp = smiles_to_molcule(ld_hashes[index], threed=False)
            ld_m.GetSubstructMatches(mtmp)
            log.info(
                "----- Known scafold index {} -----\nGenerated hash ({}) from molecule:".format(
                    index, ld_hashes[index]
                )
            )
            log.info("\n{}\n{}\n".format(display(ld_m), display(mtmp)))
            log.info("Matching indexes:\n{}\nImages".format(matches[index]))

            if dataset is not None:
                for j in matches[index]:
                    log.info(
                        "-- {}: {}  {} --".format(
                            j,
                            dataset[dataset_label].values[j],
                            dataset[dataset_structure].values[j],
                        )
                    )
                    mmtmp = smiles_to_molcule(
                        dataset[dataset_structure].values[j], threed=False
                    )
                    mmtmp.GetSubstructMatches(mtmp)
                    log.info("{}\n--\n".format(display(mmtmp)))
            else:
                for j in matches[index]:
                    log.info("-- {}: {} --".format(j, screen_smiles[j]))
                    mmtmp = smiles_to_molcule(screen_smiles[j], threed=False)
                    mmtmp.GetSubstructMatches(mtmp)
                    log.info("{}\n--\n".format(display(mmtmp)))
        else:
            log.info("{} {}".format(ld_smiles[index], ld_hashes[index]))
            ld_m = smiles_to_molcule(ld_smiles[index], threed=False)
            mtmp = Chem.MolFromSmarts(ld_hashes[index])
            ld_m.GetSubstructMatches(mtmp)
            log.info(
                "----- Known scafold index {} -----\nGenerated hash ({}) from molecule:".format(
                    index, ld_hashes[index]
                )
            )
            log.info("\n{}\n{}\n".format(display(ld_m), display(mtmp)))
            log.info("Matching indexes:\n{}\nImages".format(matches[index]))

            if dataset is not None:
                for j in matches[index]:
                    log.info(
                        "-- {}: {}  {} --".format(
                            j,
                            dataset[dataset_label].values[j],
                            dataset[dataset_structure].values[j],
                        )
                    )
                    mmtmp = smiles_to_molcule(
                        dataset[dataset_structure].values[j], threed=False
                    )
                    mmtmp.GetSubstructMatches(mtmp)
                    log.info("{}\n--\n".format(display(mmtmp)))
            else:
                for j in matches[index]:
                    log.info("-- {}: {} --".format(j, screen_smiles[j]))
                    mmtmp = Chem.MolFromSmarts(oakwood_dataset["smiles"].values[j])
                    mmtmp.GetSubstructMatches(mtmp)
                    log.info("{}\n--\n".format(display(mmtmp)))


def analyze_regio_hashes(hashes, outname=None, draw=False, figsize=(10, 10)):
    """
    Analyze a set of regio hashes to look fro most common fragments
    :param hashes: list - rdkit regio hash list
    :param outname: str - name to save plot to
    """

    log = logging.getLogger(__name__)

    raw_groups = [ent.split(".") for ent in hashes]

    all_groups = []
    for ent in raw_groups:
        all_groups = all_groups + ent

    regio_hashes = set(all_groups)
    log.debug("Raw regio hash\n{}".format(regio_hashes))

    df = pd.DataFrame(columns=regio_hashes)
    log.debug("Start data frame:\n{}".format(df))

    # loop over list of lists with sublist contents being the regio isomer fragments of a moleucle
    # i.e. each iteration is searching one moleucles hash for the presence of each unique hash
    for indx, search_grp in enumerate(raw_groups):
        tmp = [1 if ent in search_grp else 0 for ent in regio_hashes]
        log.debug("Index {} length {}\n{}".format(indx, len(tmp), tmp))
        t = pd.DataFrame([tmp], columns=regio_hashes)
        df = pd.concat([df, t], axis=0, copy=True)
        log.debug("Current dataframe:\n{}".format(df))

    log.debug("Complete dataframe:\n{}".format(df))

    aggregate_group_counts = df.sum(axis=0, skipna=True)
    log.info(
        "Regioisomer hash counts over the dataset: {}\n".format(aggregate_group_counts)
    )

    if draw is True:
        for ind, s in enumerate(regio_hashes):
            log.info(
                "\nRegio smarts: {} Number of occurances in the dataset: {}".format(
                    s, aggregate_group_counts.values[ind]
                )
            )
            log.info("{}".format(display(Chem.MolFromSmarts(s))))

    ax = aggregate_group_counts.plot.bar(figsize=figsize, sort_columns=True)

    if outname is not None:
        fig = ax.get_figure()
        fig.savefig(outname)

    return aggregate_group_counts, df, raw_groups


def search_for_regio_hashes(
    search_for_hashes, searchable_hashes, outname=None, draw=False, labels=None
):
    """
    Analyze a set of regio hashes to look fro most common fragments
    :param hashes: list - rdkit regio hash list
    :param outname: str - name to save plot to
    """

    log = logging.getLogger(__name__)

    # hashes of molecules to use as search patterns
    raw_groups = [ent.split(".") for ent in search_for_hashes]
    all_groups = []
    for ent in raw_groups:
        all_groups = all_groups + ent
    regio_hashes = set(all_groups)
    log.debug("Raw regio hash to use to search\n{}".format(regio_hashes))

    # hashes of molecules to search through
    searchable_regios = [s.split(".") for s in searchable_hashes]

    df = pd.DataFrame(columns=regio_hashes)
    log.debug("Starting data frame:\n {}".format(df))

    # Loop over the searchable list i.e. the list of unknown molecules and look for matches to a list
    # of known molecles
    for indx, search_grp in enumerate(searchable_regios):
        tmp = [1 if ent in search_grp else 0 for ent in regio_hashes]
        log.debug("Index {} length {}\n{}".format(indx, len(tmp), tmp))
        t = pd.DataFrame([tmp], columns=regio_hashes)
        df = pd.concat([df, t], axis=0, copy=True)
        # log.info("Current dataframe:\n{}".format(df))

    log.debug("Complete dataframe:\n{}".format(df))

    aggregate_group_counts = df.sum(axis=0, skipna=True)
    log.info(
        "Regioisomer hash counts over the dataset: {}\n".format(aggregate_group_counts)
    )

    if draw is True:
        for ind, s in enumerate(regio_hashes):
            log.info(
                "\nRegio smarts: {} Number of occurances in the dataset: {}".format(
                    s, aggregate_group_counts.values[ind]
                )
            )
            log.info("{}".format(display(Chem.MolFromSmarts(s))))

    ax = aggregate_group_counts.plot.bar(figsize=(10, 10), sort_columns=True)

    if labels is not None:
        df.insert(loc=0, column="labels", value=labels)
        df.to_csv("{}.csv".format(outname))

    if outname is not None:
        fig = ax.get_figure()
        fig.savefig(outname)


def show_all(
    literature_hashes,
    matches,
    dataset,
    literature_mols=None,
    dataset_label_key="zinc_id",
    dataset_structure_key="smiles",
):
    """
    Function to show all matches
    :param literature_hashes: list - smart rdkit hashes
    :param literature_mols: list - rdkit mols
    :param matches: list - match indexes
    :param dataset: pandas dataframe - Dataframe dataset
    :param dataset_label_key: str - column key setting labels for the dataset
    :param dataset_structure_key: str - column key setting structure representations for the dataset
    """

    log = logging.getLogger(__name__)

    for i, known in enumerate(literature_hashes):
        mtmp = smiles_to_molcule(known, threed=False)
        log.info(
            "----- Known scafold index {} -----\nGenerated hash ({}) from molecule:".format(
                i, known
            )
        )
        if literature_mols is not None:
            log.info("\n{}\n{}\n".format(display(literature_mols[i]), display(mtmp)))
        else:
            log.info("Matching indexes: {}\n".format(display(mtmp)))
        log.info("Matching indexes:\n{}\nImages".format(matches[i]))

        for j in matches[i]:
            log.info(
                "-- {}: {}  {} --".format(
                    j,
                    dataset[dataset_label_key].values[j],
                    dataset[dataset_structure_key].values[j],
                )
            )
            mtmp = smiles_to_molcule(
                dataset[dataset_structure_key].values[j], threed=False
            )
            log.info("{}\n--\n".format(display(mtmp)))
