# parts of this code are inspired by Chemical Space Analysis and Property Prediction for Carbon Capture Amine Molecules.
#https://doi.org/10.1039/D3DD00073G
#https://zenodo.org/records/10213104
#https://github.com/flaviucipcigan/ccus_amine_prediction_workflow

# Python packages and utilities
from io import BytesIO
import joblib
import logging
# Image libraries
from PIL import Image
# Matplotlib
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
import numpy as np
import os
import pandas as pd
# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import rdCoordGen
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import DataStructs



def smiles_to_molcule(s, addH=False, canonicalize=True, threed=True, add_stereo=False, remove_stereo=False, random_seed=10459, verbose=False):
    """Converts a SMILES string to an RDKit molecule object with optional modifications.

    Parameters:
    ----------
    s : str
        SMILES string representing the molecule.
    addH : bool, optional
        Whether to add hydrogens to the molecule (default is False).
    canonicalize : bool, optional
        Whether to canonicalize the molecule (default is True).
    threed : bool, optional
        Whether to generate a 3D conformation of the molecule (default is True if addH is True, False otherwise).
    add_stereo : bool, optional
        Whether to add stereochemistry information to the molecule (default is False).
    remove_stereo : bool, optional
        Whether to remove stereochemistry information from the molecule (default is False).
    random_seed : int, optional
        Random seed for 3D conformation generation (default is 10459).
    verbose : bool, optional
        Whether to log detailed information during molecule processing (default is False).

    Returns:
    -------
    mol : RDKit molecule object
        RDKit molecule object representing the molecule converted from SMILES.
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

    if add_stereo is True:
        rdCIPLabeler.AssignCIPLabels(mol)

    if addH is True:
        mol = Chem.rdmolops.AddHs(mol)
        if threed is True:
            AllChem.EmbedMolecule(mol, randomSeed=random_seed)
    if addH is False:
        threed = False

    return mol 
    
def get_mol_from_smiles(smiles, canonicalize=True):
    """
    Converts a SMILES string into an RDKit molecule object.

    Parameters:
    ----------
    smiles : str
        SMILES string representing the molecule.
    canonicalize : bool, optional
        Whether to use RDKit canonicalized SMILES (default is True).

    Returns:
    -------
    mol : RDKit molecule object
        RDKit molecule object corresponding to the input SMILES.

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

    Parameters:
    ----------
    mol : RDKit molecule object
        RDKit molecule object corresponding to the input SMILES.
    """
    
    log = logging.getLogger(__name__)
    non_isosmiles = None  # Initialize non_isosmiles

    for atom in mol.GetAtoms():
        try:
            cip_code = atom.GetProp('_CIPCode')
            log.info("CIP Code: {} Atom {} {} in molecule from smiles {}. "
                      "Set properties {}.".format(cip_code, atom.GetIdx(), atom.GetSymbol(), s, atom.GetPropsAsDict(includePrivate=True, includeComputed=True)))
            
            if clear_props:
                atom.ClearProp("_CIPCode")
                log.info("CIP Code: {} Atom {} {} in molecule from smiles {} tag will be cleared. "
                     "Set properties {}.".format(cip_code, atom.GetIdx(), atom.GetSymbol(), s, atom.GetPropsAsDict(includePrivate=True, includeComputed=True)))

        except KeyError:
            pass

        non_isosmiles = Chem.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=False)
        return mol, non_isosmiles

def molecule_image(mol, smile, fnam=None, label_with_num=True):
    """ 
    Save an image 2D of the molecule
    :param mol: object molecule 
    :param smile: smiles string
    :param fnam: file name
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

def mol_grid(smiles=None, mols=None, mol_row=3, subimage_size=(100, 100), labels=None, filename=None, max_mols=None):
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
            grid = Chem.Draw.MolsToGridImage(mols, molsPerRow=mol_row, subImgSize=subimage_size, maxMols=max_mols)
        else:
            grid = Chem.Draw.MolsToGridImage(mols, molsPerRow=mol_row, subImgSize=subimage_size, legends=labels, maxMols=max_mols)
    else:
        if labels is None:
            grid = Chem.Draw.MolsToGridImage(mols, molsPerRow=mol_row, returnPNG=True,  subImgSize=subimage_size, maxMols=max_mols)
        else:
            grid = Chem.Draw.MolsToGridImage(mols, molsPerRow=mol_row, returnPNG=True, subImgSize=subimage_size, legends=labels, maxMols=max_mols)

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
        temp_mol = Chem.MolFromSmiles(template_smiles)

    if temp_mol is None:
        raise ValueError("Template molecule could not be created from the provided SMARTS or SMILES.")

    AllChem.Compute2DCoords(temp_mol)
    temp_atom_inxes = temp_mol.GetSubstructMatch(temp_mol)


    for m in molecules:
        #Chem.TemplateAlign.AlignMolToTemplate2D(m, macss_mol, clearConfs=True)
        AllChem.Compute2DCoords(m)
        mol_atom_inxes = m.GetSubstructMatch(temp_mol)
        if not mol_atom_inxes:  # If the substructure match is not found, skip alignment
            continue
        rms = AllChem.AlignMol(m, temp_mol, atomMap=list(zip(mol_atom_inxes, temp_atom_inxes)))

    return molecules


def substructure_match(substructure_smarts, mols):
    """
        Find molecules in a list that match a specified substructure SMARTS pattern.

        Parameters:
        ----------
        substructure_smarts : str
            SMARTS pattern representing the substructure to search for.
        mols : list of RDKit molecule objects
            List of RDKit molecule objects to search within.

        Returns:
        -------
        matches : list
            List of RDKit molecule objects that contain the substructure defined by `substructure_smarts`.
        indices : list
            List of indices corresponding to the positions of `matches` within the input `mols` list.

    """

    log = logging.getLogger(__name__)

    matches = []
    indices = []

    patt = Chem.MolFromSmarts(substructure_smarts)
    for i, mol in enumerate(mols):
        if mol.HasSubstructMatch(patt):
            indices.append(i)
            matches.append(mol)

    log.info(f"Found {len(matches)} matches")

    return matches, indices

def overlap_venn(dict):
    """
        Generate Venn diagrams to visualize overlaps between sets of molecules based on different substructure patterns.

        Parameters:
        ----------
        dict : dict
            Dictionary where keys are substructure names and values are lists of RDKit molecule objects.

        Returns:
        -------
        None

        """

    n_substructure = len(dict.keys())
    substructure_name = list(dict.keys())
    
    # Overlap between 2 sets of molecules with different substructure patterns
    for i in range(n_substructure-1):
        for j in range(n_substructure):
            if j > i:
                intersect_i = set(dict[substructure_name[i]]).intersection(set(dict[substructure_name[j]]))
                if len(intersect_i) != 0:
                    im_overlap = Chem.Draw.MolsToGridImage(intersect_i,molsPerRow=5,
                                                        subImgSize=(800,800), returnPNG=True)
                    
                    fig, ax = plt.subplots()
                    plt.tight_layout()
                    
                    v = venn2([set(dict[substructure_name[i]]), set(dict[substructure_name[j]])], 
                        (substructure_name[i], substructure_name[j]), 
                        set_colors=('lightblue', 'grey'),
                            ax = ax)
                    
                    fig_name = 'fig_venn2_' + substructure_name[i] + '_' + substructure_name[j] + '.png'
                    plt.savefig(fig_name)
                    
                    plt.show()


    # Overlap between 3 sets of molecules with different substructure patterns
    for i in range(n_substructure-2):
        for j in range(n_substructure-1):
            for k in range(n_substructure):
                if k > j > i:
                    intersect_l = set(dict[substructure_name[i]]).intersection(set(dict[substructure_name[j]])).intersection(set(dict[substructure_name[k]]))
                    if len(intersect_l) != 0:
                        fig, ax = plt.subplots()
                        plt.tight_layout()
                        v = venn3([set(dict[substructure_name[i]]), set(dict[substructure_name[j]]), set(dict[substructure_name[k]])],
                            (substructure_name[i], substructure_name[j], substructure_name[k]), 
                            set_colors=('lightblue', 'grey', 'lightpink'),
                                ax = ax)
                        
                        fig_name = 'fig_venn3_' + substructure_name[i] + '_' + substructure_name[j] + '_' + substructure_name[k] + '.png'
                        plt.savefig(fig_name)
                        
                        plt.show()              
                    
    return

def tanimoto_plot(smiles, title=None, figsize=(6,6), dpi=300, filename=None, cmap='Blues'):
    """Plot tanimoto similarity matrix of fingerprints.
    
    Parameters
    ----------
    smiles : list
        List of SIMLES.
    title : str, optional
        Title of the figure. Default is None.
    figsize : tuple, optional
        Size of the figure. Default is (6,6).
    dpi : int, optional
        Resolution of the figure. Default is 300.
    filename : str, optional
        Name of the file. The output file will be named fig_similarity_{filename}.png. Default is None.  
    cmap : str, optional
        Colormap. Default is 'Blues'.
    
    Returns
    -------
    similarity matrix dataframe : pandas dataframe
        Dataframe containing the similarity matrix.
    """
	

    # convert smiles to mol
    mols =[Chem.MolFromSmiles(smi) for smi in smiles]

    # generate fingerprints
    fps = [AllChem.GetMorganFingerprint(m,2) for m in mols]

    # the list for the dataframe
    qu, ta, sim = [], [], []
    qu_n, ta_n = [], []

    # compare all fp pairwise without duplicates
    for n in range(len(fps)-1): # -1 so the last fp will not be used
        s = DataStructs.BulkTanimotoSimilarity(fps[n], fps[n+1:]) # +1 compare with the next to the last fp
        #print(smiles_list[n], smiles_list[n+1:]) # witch mol is compared with what group

        # collect the SMILES and values
        for m in range(len(s)):
            qu_n.append(n)
            ta_n.append(m)
            qu.append(smiles[n])
            ta.append(smiles[n+1:][m])
            sim.append(s[m])
    print()

    # build the dataframe and sort it
    d = {'query':qu, 'target':ta, 'Similarity':sim}
    df_final = pd.DataFrame(data=d)
    dlabel = {'query':qu_n, 'target':ta_n, 'Similarity':sim}
    df_label_final = pd.DataFrame(data=dlabel)

    # save as csv
    df_final.to_csv('fingerprints.csv', index=False, sep=',')
    df_label_final.to_csv('fingerprints_label.csv', index=False, sep=',')

    # visualize structural similarity
    similarity_array = np.zeros(shape=(len(fps), len(fps)))

    # compare all fp pairwise without duplicates
    for n, fp in enumerate(fps): # -1 so the last fp will not be used
        s = DataStructs.BulkTanimotoSimilarity(fp, fps)
        similarity_array[n, :] = np.array(s)

    # print mean and std of the similarity array
    print('mean: ', np.mean(similarity_array))
    print('std: ', np.std(similarity_array))

    # plot the similarity array
    plt.figure(figsize=figsize, dpi=dpi)
    im = plt.imshow(similarity_array,cmap=cmap)
    plt.title(f'Tanimoto similarity {title}')
    plt.colorbar(im, spacing = 'uniform')
    plt.clim(0, 1)
    plt.savefig(f'fig_similarity_{filename}.png', dpi=dpi)
    plt.show()

    return df_label_final
    
def tanimoto_similarity_comparison(fp_list1, fp_list2, title=None, figsize=(12,4), dpi=300, filename='train_test', cmap='Blues'):
    """Plot tanimoto similarity matrix of fingerprints.
    
    Parameters
    ----------
    fp_list1 : list
        List of fingerprints.
    fp_list2 : list
        List of fingerprints.
    title : str, optional
        Title of the figure. Default is None.
    figsize : tuple, optional
        Size of the figure.
    dpi : int, optional
        Resolution of the figure. Default is 300.
    filename : str, optional
        Name of the file. The output file will be named fig_similarity_{filename}.png. Default is 'train_test'.
    cmap : str, optional
        Colormap. Default is 'Blues'.
    
    Returns
    -------
    None.
    """
	
	
	# Calculate the similarity matrix within fp_list1
    similarity_matrix1 = np.zeros((len(fp_list1), len(fp_list1)))  
    for i in range(len(fp_list1)):
        for j in range(len(fp_list1)):
            similarity_matrix1[i][j] = DataStructs.FingerprintSimilarity(fp_list1[i], fp_list1[j])
            
    # Calculate the similarity matrix within fp_list2
    similarity_matrix2 = np.zeros((len(fp_list2), len(fp_list2)))
    for i in range(len(fp_list2)):
        for j in range(len(fp_list2)):
            similarity_matrix2[i][j] = DataStructs.FingerprintSimilarity(fp_list2[i], fp_list2[j])
    
    # Calculate the similarity matrix between fp_list1 and fp_list2
    similarity_matrix = np.zeros((len(fp_list1), len(fp_list2)))
    for i in range(len(fp_list1)):
        for j in range(len(fp_list2)):
            similarity_matrix[i][j] = DataStructs.FingerprintSimilarity(fp_list1[i], fp_list2[j])

    # Plot the similarity matrix.
    fig, ax = plt.subplots(1,3, figsize=figsize, dpi=dpi)
    fig.suptitle(title, fontsize=16)
    
    ax[0].imshow(similarity_matrix1, aspect='1', cmap=cmap, interpolation='nearest')
    ax[0].set_title('Train set')

    ax[1].imshow(similarity_matrix2, aspect='1', cmap=cmap, interpolation='nearest')
    ax[1].set_title('Test set')

    ax[2].imshow(similarity_matrix, aspect='1', cmap=cmap, interpolation='nearest')
    ax[2].set_title('Train vs test set')
    ax[2].set_xlabel('Test set')
    ax[2].set_ylabel('Train set')
    
    # show colorbar
    fig.colorbar(ax[2].imshow(similarity_matrix, aspect='1', cmap=cmap, interpolation='nearest'), location='right')
    
    plt.tight_layout()
    plt.savefig(f'fig_similarity_{filename}.png', dpi=dpi)
    plt.show()
    
    return






def get_fingerprints_bit_importance(model_file, features_df, non_fingerprint_features=None, nBits=1024):
    """
    This function takes in a model file, a list of SMILES strings, and a dataframe of features. It returns a dataframe of the feature importance of each bit in the fingerprint.
    Parameters:
        model_file (str): the name of the model file. For example, 'model_ExtraTreesClassifier.sav'.
        features_df (dataframe): a dataframe of the features used to train the model.
        non_fingerprint_features (list): a list of the names of the features that are not morgan fingerprints. Default is None. If you used custom features, you need to specify the names of those features here.
        nBits (int): the number of bits used to generate the morgan fingerprints. Default is 1024. This should be the same as the nBits used to generate the features.
    Returns:
        df: a dataframe of the feature importance of each bit in the fingerprint.
    """
    model_name = model_file.split(".")[0].replace("model_", "")
    print('Model name is: ', model_name)

    # load the model from disk
    model = joblib.load(model_file)

    # extract the feature importances using the feature_importances_ attribute from sklearn
    try:
        importances = model.feature_importances_
    except:
        importances = model.coef_

    # sort the feature importances in descending order
    #indices = np.argsort(importances)[::-1]
    #print(indices)

    # get the names of the features from the original training data
    features = features_df.columns

    # create a list of feature names and their importance
    feature_importance = []
    for f in range(features_df.shape[1]):
        #feature_importance.append((features[indices[f]], importances[indices[f]]))
        feature_importance.append((features[f], importances[f]))
    df_feature_importance_all = pd.DataFrame(feature_importance, columns=["feature", "importance"])

    if non_fingerprint_features is None:
        df_feature_importance = df_feature_importance_all
    else:
        # remove features that are not morgan fingerprints (i.e. the custom features we added)
        #df_feature_importance = df_feature_importance_all[df_feature_importance_all["feature"].isin(non_fingerprint_features) == False]

        # rewrite this using pandas.concat
        df_feature_importance = pd.concat([df_feature_importance_all[df_feature_importance_all["feature"].isin(non_fingerprint_features) == False]])

    # for the bits that are all 0s, add them back to the dataframe with importance 0
    for i in range(nBits):
        if i not in df_feature_importance["feature"].values:
            #df_feature_importance = df_feature_importance.append({"feature": i, "importance": 0}, ignore_index=True)
            #df = df_feature_importance.append({"feature": i, "importance": 0}, ignore_index=True)
            #print(i)
            df_feature_importance = pd.concat([df_feature_importance, pd.DataFrame([{"feature": i, "importance": 0}])], ignore_index=True)

    df = df_feature_importance.sort_values(by=["feature"])
    print(len(df.feature))
    # save the feature importance to a csv file
    df.to_csv(model_name + "_feature_importance.csv", index=False)
    print("Feature importance saved to " + model_name + "_feature_importance.csv")

    return df


    


def plot_fingerprints_bit_importance(smiles, df_feature_importance, radius=2, nBits=1024, nCol=5, name=None, plot=False, show=False):
    """
    Plot the bit importance for a given list of smiles from a trained model. 
    NOTE: Currently, this only supports plotting bit info map of Morgan fingerprints.
    Parameters:
        smiles: list of smiles
        df_feature_importance: dataframe with the feature importance
        radius: radius of the Morgan fingerprint. Default: 2. This is the same as the radius used for training the model.
        nBits: number of bits of the Morgan fingerprint. Default: 1024. This is the same as the nBits used for training the model.
        nFeats: number of features to plot. Default: 1024. If nFeats >= nBits, then all the features are plotted. 
        nCol: number of molecules per row to plot for the fingerprint bits. Default: 5.
        name: directory to save the plots: "analysis_fingerprints_bits_importance_{name}"
        show: plot the plot. Default: False. If True, the plot is plotted.
        show: show the plot. Default: False. If True, the plot is shown.
    Returns:
        None
    """
    savedir="analysis_fingerprints_bits_importance_" + str(name) 
    if not os.path.exists(savedir):
        os.makedirs(savedir)
        print("Created directory {}".format(savedir))
    else:
        print("Directory {} already exists".format(savedir))

    # get the top nFeats features
    topfeats = df_feature_importance.index.tolist()
    topfeats_importance = df_feature_importance['importance'].tolist()
    topfeats_names = df_feature_importance['feature'].tolist()
    #print("N features: {}".format(len(topfeats)))

    # loop over the smiles
    n_mols = len(smiles)
    print("Plotting the bit importance for {} molecules".format(n_mols))
    
    for i in range(n_mols):
        smi = smiles[i]
        #print(smi)
        try: 
            mol = Chem.MolFromSmiles(smi)

            # get Morgan fingerprint with bitInfo from the mol object
            bitinfo={}
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits, useFeatures=True, bitInfo=bitinfo)
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            onbit = [bit for bit in bitinfo.keys()]

            # create a list with the bit info and the feature name
            importantonbits = list(set(onbit) & set(topfeats))

            # sort by importance
            importantonbits = sorted(importantonbits, key=lambda x:topfeats_importance[topfeats.index(x)], reverse=True)
            print(importantonbits)

            # get the bitId and the feature name
            tpls = [(mol, x, bitinfo) for x in importantonbits]


            topimportances = [topfeats_importance[x] for x in importantonbits]
            #print(topimportances)

            # legend shows both the bitId and the importance
            legend = ["{} ({:0.4f})".format(x[1], x[0]) for x in zip(topimportances, importantonbits)]


            if plot==True:
                # plot the molecule and the fingerprint bits with the importances
                fig, ax = plt.subplots(1, 2, dpi = 300, gridspec_kw={'width_ratios': [1, 6]}, figsize=(10, 6))
                fig.suptitle("Molecule {}".format(i))
                # draw the molecule on the left
                im = Draw.MolToImage(mol)
                ax[0].imshow(im)
                ax[0].axis("off")
                # draw the Morgan fingerprint with the bit info on the right
                p = Draw.DrawMorganBits(tpls, legends = legend, molsPerRow=nCol, subImgSize=(200,200))
                ax[1].imshow(p)
                ax[1].axis("off")

                plt.tight_layout()
                
                # save the figure as pdf to the savedir
                fig.savefig(os.path.join(savedir, "mol_{}.pdf".format(i)), dpi=300)
                print("Saved figure to {}".format(os.path.join(savedir, "mol_{}.pdf".format(i))))

                if show==True:
                    plt.show()              
                if show==False:
                    plt.close()
        except:
            print("Error with molecule {}".format(i))
            pass
